import cv2
from typing import Dict, Any
from pathlib import Path
from scenedetect import detect, ContentDetector
from PIL import Image

from ..config import (
    KEYFRAMES_DIR,
    CROPPED_KEYFRAMES_DIR,
    MASKED_KEYFRAMES_DIR,
    ENABLE_MASKING,
    UPLOAD_DIR,
)
from .sam2_gdino import Sam2GroundingDinoService
from .deduplication import FaissDeduplicationService
from .cache import FileProcessingCache


class VideoProcessingService:
    def __init__(self):
        self.grounding_service = Sam2GroundingDinoService()
        self.deduplication_service = FaissDeduplicationService()
        self.cache_service = FileProcessingCache()
        self._counter_file = UPLOAD_DIR / "video_counter.txt"
        self._video_counter = self._load_counter()

    def _load_counter(self) -> int:
        """Load the persistent video counter from file"""
        try:
            if self._counter_file.exists():
                with open(self._counter_file, "r") as f:
                    return int(f.read().strip())
        except Exception:
            pass
        return 0

    def _save_counter(self) -> None:
        """Save the current counter to file"""
        try:
            with open(self._counter_file, "w") as f:
                f.write(str(self._video_counter))
        except Exception:
            pass

    def get_next_video_id(self) -> str:
        """Generate next video ID with persistent counter"""
        self._video_counter += 1
        self._save_counter()
        return f"video_{self._video_counter:04d}"

    def get_cached_result(self, file_hash: str) -> Dict[str, Any]:
        """Get cached processing result without incrementing counter"""
        cached_data = self.cache_service.get_cached_result(file_hash)
        if not cached_data:
            return None

        # Return cached data as-is without any modifications
        result = cached_data["result"].copy()

        # Mark as cached but keep original video_id
        result["cache_info"] = {
            "cached": True,
            "original_video_id": result.get("video_id"),
            "cached_at": cached_data["cached_at"],
            "file_hash": file_hash,
        }

        return result

    def get_response_video_id_for_cached_result(
        self, cached_result: Dict[str, Any]
    ) -> str:
        """Generate a response video ID for cached result WITHOUT incrementing the persistent counter"""
        # Use a temporary counter for response IDs that doesn't affect the persistent counter
        import time

        timestamp = int(time.time() * 1000) % 100000  # Use timestamp for uniqueness
        return f"cached_{timestamp:05d}"

    def store_processing_result(
        self, file_hash: str, result: Dict[str, Any], original_filename: str
    ) -> None:
        """Store processing result in cache"""
        self.cache_service.store_result(file_hash, result, original_filename)

    def process_video(
        self, video_path: Path, video_id: str, text_prompt: str = None
    ) -> Dict[str, Any]:
        """Process video to extract keyframes and generate crops"""
        keyframes_path = KEYFRAMES_DIR / video_id
        keyframes_path.mkdir(exist_ok=True)

        cropped_keyframes_path = CROPPED_KEYFRAMES_DIR / video_id
        cropped_keyframes_path.mkdir(exist_ok=True)

        masked_keyframes_path = MASKED_KEYFRAMES_DIR / video_id
        masked_keyframes_path.mkdir(exist_ok=True)

        # Scene detection
        scene_list = detect(str(video_path), ContentDetector())

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)

        keyframe_files = []
        all_cropped_files = []
        all_masked_files = []

        for i, scene in enumerate(scene_list):
            start_time = scene[0]
            frame_number = int(start_time.get_seconds() * fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()

            if ret:
                keyframe_filename = f"keyframe_{i:03d}.jpg"
                keyframe_path = keyframes_path / keyframe_filename
                cv2.imwrite(str(keyframe_path), frame)

                keyframe_files.append(
                    {
                        "filename": keyframe_filename,
                        "timecode": str(start_time),
                    }
                )

                # Process keyframe with Grounding DINO and SAM2
                try:
                    result = self.grounding_service.process_keyframe(
                        keyframe_path,
                        cropped_keyframes_path,
                        masked_keyframes_path,
                        text_prompt,
                    )
                    all_cropped_files.extend(result["cropped_files"])
                    all_masked_files.extend(result["masked_files"])
                except Exception as e:
                    print(
                        f"Failed to process keyframe {keyframe_filename} with Grounding DINO: {e}"
                    )

        cap.release()

        # Deduplicate cropped images using FAISS
        points_to_store = []
        if all_cropped_files:
            print("Starting deduplication of cropped images...")
            unique_cropped_files, removed_duplicates, points_to_store = (
                self.deduplication_service.deduplicate_crops(
                    cropped_keyframes_path, all_cropped_files, video_id
                )
            )
            all_cropped_files = unique_cropped_files

        return {
            "video_id": video_id,
            "scenes_detected": len(scene_list),
            "keyframe_files": keyframe_files,
            "cropped_files": all_cropped_files,
            "masked_files": all_masked_files,
            "points_to_store": points_to_store,
        }

    def process_image(
        self, image_path: Path, video_id: str, text_prompt: str = None
    ) -> Dict[str, Any]:
        """Process a single image to generate crops (treats image as single keyframe)"""
        keyframes_path = KEYFRAMES_DIR / video_id
        keyframes_path.mkdir(exist_ok=True)

        cropped_keyframes_path = CROPPED_KEYFRAMES_DIR / video_id
        cropped_keyframes_path.mkdir(exist_ok=True)

        masked_keyframes_path = MASKED_KEYFRAMES_DIR / video_id
        masked_keyframes_path.mkdir(exist_ok=True)

        # Copy image as keyframe
        keyframe_filename = "keyframe_000.jpg"
        keyframe_path = keyframes_path / keyframe_filename

        # Convert to RGB if needed and save as JPEG
        with Image.open(image_path) as img:
            if img.mode in ("RGBA", "LA"):
                # Convert RGBA/LA to RGB with white background
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "RGBA":
                    background.paste(img, mask=img.split()[-1])
                else:
                    background.paste(img, mask=img.split()[-1])
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")

            img.save(keyframe_path, "JPEG")

        keyframe_files = [
            {
                "filename": keyframe_filename,
                "timecode": "00:00:00.000",
            }
        ]

        all_cropped_files = []
        all_masked_files = []

        # Process keyframe with Grounding DINO and SAM2
        try:
            result = self.grounding_service.process_keyframe(
                keyframe_path,
                cropped_keyframes_path,
                masked_keyframes_path,
                text_prompt,
            )
            all_cropped_files.extend(result["cropped_files"])
            all_masked_files.extend(result["masked_files"])
        except Exception as e:
            print(f"Failed to process image with Grounding DINO: {e}")

        # Deduplicate cropped images using FAISS
        points_to_store = []
        if all_cropped_files:
            print("Starting deduplication of cropped images...")
            unique_cropped_files, removed_duplicates, points_to_store = (
                self.deduplication_service.deduplicate_crops(
                    cropped_keyframes_path, all_cropped_files, video_id
                )
            )
            all_cropped_files = unique_cropped_files

        return {
            "video_id": video_id,
            "scenes_detected": 1,  # Single image = 1 "scene"
            "keyframe_files": keyframe_files,
            "cropped_files": all_cropped_files,
            "masked_files": all_masked_files,
            "points_to_store": points_to_store,
        }
