import cv2
from typing import Dict, Any
from pathlib import Path
from scenedetect import detect, ContentDetector

from ..config import KEYFRAMES_DIR, CROPPED_KEYFRAMES_DIR
from .grounding_dino import GroundingDinoService


class VideoProcessingService:
    def __init__(self):
        self.grounding_service = GroundingDinoService()
        self._video_counter = 0

    def get_next_video_id(self) -> str:
        """Generate next video ID"""
        self._video_counter += 1
        return f"video_{self._video_counter:04d}"

    def process_video(self, video_path: Path, video_id: str) -> Dict[str, Any]:
        """Process video to extract keyframes and generate crops"""
        keyframes_path = KEYFRAMES_DIR / video_id
        keyframes_path.mkdir(exist_ok=True)

        cropped_keyframes_path = CROPPED_KEYFRAMES_DIR / video_id
        cropped_keyframes_path.mkdir(exist_ok=True)

        # Scene detection
        scene_list = detect(str(video_path), ContentDetector())

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)

        keyframe_files = []
        all_cropped_files = []

        for i, scene in enumerate(scene_list):
            start_time = scene[0]
            frame_number = int(start_time.get_seconds() * fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()

            if ret:
                keyframe_filename = f"keyframe_{i:03d}.jpg"
                keyframe_path = keyframes_path / keyframe_filename
                cv2.imwrite(str(keyframe_path), frame)
                
                keyframe_files.append({
                    "filename": keyframe_filename,
                    "timecode": str(start_time),
                })

                # Process keyframe with Grounding DINO
                try:
                    cropped_files = self.grounding_service.process_keyframe(
                        keyframe_path, cropped_keyframes_path
                    )
                    all_cropped_files.extend(cropped_files)
                except Exception as e:
                    print(f"Failed to process keyframe {keyframe_filename} with Grounding DINO: {e}")

        cap.release()

        return {
            "scenes_detected": len(scene_list),
            "keyframe_files": keyframe_files,
            "cropped_files": all_cropped_files,
        }
