import torch
from typing import List, Dict, Any, Tuple
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from ..config import GROUNDING_MODEL, TEXT_PROMPT, DEVICE, BOX_THRESHOLD, TEXT_THRESHOLD


class GroundingDinoService:
    def __init__(self):
        self._processor = None
        self._model = None

    def _load_models(self) -> Tuple[AutoProcessor, AutoModelForZeroShotObjectDetection]:
        """Lazy load Grounding DINO model"""
        if self._processor is None or self._model is None:
            print(f"Loading Grounding DINO model: {GROUNDING_MODEL}")
            self._processor = AutoProcessor.from_pretrained(GROUNDING_MODEL)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                GROUNDING_MODEL,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).to(DEVICE)

        return self._processor, self._model

    def _crop_image_with_bbox(self, image: Image.Image, bbox: List[float]) -> Image.Image:
        """Crop image using bounding box"""
        x1, y1, x2, y2 = map(int, bbox)
        return image.crop((x1, y1, x2, y2))

    def process_keyframe(self, keyframe_path: Path, output_dir: Path) -> List[Dict[str, Any]]:
        """Process a keyframe with Grounding DINO to generate bounding box crops"""
        processor, model = self._load_models()

        image = Image.open(keyframe_path)

        # Grounding DINO inference
        inputs = processor(
            images=image, text=TEXT_PROMPT, return_tensors="pt"
        ).to(DEVICE)

        device_type = "cuda" if DEVICE == "cuda" else "cpu"
        with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.float16):
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[image.size[::-1]],
        )

        if not results[0]["boxes"].numel():
            return []

        input_boxes = results[0]["boxes"].cpu().numpy()
        class_names = results[0]["text_labels"]

        # Track counts for each class to create unique names
        class_counts = {}
        unique_class_names = []

        for class_name in class_names:
            if class_name in class_counts:
                class_counts[class_name] += 1
                unique_name = f"{class_name}_{class_counts[class_name]}"
            else:
                class_counts[class_name] = 1
                unique_name = f"{class_name}_1"
            unique_class_names.append(unique_name)

        # Generate cropped images
        cropped_files = []
        keyframe_stem = keyframe_path.stem

        for original_class_name, unique_class_name, bbox in zip(
            class_names, unique_class_names, input_boxes
        ):
            cropped_img = self._crop_image_with_bbox(image, bbox)

            crop_filename = f"{unique_class_name.replace(' ', '_')}-{keyframe_stem}.png"
            crop_path = output_dir / crop_filename
            cropped_img.save(crop_path)

            cropped_files.append({
                "filename": crop_filename,
                "class_name": unique_class_name,
                "original_class_name": original_class_name,
                "bbox": bbox.tolist(),
            })

        # Cleanup
        del inputs, outputs
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        return cropped_files
