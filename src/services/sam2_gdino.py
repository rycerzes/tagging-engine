import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sam2.sam2_image_predictor import SAM2ImagePredictor

from ..config import (
    GROUNDING_MODEL,
    SAM2_MODEL,
    TEXT_PROMPT,
    DEVICE,
    BOX_THRESHOLD,
    TEXT_THRESHOLD,
    ENABLE_MASKING,
)


class Sam2GroundingDinoService:
    def __init__(self):
        self._processor = None
        self._model = None
        self._sam2_predictor = None

    def _load_models(
        self,
    ) -> Tuple[AutoProcessor, AutoModelForZeroShotObjectDetection, SAM2ImagePredictor]:
        """Lazy load Grounding DINO and SAM2 models"""
        if self._processor is None or self._model is None:
            print(f"Loading Grounding DINO model: {GROUNDING_MODEL}")
            self._processor = AutoProcessor.from_pretrained(GROUNDING_MODEL)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                GROUNDING_MODEL,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).to(DEVICE)

        # Only load SAM2 if masking is enabled
        if ENABLE_MASKING and self._sam2_predictor is None:
            print(f"Loading SAM2 model: {SAM2_MODEL}")
            self._sam2_predictor = SAM2ImagePredictor.from_pretrained(
                SAM2_MODEL, device=DEVICE
            )

        return self._processor, self._model, self._sam2_predictor

    def _crop_image_with_bbox(
        self, image: Image.Image, bbox: List[float]
    ) -> Image.Image:
        """Crop image using bounding box"""
        x1, y1, x2, y2 = map(int, bbox)
        return image.crop((x1, y1, x2, y2))

    def _crop_image_with_mask(
        self, image: Image.Image, mask: np.ndarray, bbox: List[float]
    ) -> Image.Image:
        """Crop image using mask and bounding box, returns RGBA image with transparent background"""
        x1, y1, x2, y2 = map(int, bbox)

        # Crop the original image to bounding box
        cropped_img = image.crop((x1, y1, x2, y2))

        # Crop the mask to the same bounding box
        cropped_mask = mask[y1:y2, x1:x2]

        # Convert to RGBA
        cropped_img = cropped_img.convert("RGBA")

        # Apply mask - set alpha channel based on mask
        img_array = np.array(cropped_img)
        img_array[:, :, 3] = cropped_mask * 255  # Set alpha channel

        return Image.fromarray(img_array)

    def process_keyframe(
        self, keyframe_path: Path, output_dir: Path, masked_output_dir: Path, text_prompt: str = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Process a keyframe with Grounding DINO and SAM2 to generate both regular and masked crops"""
        processor, model, sam2_predictor = self._load_models()

        prompt_to_use = text_prompt or TEXT_PROMPT

        image = Image.open(keyframe_path)

        # Grounding DINO inference
        inputs = processor(images=image, text=prompt_to_use, return_tensors="pt").to(
            DEVICE
        )

        device_type = "cuda" if DEVICE == "cuda" else "cpu"
        with (
            torch.no_grad(),
            torch.autocast(device_type=device_type, dtype=torch.float16),
        ):
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[image.size[::-1]],
        )

        if not results[0]["boxes"].numel():
            return {"cropped_files": [], "masked_files": []}

        input_boxes = results[0]["boxes"].cpu().numpy()
        class_names = results[0]["text_labels"]

        # SAM2 inference for masks (only if masking is enabled)
        masks = None
        if ENABLE_MASKING:
            if sam2_predictor is None:
                raise RuntimeError("SAM2 predictor not loaded but masking is enabled")
            sam2_predictor.set_image(np.array(image))
            with torch.no_grad():
                masks, scores, _ = sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )

            # Convert masks shape if needed
            if masks.ndim == 4:
                masks = masks.squeeze(1)

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
        masked_files = []
        keyframe_stem = keyframe_path.stem

        for i, (original_class_name, unique_class_name, bbox) in enumerate(
            zip(class_names, unique_class_names, input_boxes)
        ):
            # Regular crop
            cropped_img = self._crop_image_with_bbox(image, bbox)
            crop_filename = f"{unique_class_name.replace(' ', '_')}-{keyframe_stem}.png"
            crop_path = output_dir / crop_filename
            cropped_img.save(crop_path)

            cropped_files.append(
                {
                    "filename": crop_filename,
                    "class_name": unique_class_name,
                    "original_class_name": original_class_name,
                    "bbox": bbox.tolist(),
                }
            )

            # Masked crop (only if masking is enabled)
            if ENABLE_MASKING and masks is not None:
                mask = masks[i]
                masked_img = self._crop_image_with_mask(image, mask, bbox)
                masked_filename = (
                    f"{unique_class_name.replace(' ', '_')}-{keyframe_stem}-masked.png"
                )
                masked_path = masked_output_dir / masked_filename
                masked_img.save(masked_path)

                masked_files.append(
                    {
                        "filename": masked_filename,
                        "class_name": unique_class_name,
                        "original_class_name": original_class_name,
                        "bbox": bbox.tolist(),
                    }
                )

        # Cleanup
        del inputs, outputs
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        return {"cropped_files": cropped_files, "masked_files": masked_files}
