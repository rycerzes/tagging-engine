import argparse
import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from supervision.draw.color import ColorPalette
from supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sam2.sam2_image_predictor import SAM2ImagePredictor

"""
Hyper parameters
"""
parser = argparse.ArgumentParser()
parser.add_argument("--grounding-model", default="IDEA-Research/grounding-dino-tiny")
parser.add_argument("--sam2-model", default="facebook/sam2-hiera-base-plus")
parser.add_argument("--text-prompt", default="watch. topwear. bottomwear. shoes. headgear.")
parser.add_argument(
    "--img-path",
    default="/root/flickd-ai/tagging-engine/data/raw/downloaded_images/product_15824/15824_000.jpg",
)
parser.add_argument("--output-dir", default="/root/flickd-ai/tagging-engine/data/processed/grounded_sam2_hf_demo")
parser.add_argument("--no-dump-json", action="store_true")
parser.add_argument("--force-cpu", action="store_true")
args = parser.parse_args()

GROUNDING_MODEL = args.grounding_model
SAM2_MODEL = args.sam2_model
TEXT_PROMPT = args.text_prompt
IMG_PATH = args.img_path
DEVICE = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
OUTPUT_DIR = Path(args.output_dir)
DUMP_JSON_RESULTS = not args.no_dump_json

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# mixed precision for memory efficiency
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)

# SAM2 model
print(f"Loading SAM2 model: {SAM2_MODEL}")
sam2_predictor = SAM2ImagePredictor.from_pretrained(SAM2_MODEL, device=DEVICE)

# Grounding DINO tiny w memory optimization
print(f"Loading Grounding DINO model: {GROUNDING_MODEL}")
grounding_processor = AutoProcessor.from_pretrained(GROUNDING_MODEL)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    GROUNDING_MODEL,
    torch_dtype=torch.float16,  # Use FP16 for memory efficiency
    low_cpu_mem_usage=True,  # Reduce CPU memory during loading
).to(DEVICE)

text = TEXT_PROMPT
img_path = IMG_PATH
image = Image.open(img_path)

print("Running Grounding DINO inference...")
# Process with Grounding DINO using mixed precision
grounding_inputs = grounding_processor(images=image, text=text, return_tensors="pt").to(
    DEVICE
)
with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=torch.float16):
    grounding_outputs = grounding_model(**grounding_inputs)

# Post-process Grounding DINO results
grounding_results = grounding_processor.post_process_grounded_object_detection(
    grounding_outputs,
    grounding_inputs.input_ids,
    box_threshold=0.4,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]],
)

del grounding_inputs, grounding_outputs
torch.cuda.empty_cache() if DEVICE == "cuda" else None

print("Running SAM2 inference...")
# Prepare inputs for SAM2
input_boxes = grounding_results[0]["boxes"].cpu().numpy()

# Set image for SAM2 predictor
sam2_predictor.set_image(np.array(image))

# Get masks from SAM2
with torch.no_grad():
    masks, scores, _ = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

# Clear memory
torch.cuda.empty_cache() if DEVICE == "cuda" else None

# Convert masks shape if needed
if masks.ndim == 4:
    masks = masks.squeeze(1)

# Prepare detection results
confidences = grounding_results[0]["scores"].cpu().numpy().tolist()
class_names = grounding_results[0]["text_labels"]
class_ids = np.array(list(range(len(class_names))))

labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence in zip(class_names, confidences)
]


def crop_image_with_mask(image, mask, bbox):
    """
    Crop image using mask and bounding box
    Returns RGBA image with transparent background
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    cropped_img = image.crop((x1, y1, x2, y2))
    
    cropped_mask = mask[y1:y2, x1:x2]
    
    cropped_img = cropped_img.convert("RGBA")
    
    img_array = np.array(cropped_img)
    img_array[:, :, 3] = cropped_mask * 255  # Set alpha channel
    
    return Image.fromarray(img_array)


print("Creating visualizations...")
# Visualize results
img = cv2.imread(img_path)
detections = sv.Detections(
    xyxy=input_boxes, mask=masks.astype(bool), class_id=class_ids
)

print("Generating cropped images...")
for i, (class_name, bbox, mask) in enumerate(zip(class_names, input_boxes, masks)):
    cropped_img = crop_image_with_mask(image, mask, bbox)
    
    crop_filename = f"crop_{i}_{class_name.replace(' ', '_')}.png"
    crop_path = OUTPUT_DIR / crop_filename
    cropped_img.save(crop_path)
    print(f"Saved cropped image: {crop_filename}")

# Annotate image
box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
annotated_frame = label_annotator.annotate(
    scene=annotated_frame, detections=detections, labels=labels
)
cv2.imwrite(
    os.path.join(OUTPUT_DIR, "groundingdino_annotated_image.jpg"), annotated_frame
)

mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
cv2.imwrite(
    os.path.join(OUTPUT_DIR, "grounded_sam2_annotated_image_with_mask.jpg"),
    annotated_frame,
)


def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def crop_image_with_mask(image, mask, bbox):
    """
    Crop image using mask and bounding box
    Returns RGBA image with transparent background
    """
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


if DUMP_JSON_RESULTS:
    print("Saving results to JSON...")
    # Convert mask into rle format
    mask_rles = [single_mask_to_rle(mask) for mask in masks]

    input_boxes = input_boxes.tolist()
    scores = scores.tolist()

    # Save results in standard format
    results = {
        "image_path": img_path,
        "annotations": [
            {
                "class_name": class_name,
                "bbox": box,
                "segmentation": mask_rle,
                "score": score,
            }
            for class_name, box, mask_rle, score in zip(
                class_names, input_boxes, mask_rles, scores
            )
        ],
        "box_format": "xyxy",
        "img_width": image.width,
        "img_height": image.height,
    }

    with open(
        os.path.join(OUTPUT_DIR, "grounded_sam2_hf_model_demo_results.json"), "w"
    ) as f:
        json.dump(results, f, indent=4)

print(f"Results saved to {OUTPUT_DIR}")
print(f"Found {len(class_names)} objects: {class_names}")
