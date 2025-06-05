import argparse
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
from sam2.sam2_video_predictor import SAM2VideoPredictor

"""
Hyper parameters
"""
parser = argparse.ArgumentParser()
parser.add_argument("--grounding-model", default="IDEA-Research/grounding-dino-tiny")
parser.add_argument("--sam2-model", default="facebook/sam2-hiera-base-plus")
parser.add_argument("--text-prompt", default="clothing. accessories.")
parser.add_argument(
    "--video-path",
    default="/root/flickd-ai/tagging-engine/data/raw/videos/2025-05-27_13-46-16_UTC.mp4",
)
parser.add_argument("--output-dir", default="/root/flickd-ai/tagging-engine/data/processed/grounded_sam2_video_demo")
parser.add_argument("--no-dump-json", action="store_true")
parser.add_argument("--force-cpu", action="store_true")
parser.add_argument("--max-frames", type=int, default=100, help="Maximum number of frames to process")
parser.add_argument("--frame-stride", type=int, default=1, help="Process every nth frame")
args = parser.parse_args()

GROUNDING_MODEL = args.grounding_model
SAM2_MODEL = args.sam2_model
TEXT_PROMPT = args.text_prompt
VIDEO_PATH = args.video_path
DEVICE = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
OUTPUT_DIR = Path(args.output_dir)
DUMP_JSON_RESULTS = not args.no_dump_json
MAX_FRAMES = args.max_frames
FRAME_STRIDE = args.frame_stride

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "frames").mkdir(exist_ok=True)

# mixed precision for memory efficiency
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)

def extract_frames(video_path, output_dir, max_frames=None, stride=1):
    """Extract frames from video"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    extracted_count = 0
    
    print(f"Extracting frames from {video_path}...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % stride == 0:
            frame_path = output_dir / "frames" / f"frame_{extracted_count:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frames.append(str(frame_path))
            extracted_count += 1
            
            if max_frames and extracted_count >= max_frames:
                break
                
        frame_count += 1
    
    cap.release()
    print(f"Extracted {len(frames)} frames")
    return frames

def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

# Extract frames from video
frame_paths = extract_frames(VIDEO_PATH, OUTPUT_DIR, MAX_FRAMES, FRAME_STRIDE)

if not frame_paths:
    print("No frames extracted from video")
    exit(1)

# SAM2 video model
print(f"Loading SAM2 video model: {SAM2_MODEL}")
sam2_predictor = SAM2VideoPredictor.from_pretrained(SAM2_MODEL, device=DEVICE)

# Grounding DINO tiny w memory optimization
print(f"Loading Grounding DINO model: {GROUNDING_MODEL}")
grounding_processor = AutoProcessor.from_pretrained(GROUNDING_MODEL)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    GROUNDING_MODEL,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(DEVICE)

# Use first frame to get initial detections with Grounding DINO
first_frame_path = frame_paths[0]
first_frame = Image.open(first_frame_path)

print("Running Grounding DINO inference on first frame...")
grounding_inputs = grounding_processor(images=first_frame, text=TEXT_PROMPT, return_tensors="pt").to(DEVICE)

with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=torch.float16):
    grounding_outputs = grounding_model(**grounding_inputs)

# Post-process Grounding DINO results
grounding_results = grounding_processor.post_process_grounded_object_detection(
    grounding_outputs,
    grounding_inputs.input_ids,
    box_threshold=0.4,
    text_threshold=0.3,
    target_sizes=[first_frame.size[::-1]],
)

del grounding_inputs, grounding_outputs
torch.cuda.empty_cache() if DEVICE == "cuda" else None

# Get detection results
input_boxes = grounding_results[0]["boxes"].cpu().numpy()
confidences = grounding_results[0]["scores"].cpu().numpy().tolist()
class_names = grounding_results[0]["text_labels"]

print(f"Found {len(class_names)} objects in first frame: {class_names}")

# Initialize SAM2 video predictor
print("Initializing SAM2 video predictor...")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    # Create temporary directory with frames for SAM2
    temp_frame_dir = OUTPUT_DIR / "temp_frames"
    temp_frame_dir.mkdir(exist_ok=True)
    
    # Copy frames to temporary directory with sequential naming
    for i, frame_path in enumerate(frame_paths):
        temp_frame_path = temp_frame_dir / f"{i:06d}.jpg"
        import shutil
        shutil.copy2(frame_path, temp_frame_path)
    
    # Initialize video state
    inference_state = sam2_predictor.init_state(video_path=str(temp_frame_dir))
    
    # Add prompts on first frame (frame 0)
    frame_idx = 0
    
    # Convert boxes to points (center of each box) and add each object separately
    all_object_ids = []
    
    for i, box in enumerate(input_boxes):
        # Use center of bounding box as positive point
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        points = np.array([[center_x, center_y]])
        point_labels = np.array([1])  # positive point
        
        # Add prompts to first frame for each object
        _, out_obj_ids, out_mask_logits = sam2_predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=i,
            points=points,
            labels=point_labels,
        )
        all_object_ids.extend(out_obj_ids)
    
    print("Propagating masks across video frames...")
    
    # Store all results
    all_results = []
    
    # Propagate masks through video
    for frame_idx, object_ids, masks in sam2_predictor.propagate_in_video(inference_state):
        print(f"Processing frame {frame_idx + 1}/{len(frame_paths)}")
        
        # Load current frame
        current_frame_path = frame_paths[frame_idx]
        current_frame = cv2.imread(current_frame_path)
        
        if masks is not None and len(masks) > 0:
            # Convert masks to numpy array and fix shape
            masks_np = masks.cpu().numpy()
            
            # Remove extra dimension if present (squeeze dimension 1)
            if masks_np.ndim == 4:
                masks_np = masks_np.squeeze(1)
            
            # Ensure we have the right number of bounding boxes for the masks
            num_masks = len(masks_np)
            current_boxes = input_boxes[:num_masks] if num_masks <= len(input_boxes) else input_boxes
            current_class_names = class_names[:num_masks] if num_masks <= len(class_names) else class_names
            current_confidences = confidences[:num_masks] if num_masks <= len(confidences) else confidences
            
            # Create supervision detections
            detections = sv.Detections(
                xyxy=current_boxes,
                mask=masks_np.astype(bool),
                class_id=np.array(object_ids)
            )
            
            # Create labels based on actual object IDs
            labels = []
            for obj_id in object_ids:
                if obj_id < len(current_class_names):
                    class_name = current_class_names[obj_id]
                    confidence = current_confidences[obj_id]
                    labels.append(f"{class_name} {confidence:.2f}")
                else:
                    labels.append(f"object_{obj_id}")
            
            # Annotate frame
            annotated_frame = current_frame.copy()
            
            # Add bounding boxes
            box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
            
            # Add labels
            label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )
            
            # Add masks
            mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            
            # Save annotated frame
            output_frame_path = OUTPUT_DIR / f"annotated_frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(output_frame_path), annotated_frame)
            
            # Store results for JSON
            if DUMP_JSON_RESULTS:
                frame_results = {
                    "frame_idx": frame_idx,
                    "frame_path": current_frame_path,
                    "annotations": []
                }
                
                for i, (obj_id, mask) in enumerate(zip(object_ids, masks_np)):
                    mask_rle = single_mask_to_rle(mask)
                    
                    # Use correct class name and bbox for this object
                    class_name = current_class_names[min(obj_id, len(current_class_names)-1)]
                    bbox = current_boxes[min(i, len(current_boxes)-1)].tolist()
                    score = current_confidences[min(obj_id, len(current_confidences)-1)]
                    
                    frame_results["annotations"].append({
                        "object_id": int(obj_id),
                        "class_name": class_name,
                        "bbox": bbox,
                        "segmentation": mask_rle,
                        "score": score,
                    })
                
                all_results.append(frame_results)

    # Clean up temporary directory
    import shutil
    shutil.rmtree(temp_frame_dir)

# Clear memory
torch.cuda.empty_cache() if DEVICE == "cuda" else None

# Save JSON results
if DUMP_JSON_RESULTS and all_results:
    print("Saving results to JSON...")
    
    # Get video info
    cap = cv2.VideoCapture(VIDEO_PATH)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    results = {
        "video_path": VIDEO_PATH,
        "video_info": {
            "width": video_width,
            "height": video_height,
            "fps": fps,
            "total_frames": total_frames,
            "processed_frames": len(all_results)
        },
        "grounding_prompt": TEXT_PROMPT,
        "detected_classes": class_names,
        "box_format": "xyxy",
        "frames": all_results
    }
    
    with open(OUTPUT_DIR / "grounded_sam2_video_results.json", "w") as f:
        json.dump(results, f, indent=2)

print(f"Results saved to {OUTPUT_DIR}")
print(f"Processed {len(frame_paths)} frames")
print(f"Found {len(class_names)} object classes: {class_names}")
