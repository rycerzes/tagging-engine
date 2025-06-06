import os
import cv2
import torch
import numpy as np
import supervision as sv

from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images

"""
Hyperparam for Ground and Tracking
"""
MODEL_ID = "IDEA-Research/grounding-dino-tiny"
SAM2_MODEL = "facebook/sam2.1-hiera-small"
VIDEO_PATH = (
    "/root/flickd-ai/tagging-engine/data/raw/videos/2025-05-27_13-46-16_UTC.mp4"
)
TEXT_PROMPT = "watch. topwear. bottomwear. shoes. headwear."
OUTPUT_VIDEO_PATH = "/root/flickd-ai/tagging-engine/data/processed/sam2_gdino_video/sam2_gdino_video.mp4"
SOURCE_VIDEO_FRAME_DIR = (
    "/root/flickd-ai/tagging-engine/data/processed/sam2_gdino_video/custom_video_frames"
)
SAVE_TRACKING_RESULTS_DIR = (
    "/root/flickd-ai/tagging-engine/data/processed/sam2_gdino_video/tracking_results"
)
PROMPT_TYPE_FOR_VIDEO = "box"  # ["point", "box", "mask"]

"""
Step 1: Environment settings and model initialization for SAM 2
"""
device = "cuda" if torch.cuda.is_available() else "cpu"

# Enable optimizations for CUDA
if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Initialize SAM2 models using the new API

print(f"Loading SAM2 predictor: {SAM2_MODEL}")
image_predictor = SAM2ImagePredictor.from_pretrained(SAM2_MODEL)
video_predictor = SAM2VideoPredictor.from_pretrained(SAM2_MODEL)

# build grounding dino from huggingface
model_id = MODEL_ID
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
    device
)


"""
Custom video input directly using video files
"""
video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)  # get video info
print(video_info)
frame_generator = sv.get_video_frames_generator(VIDEO_PATH, stride=1, start=0, end=None)

# saving video to frames
source_frames = Path(SOURCE_VIDEO_FRAME_DIR)
source_frames.mkdir(parents=True, exist_ok=True)

with sv.ImageSink(
    target_dir_path=source_frames, overwrite=True, image_name_pattern="{:05d}.jpg"
) as sink:
    for frame in tqdm(frame_generator, desc="Saving Video Frames"):
        sink.save_image(frame)

# scan all the JPEG frame names in this directory
frame_names = [
    p
    for p in os.listdir(SOURCE_VIDEO_FRAME_DIR)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# init video predictor state
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    inference_state = video_predictor.init_state(video_path=SOURCE_VIDEO_FRAME_DIR)

    # Reset state to ensure clean inference (important for multiple object tracking)
    video_predictor.reset_state(inference_state)

    ann_frame_idx = 0  # the frame index we interact with
    """
    Step 2: Prompt Grounding DINO for box coordinates
    """

    # prompt grounding dino to get the box coordinates on specific frame
    img_path = os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[ann_frame_idx])
    image = Image.open(img_path)
    inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]],
    )

    input_boxes = results[0]["boxes"].cpu().numpy()
    confidences = results[0]["scores"].cpu().numpy().tolist()
    class_names = results[0]["text_labels"]

    print(f"Detected {len(input_boxes)} objects: {class_names}")
    print("Bounding boxes:", input_boxes)
    print("Confidences:", confidences)

    # prompt SAM image predictor to get the mask for the object
    image_predictor.set_image(np.array(image.convert("RGB")))

    # Create unique object identifiers for multiple instances of same class
    OBJECTS = []
    for i, (class_name, confidence) in enumerate(zip(class_names, confidences)):
        # Create unique identifier for each detection instance
        unique_name = f"{class_name}_{i+1}" if class_names.count(class_name) > 1 else class_name
        OBJECTS.append(unique_name)
    
    print(f"Unique object identifiers: {OBJECTS}")

    # process the detection results
    OBJECTS = class_names

    # prompt SAM 2 image predictor to get the mask for the object
    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    # convert the mask shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    """
    Step 3: Register multiple objects simultaneously to video predictor
    """

    assert PROMPT_TYPE_FOR_VIDEO in ["point", "box", "mask"], (
        "SAM 2 video predictor only support point/box/mask prompt"
    )

    # Store prompts for visualization and tracking
    prompts = {}  # hold all the clicks we add for visualization

    # Process multiple objects simultaneously
    if PROMPT_TYPE_FOR_VIDEO == "point":
        # sample the positive points from mask for each objects
        all_sample_points = sample_points_from_masks(masks=masks, num_points=10)

        # Add all objects with point prompts
        for object_id, (label, points) in enumerate(
            zip(OBJECTS, all_sample_points), start=1
        ):
            labels = np.ones((points.shape[0]), dtype=np.int32)
            prompts[object_id] = points, labels

            # Add each object to the video predictor
            frame_idx, out_obj_ids, out_mask_logits = (
                video_predictor.add_new_points_or_box(
                    inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    points=points,
                    labels=labels,
                )
            )
            print(f"Added object {object_id} ({label}) with {len(points)} points")

    elif PROMPT_TYPE_FOR_VIDEO == "box":
        # Add all objects with box prompts - ensure we process ALL detected boxes
        for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
            prompts[object_id] = box, None  # Store box for visualization

            # Add each object to the video predictor
            frame_idx, out_obj_ids, out_mask_logits = (
                video_predictor.add_new_points_or_box(
                    inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    box=box,
                )
            )
            print(f"Added object {object_id} ({label}) with box prompt")

    elif PROMPT_TYPE_FOR_VIDEO == "mask":
        # Add all objects with mask prompts - ensure we process ALL detected masks
        for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
            prompts[object_id] = mask, None  # Store mask for visualization

            # Add each object to the video predictor
            frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                inference_state, frame_idx=ann_frame_idx, obj_id=object_id, mask=mask
            )
            print(f"Added object {object_id} ({label}) with mask prompt")

    else:
        raise NotImplementedError(
            "SAM 2 video predictor only support point/box/mask prompts"
        )

    print(f"Successfully added {len(OBJECTS)} objects for tracking")

    """
    Step 4: Propagate the video predictor to get the segmentation results for each frame
    """
    print("Propagating segmentation across video frames...")
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for (
        out_frame_idx,
        out_obj_ids,
        out_mask_logits,
    ) in video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    print(f"Segmentation completed for {len(video_segments)} frames")

"""
Step 5: Visualize the segment results across the video and save them
"""

if not os.path.exists(SAVE_TRACKING_RESULTS_DIR):
    os.makedirs(SAVE_TRACKING_RESULTS_DIR)

ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}

for frame_idx, segments in video_segments.items():
    img = cv2.imread(os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[frame_idx]))

    object_ids = list(segments.keys())
    masks = list(segments.values())
    masks = np.concatenate(masks, axis=0)

    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
        mask=masks,  # (n, h, w)
        class_id=np.array(object_ids, dtype=np.int32),
    )
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(
        annotated_frame,
        detections=detections,
        labels=[ID_TO_OBJECTS[i] for i in object_ids],
    )
    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(
        scene=annotated_frame, detections=detections
    )
    cv2.imwrite(
        os.path.join(SAVE_TRACKING_RESULTS_DIR, f"annotated_frame_{frame_idx:05d}.jpg"),
        annotated_frame,
    )


"""
Step 6: Convert the annotated frames to video
"""

create_video_from_images(SAVE_TRACKING_RESULTS_DIR, OUTPUT_VIDEO_PATH)
