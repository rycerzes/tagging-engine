import os
import torch
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from scenedetect import detect, ContentDetector
import cv2
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

router = APIRouter(tags=["upload"])

UPLOAD_DIR = Path("/tmp/uploads")
KEYFRAMES_DIR = Path("/tmp/keyframes")
CROPPED_KEYFRAMES_DIR = Path("/tmp/keyframes-cropped")

UPLOAD_DIR.mkdir(exist_ok=True)
KEYFRAMES_DIR.mkdir(exist_ok=True)
CROPPED_KEYFRAMES_DIR.mkdir(exist_ok=True)

# Simple counter for video IDs
_video_counter = 0

# Grounding DINO configuration
GROUNDING_MODEL = "IDEA-Research/grounding-dino-tiny"
TEXT_PROMPT = "watch. topwear. bottomwear. shoes. headgear."
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize models (lazy loading)
_grounding_processor = None
_grounding_model = None


def get_next_video_id():
    global _video_counter
    _video_counter += 1
    return f"video_{_video_counter:04d}"


def get_models():
    """Lazy load Grounding DINO model"""
    global _grounding_processor, _grounding_model

    if _grounding_processor is None or _grounding_model is None:
        print(f"Loading Grounding DINO model: {GROUNDING_MODEL}")
        _grounding_processor = AutoProcessor.from_pretrained(GROUNDING_MODEL)
        _grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            GROUNDING_MODEL,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(DEVICE)

    return _grounding_processor, _grounding_model


def crop_image_with_bbox(image, bbox):
    """Crop image using bounding box"""
    x1, y1, x2, y2 = map(int, bbox)
    return image.crop((x1, y1, x2, y2))


def process_keyframe_with_grounding_dino(keyframe_path, output_dir):
    """Process a keyframe with Grounding DINO to generate bounding box crops"""
    grounding_processor, grounding_model = get_models()

    image = Image.open(keyframe_path)

    # Grounding DINO inference
    grounding_inputs = grounding_processor(
        images=image, text=TEXT_PROMPT, return_tensors="pt"
    ).to(DEVICE)

    device_type = "cuda" if DEVICE == "cuda" else "cpu"
    with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.float16):
        grounding_outputs = grounding_model(**grounding_inputs)

    grounding_results = grounding_processor.post_process_grounded_object_detection(
        grounding_outputs,
        grounding_inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]],
    )

    if not grounding_results[0]["boxes"].numel():
        return []

    input_boxes = grounding_results[0]["boxes"].cpu().numpy()
    class_names = grounding_results[0]["text_labels"]

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

    for i, (original_class_name, unique_class_name, bbox) in enumerate(
        zip(class_names, unique_class_names, input_boxes)
    ):
        cropped_img = crop_image_with_bbox(image, bbox)

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

    # Cleanup
    del grounding_inputs, grounding_outputs
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return cropped_files


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload an MP4 video and generate keyframes from scene detection."""

    if not file.filename.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only MP4 files are supported")

    video_id = get_next_video_id()
    video_name = f"{video_id}_{file.filename}"
    video_path = UPLOAD_DIR / video_name

    try:
        content = await file.read()
        with open(video_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {str(e)}")

    keyframes_path = KEYFRAMES_DIR / video_id
    keyframes_path.mkdir(exist_ok=True)

    cropped_keyframes_path = CROPPED_KEYFRAMES_DIR / video_id
    cropped_keyframes_path.mkdir(exist_ok=True)

    try:
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
                keyframe_files.append(
                    {
                        "filename": keyframe_filename,
                        "timecode": str(start_time),
                    }
                )

                # Process keyframe with Grounding DINO
                try:
                    cropped_files = process_keyframe_with_grounding_dino(
                        keyframe_path, cropped_keyframes_path
                    )
                    all_cropped_files.extend(cropped_files)
                except Exception as e:
                    print(
                        f"Failed to process keyframe {keyframe_filename} with Grounding DINO: {e}"
                    )

        cap.release()
        os.remove(video_path)

        return {
            "video_id": video_id,
            "original_filename": file.filename,
            "scenes_detected": len(scene_list),
            "keyframes_generated": len(keyframe_files),
            "keyframes": keyframe_files,
            "keyframes_url": f"/upload/{video_id}/keyframes",
            "cropped_keyframes_generated": len(all_cropped_files),
            "cropped_keyframes": all_cropped_files,
            "cropped_keyframes_url": f"/upload/{video_id}/keyframes-cropped",
        }

    except Exception as e:
        if video_path.exists():
            os.remove(video_path)
        raise HTTPException(
            status_code=500, detail=f"Failed to process video: {str(e)}"
        )


@router.get("/upload/{video_id}/keyframes")
async def list_keyframes(video_id: str):
    """List all keyframes for a given video."""
    keyframes_path = KEYFRAMES_DIR / video_id

    if not keyframes_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    keyframes = []
    for keyframe_file in sorted(keyframes_path.glob("*.jpg")):
        frame_num = int(keyframe_file.stem.split("_")[1])
        keyframes.append(
            {
                "filename": keyframe_file.name,
                "frame_number": frame_num,
                "url": f"/upload/{video_id}/keyframes/{keyframe_file.name}",
            }
        )

    return {"video_id": video_id, "keyframes": keyframes}


@router.get("/upload/{video_id}/keyframes/{keyframe_name}")
async def get_keyframe(video_id: str, keyframe_name: str):
    """Serve a specific keyframe image."""
    keyframe_path = KEYFRAMES_DIR / video_id / keyframe_name

    if not keyframe_path.exists():
        raise HTTPException(status_code=404, detail="Keyframe not found")

    return FileResponse(keyframe_path, media_type="image/jpeg")


@router.get("/upload/{video_id}/keyframes-cropped")
async def list_cropped_keyframes(video_id: str):
    """List all cropped keyframes for a given video."""
    cropped_keyframes_path = CROPPED_KEYFRAMES_DIR / video_id

    if not cropped_keyframes_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    cropped_keyframes = []
    for cropped_file in sorted(cropped_keyframes_path.glob("*.png")):
        cropped_keyframes.append(
            {
                "filename": cropped_file.name,
                "url": f"/upload/{video_id}/keyframes-cropped/{cropped_file.name}",
            }
        )

    return {"video_id": video_id, "cropped_keyframes": cropped_keyframes}


@router.get("/upload/{video_id}/keyframes-cropped/{cropped_keyframe_name}")
async def get_cropped_keyframe(video_id: str, cropped_keyframe_name: str):
    """Serve a specific cropped keyframe image."""
    cropped_keyframe_path = CROPPED_KEYFRAMES_DIR / video_id / cropped_keyframe_name

    if not cropped_keyframe_path.exists():
        raise HTTPException(status_code=404, detail="Cropped keyframe not found")

    return FileResponse(cropped_keyframe_path, media_type="image/png")
