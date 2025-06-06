import os
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from scenedetect import detect, ContentDetector
import cv2

router = APIRouter(tags=["upload"])

UPLOAD_DIR = Path("/tmp/uploads")
KEYFRAMES_DIR = Path("/tmp/keyframes")

UPLOAD_DIR.mkdir(exist_ok=True)
KEYFRAMES_DIR.mkdir(exist_ok=True)

# Simple counter for video IDs
_video_counter = 0


def get_next_video_id():
    global _video_counter
    _video_counter += 1
    return f"video_{_video_counter:04d}"


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

    try:
        scene_list = detect(str(video_path), ContentDetector())

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)

        keyframe_files = []

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

        cap.release()

        os.remove(video_path)

        return {
            "video_id": video_id,
            "original_filename": file.filename,
            "scenes_detected": len(scene_list),
            "keyframes_generated": len(keyframe_files),
            "keyframes": keyframe_files,
            "keyframes_url": f"/upload/{video_id}/keyframes",
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
