import os

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import FileResponse

from ..config import UPLOAD_DIR, KEYFRAMES_DIR, CROPPED_KEYFRAMES_DIR
from ..services import VideoProcessingService
from ..models import (
    UploadVideoResponse,
    KeyframeListResponse,
    CroppedKeyframeListResponse,
)

router = APIRouter(tags=["upload"])


def get_video_service() -> VideoProcessingService:
    """Dependency to get video processing service"""
    return VideoProcessingService()


@router.post("/upload", response_model=UploadVideoResponse)
async def upload_video(
    file: UploadFile = File(...),
    video_service: VideoProcessingService = Depends(get_video_service),
):
    """Upload an MP4 video and generate keyframes from scene detection."""
    if not file.filename.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only MP4 files are supported")

    video_id = video_service.get_next_video_id()
    video_name = f"{video_id}_{file.filename}"
    video_path = UPLOAD_DIR / video_name

    try:
        content = await file.read()
        with open(video_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {str(e)}")

    try:
        result = video_service.process_video(video_path, video_id)
        os.remove(video_path)

        return UploadVideoResponse(
            video_id=video_id,
            original_filename=file.filename,
            scenes_detected=result["scenes_detected"],
            keyframes_generated=len(result["keyframe_files"]),
            keyframes=result["keyframe_files"],
            keyframes_url=f"/upload/{video_id}/keyframes",
            cropped_keyframes_generated=len(result["cropped_files"]),
            cropped_keyframes=result["cropped_files"],
            cropped_keyframes_url=f"/upload/{video_id}/keyframes-cropped",
        )

    except Exception as e:
        if video_path.exists():
            os.remove(video_path)
        raise HTTPException(
            status_code=500, detail=f"Failed to process video: {str(e)}"
        )


@router.get("/upload/{video_id}/keyframes", response_model=KeyframeListResponse)
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

    return KeyframeListResponse(video_id=video_id, keyframes=keyframes)


@router.get("/upload/{video_id}/keyframes/{keyframe_name}")
async def get_keyframe(video_id: str, keyframe_name: str):
    """Serve a specific keyframe image."""
    keyframe_path = KEYFRAMES_DIR / video_id / keyframe_name

    if not keyframe_path.exists():
        raise HTTPException(status_code=404, detail="Keyframe not found")

    return FileResponse(keyframe_path, media_type="image/jpeg")


@router.get(
    "/upload/{video_id}/keyframes-cropped", response_model=CroppedKeyframeListResponse
)
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

    return CroppedKeyframeListResponse(
        video_id=video_id, cropped_keyframes=cropped_keyframes
    )


@router.get("/upload/{video_id}/keyframes-cropped/{cropped_keyframe_name}")
async def get_cropped_keyframe(video_id: str, cropped_keyframe_name: str):
    """Serve a specific cropped keyframe image."""
    cropped_keyframe_path = CROPPED_KEYFRAMES_DIR / video_id / cropped_keyframe_name

    if not cropped_keyframe_path.exists():
        raise HTTPException(status_code=404, detail="Cropped keyframe not found")

    return FileResponse(cropped_keyframe_path, media_type="image/png")
