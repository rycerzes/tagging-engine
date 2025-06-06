import os
import logging
import asyncio

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import FileResponse

from ..config import (
    UPLOAD_DIR,
    KEYFRAMES_DIR,
    CROPPED_KEYFRAMES_DIR,
    MASKED_KEYFRAMES_DIR,
    USE_GEMINI_FOR_TEXT_PROMPT,
    TEXT_PROMPT,
)
from ..services import VideoProcessingService, GeminiService
from ..models import (
    UploadVideoResponse,
    KeyframeListResponse,
    CroppedKeyframeListResponse,
    ContentAnalysis,
    VibeAnalysis,
)

router = APIRouter(tags=["upload"])

logger = logging.getLogger(__name__)


def get_video_service() -> VideoProcessingService:
    """Dependency to get video processing service"""
    return VideoProcessingService()


def get_gemini_service() -> GeminiService:
    """Dependency to get Gemini service"""
    return GeminiService()


@router.post("/upload", response_model=UploadVideoResponse)
async def upload_video(
    file: UploadFile = File(...),
    video_service: VideoProcessingService = Depends(get_video_service),
    gemini_service: GeminiService = Depends(get_gemini_service),
):
    """Upload an MP4 video or image (JPG/JPEG/PNG) and generate keyframes/crops."""
    file_extension = file.filename.lower().split('.')[-1]
    supported_extensions = ["mp4", "jpg", "jpeg", "png"]
    
    if file_extension not in supported_extensions:
        raise HTTPException(
            status_code=400, 
            detail="Only MP4 videos and JPG/JPEG/PNG images are supported"
        )

    video_id = video_service.get_next_video_id()
    file_name = f"{video_id}_{file.filename}"
    file_path = UPLOAD_DIR / file_name

    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    try:
        content_analysis = None
        
        # Generate dynamic text prompt using Gemini if enabled
        if USE_GEMINI_FOR_TEXT_PROMPT:
            logger.info("Using Gemini for text prompt generation")
            if file_extension == "mp4":
                text_prompt = gemini_service.generate_text_prompt_from_video(file_path)
            else:
                text_prompt = gemini_service.generate_text_prompt_from_image(file_path)
            
            # Process file based on type
            if file_extension == "mp4":
                result = video_service.process_video(file_path, video_id, text_prompt)
            else:
                result = video_service.process_image(file_path, video_id, text_prompt)
        else:
            logger.info("Using default text prompt and performing concurrent content analysis")
            text_prompt = TEXT_PROMPT
            
            # Run both processes concurrently
            is_video = file_extension == "mp4"
            
            # Create tasks for concurrent execution
            analysis_task = asyncio.create_task(
                gemini_service.analyze_content_async(file_path, is_video)
            )
            
            # Run video/image processing in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            if is_video:
                processing_task = loop.run_in_executor(
                    None, video_service.process_video, file_path, video_id, text_prompt
                )
            else:
                processing_task = loop.run_in_executor(
                    None, video_service.process_image, file_path, video_id, text_prompt
                )
            
            # Wait for both tasks to complete
            analysis_result, result = await asyncio.gather(analysis_task, processing_task)
            
            # Convert to ContentAnalysis model
            vibes = [
                VibeAnalysis(
                    id=vibe["id"],
                    name=vibe["name"],
                    confidence=vibe["confidence"]
                )
                for vibe in analysis_result.get("vibes", [])
            ]
            
            content_analysis = ContentAnalysis(
                audio_transcription=analysis_result.get("audio_transcription"),
                clothing_description=analysis_result.get("clothing_description", ""),
                vibes=vibes
            )
        
        os.remove(file_path)

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
            masked_keyframes_generated=len(result["masked_files"]),
            masked_keyframes=result["masked_files"],
            masked_keyframes_url=f"/upload/{video_id}/keyframes-masked",
            content_analysis=content_analysis,
        )

    except Exception as e:
        if file_path.exists():
            os.remove(file_path)
        raise HTTPException(
            status_code=500, detail=f"Failed to process file: {str(e)}"
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


@router.get(
    "/upload/{video_id}/keyframes-masked", response_model=CroppedKeyframeListResponse
)
async def list_masked_keyframes(video_id: str):
    """List all masked keyframes for a given video."""
    masked_keyframes_path = MASKED_KEYFRAMES_DIR / video_id

    if not masked_keyframes_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    masked_keyframes = []
    for masked_file in sorted(masked_keyframes_path.glob("*.png")):
        masked_keyframes.append(
            {
                "filename": masked_file.name,
                "url": f"/upload/{video_id}/keyframes-masked/{masked_file.name}",
            }
        )

    return CroppedKeyframeListResponse(
        video_id=video_id, cropped_keyframes=masked_keyframes
    )


@router.get("/upload/{video_id}/keyframes-masked/{masked_keyframe_name}")
async def get_masked_keyframe(video_id: str, masked_keyframe_name: str):
    """Serve a specific masked keyframe image."""
    masked_keyframe_path = MASKED_KEYFRAMES_DIR / video_id / masked_keyframe_name

    if not masked_keyframe_path.exists():
        raise HTTPException(status_code=404, detail="Masked keyframe not found")

    return FileResponse(masked_keyframe_path, media_type="image/png")
