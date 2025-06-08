import json
import logging
from fastapi import APIRouter, HTTPException, Depends, Query

from ..services import VideoProcessingService
from ..models import (
    CombinedVideoResponse,
    SimpleCropMatch,
    SimpleFashionMatch,
    ContentAnalysis,
    VibeAnalysis,
)

router = APIRouter(tags=["simplified_query"])

logger = logging.getLogger(__name__)


def get_video_service() -> VideoProcessingService:
    """Dependency to get video processing service"""
    return VideoProcessingService()


def get_query_service():
    """Dependency to get query service"""
    from ..services.query import QueryService

    return QueryService()


def get_match_quality(score: float) -> str:
    """Classify match quality based on similarity score"""
    if score > 0.9:
        return "Exact Match"
    elif score >= 0.75:
        return "Similar Match"
    else:
        return "No Match"


@router.get("/simplified_query", response_model=CombinedVideoResponse)
async def get_simplified_query_data(
    video_id: str = Query(..., description="Video ID to query"),
    video_service: VideoProcessingService = Depends(get_video_service),
    query_service=Depends(get_query_service),
):
    """Get simplified video data combining content analysis and fashion matches."""
    try:
        # Get cached upload data to find the original video_id and file info
        cached_data = None
        cache_files = list(video_service.cache_service.cache_dir.glob("*.json"))

        for cache_file in cache_files:
            try:
                with open(cache_file, "r") as f:
                    cache_content = json.load(f)
                    result = cache_content.get("result", {})

                    # Check if this is the video_id we're looking for
                    if result.get("video_id") == video_id:
                        cached_data = cache_content
                        break

                    # Check cache_info for cached results
                    cache_info = result.get("cache_info", {})
                    if cache_info.get("original_video_id"):
                        # This is a cached result, check if we can find the original
                        original_video_id = cache_info.get("original_video_id")
                        # Look for the original data by checking other cache files
                        for other_cache_file in cache_files:
                            try:
                                with open(other_cache_file, "r") as of:
                                    other_content = json.load(of)
                                    other_result = other_content.get("result", {})
                                    if other_result.get(
                                        "video_id"
                                    ) == original_video_id and not other_result.get(
                                        "cache_info", {}
                                    ).get("cached"):
                                        # Found the original, use it but update response video_id
                                        cached_data = other_content
                                        cached_data["result"]["requested_video_id"] = (
                                            video_id
                                        )
                                        break
                            except Exception:
                                continue
                        if cached_data:
                            break
            except Exception:
                continue

        if not cached_data:
            raise HTTPException(status_code=404, detail="Video data not found")

        result = cached_data["result"]
        original_filename = cached_data.get("original_filename", "")
        file_hash = cached_data.get("file_hash", "")
        requested_video_id = result.get("requested_video_id", video_id)

        # Determine the actual video_id to use for queries (original processing video_id)
        actual_video_id = result.get("video_id")
        cache_info = result.get("cache_info", {})
        original_video_id = (
            cache_info.get("original_video_id")
            if cache_info.get("cached")
            else actual_video_id
        )

        # Get content analysis
        content_analysis = None
        if result.get("content_analysis"):
            analysis_data = result["content_analysis"]
            vibes = [
                VibeAnalysis(
                    id=vibe["id"], name=vibe["name"], confidence=vibe["confidence"]
                )
                for vibe in analysis_data.get("vibes", [])
            ]
            content_analysis = ContentAnalysis(
                audio_transcription=analysis_data.get("audio_transcription"),
                clothing_description=analysis_data.get("clothing_description", ""),
                vibes=vibes,
            )

        # Get fashion matches from query service using the original video_id
        query_video_id = original_video_id or actual_video_id
        crops_data = await query_service.query_video_crops(query_video_id)

        # Transform crop matches to simplified format with filtering
        crop_matches = []
        for crop in crops_data:
            fashion_matches = crop.get("fashion_matches", [])

            # Filter matches above 0.75 score and get only the highest scoring one
            high_confidence_matches = [
                match for match in fashion_matches if match["score"] >= 0.75
            ]

            if high_confidence_matches:
                # Sort by score and take the highest one
                best_match = max(high_confidence_matches, key=lambda x: x["score"])
                payload = best_match.get("payload", {})

                simple_match = SimpleFashionMatch(
                    product_id=best_match["product_id"],
                    score=best_match["score"],
                    match_quality=get_match_quality(best_match["score"]),
                    product_name=payload.get("product_name", ""),
                    title=payload.get("title", ""),
                    description=payload.get("description", ""),
                    product_type=payload.get("product_type", ""),
                    price=payload.get("price", ""),
                    tags=payload.get("tags", ""),
                    collections=payload.get("collections", ""),
                )

                crop_match = SimpleCropMatch(
                    crop_id=crop["crop_id"],
                    filename=crop["filename"],
                    class_name=crop["class_name"],
                    original_class_name=crop["original_class_name"],
                    fashion_matches=[simple_match],  # Only one match now
                )
                crop_matches.append(crop_match)

        return CombinedVideoResponse(
            video_id=requested_video_id,  # Return the requested video_id
            original_filename=original_filename,
            file_hash=file_hash,
            cached=cache_info.get("cached", False),
            cached_from_video_id=original_video_id
            if cache_info.get("cached")
            else None,
            scenes_detected=result.get("scenes_detected", 0),
            keyframes_generated=len(result.get("keyframe_files", [])),
            cropped_keyframes_generated=len(result.get("cropped_files", [])),
            masked_keyframes_generated=len(result.get("masked_files", [])),
            content_analysis=content_analysis,
            total_crops_with_matches=len(crop_matches),
            crop_matches=crop_matches,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get simplified query data: {str(e)}"
        )
