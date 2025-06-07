from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import asyncio

from ..services.query import QueryService
from ..models import (
    VideoQueryResponse,
    CropMatch,
    FashionMatch,
)

router = APIRouter(tags=["query"])


def get_query_service() -> QueryService:
    """Dependency to get query service"""
    return QueryService()


@router.get("/query", response_model=VideoQueryResponse)
async def query_video_crops(
    video_id: str,
    query_service: QueryService = Depends(get_query_service),
):
    """Query all crops for a video and their fashion product matches."""
    try:
        crops_data = await query_service.query_video_crops(video_id)
        
        crop_matches = []
        for crop in crops_data:
            fashion_matches = [
                FashionMatch(
                    product_id=match["product_id"],
                    score=match["score"],
                    payload=match["payload"]
                )
                for match in crop.get("fashion_matches", [])
            ]
            
            crop_match = CropMatch(
                crop_id=crop["crop_id"],
                filename=crop["filename"],
                class_name=crop["class_name"],
                original_class_name=crop["original_class_name"],
                bbox=crop["bbox"],
                fashion_matches=fashion_matches
            )
            crop_matches.append(crop_match)
        
        return VideoQueryResponse(
            video_id=video_id,
            total_crops=len(crop_matches),
            crop_matches=crop_matches
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to query video crops: {str(e)}"
        )


class VideoRecreateRequest(BaseModel):
    video_id: str
    vector_size: int = 512


@router.post("/recreate")
async def recreate_collection(
    request: VideoRecreateRequest,
    query_service: QueryService = Depends(get_query_service),
):
    """Recreate a video collection."""
    try:
        success = await asyncio.get_event_loop().run_in_executor(
            None, lambda: query_service.recreate_collection(
                video_id=request.video_id,
                vector_size=request.vector_size
            )
        )
        
        if success:
            return {
                "message": f"Successfully recreated collection {request.video_id}",
                "video_id": request.video_id
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to recreate collection"
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to recreate collection: {str(e)}"
        )
