from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/")
async def health_check():
    """
    Health check endpoint that returns the service status.
    """
    return {"status": "healthy", "service": "tagging-engine"}
