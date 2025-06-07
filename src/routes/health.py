from fastapi import APIRouter
import asyncio
from datetime import datetime

from ..services.video_processing import VideoProcessingService
from ..services.gemini import GeminiService
from ..services.sam2_gdino import Sam2GroundingDinoService
from ..services.deduplication import FaissDeduplicationService
from ..services.cache import FileProcessingCache
from ..services.query import QueryService

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/")
async def health_check():
    """
    Comprehensive health check endpoint that returns the status of all services.
    """
    health_status = {
        "status": "healthy",
        "service": "tagging-engine",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {},
    }

    overall_healthy = True

    # Check Gemini Service
    try:
        gemini_service = GeminiService()
        gemini_status = await _check_gemini_service(gemini_service)
        health_status["services"]["gemini"] = gemini_status
        if not gemini_status["healthy"]:
            overall_healthy = False
    except Exception as e:
        health_status["services"]["gemini"] = {
            "healthy": False,
            "error": str(e),
            "details": "Failed to initialize Gemini service",
        }
        overall_healthy = False

    # Check SAM2 and Grounding DINO Service
    try:
        sam2_gdino_service = Sam2GroundingDinoService()
        sam2_gdino_status = await _check_sam2_gdino_service(sam2_gdino_service)
        health_status["services"]["sam2_gdino"] = sam2_gdino_status
        if not sam2_gdino_status["healthy"]:
            overall_healthy = False
    except Exception as e:
        health_status["services"]["sam2_gdino"] = {
            "healthy": False,
            "error": str(e),
            "details": "Failed to initialize SAM2/GDINO service",
        }
        overall_healthy = False

    # Check Video Processing Service
    try:
        video_service = VideoProcessingService()
        video_status = await _check_video_service(video_service)
        health_status["services"]["video_processing"] = video_status
        if not video_status["healthy"]:
            overall_healthy = False
    except Exception as e:
        health_status["services"]["video_processing"] = {
            "healthy": False,
            "error": str(e),
            "details": "Failed to initialize video processing service",
        }
        overall_healthy = False

    # Check Cache Service
    try:
        cache_service = FileProcessingCache()
        cache_status = await _check_cache_service(cache_service)
        health_status["services"]["cache"] = cache_status
        if not cache_status["healthy"]:
            overall_healthy = False
    except Exception as e:
        health_status["services"]["cache"] = {
            "healthy": False,
            "error": str(e),
            "details": "Failed to initialize cache service",
        }
        overall_healthy = False

    # Check Deduplication Service (includes Qdrant)
    try:
        dedup_service = FaissDeduplicationService()
        dedup_status = await _check_deduplication_service(dedup_service)
        health_status["services"]["deduplication"] = dedup_status
        if not dedup_status["healthy"]:
            overall_healthy = False
    except Exception as e:
        health_status["services"]["deduplication"] = {
            "healthy": False,
            "error": str(e),
            "details": "Failed to initialize deduplication service",
        }
        overall_healthy = False

    # Check Query Service (Qdrant connectivity)
    try:
        query_service = QueryService()
        query_status = await _check_query_service(query_service)
        health_status["services"]["query"] = query_status
        if not query_status["healthy"]:
            overall_healthy = False
    except Exception as e:
        health_status["services"]["query"] = {
            "healthy": False,
            "error": str(e),
            "details": "Failed to initialize query service",
        }
        overall_healthy = False

    # Set overall status
    health_status["status"] = "healthy" if overall_healthy else "unhealthy"

    return health_status


async def _check_gemini_service(service: GeminiService) -> dict:
    """Check Gemini service health"""
    try:
        # Test if the model is accessible
        if service.model is None:
            return {"healthy": False, "details": "Gemini model not initialized"}

        return {
            "healthy": True,
            "details": "Gemini service operational",
            "model": service.model.model_name
            if hasattr(service.model, "model_name")
            else "gemini-1.5-flash",
        }
    except Exception as e:
        return {
            "healthy": False,
            "error": str(e),
            "details": "Gemini service check failed",
        }


async def _check_sam2_gdino_service(service: Sam2GroundingDinoService) -> dict:
    """Check SAM2 and Grounding DINO service health"""
    try:
        status = {"healthy": True, "details": {}, "models_loaded": {}}

        # Check if models are loaded (lazy loading means they might not be loaded yet)
        status["models_loaded"]["grounding_dino"] = service._model is not None
        status["models_loaded"]["sam2"] = service._sam2_predictor is not None

        # Try to load models to verify they work
        try:
            processor, model, sam2_predictor = service._load_models()
            status["models_loaded"]["grounding_dino"] = model is not None
            status["models_loaded"]["sam2"] = sam2_predictor is not None
            status["details"]["grounding_dino"] = "Model loaded successfully"
            if sam2_predictor:
                status["details"]["sam2"] = "Model loaded successfully"
            else:
                status["details"]["sam2"] = "Masking disabled - model not loaded"
        except Exception as e:
            status["healthy"] = False
            status["details"]["error"] = f"Model loading failed: {str(e)}"

        return status
    except Exception as e:
        return {
            "healthy": False,
            "error": str(e),
            "details": "SAM2/GDINO service check failed",
        }


async def _check_video_service(service: VideoProcessingService) -> dict:
    """Check video processing service health"""
    try:
        # Check if sub-services are initialized
        services_status = {
            "grounding_service": service.grounding_service is not None,
            "deduplication_service": service.deduplication_service is not None,
            "cache_service": service.cache_service is not None,
        }

        healthy = all(services_status.values())

        return {
            "healthy": healthy,
            "details": "Video processing service operational"
            if healthy
            else "Some sub-services not initialized",
            "sub_services": services_status,
            "video_counter": service._video_counter,
        }
    except Exception as e:
        return {
            "healthy": False,
            "error": str(e),
            "details": "Video service check failed",
        }


async def _check_cache_service(service: FileProcessingCache) -> dict:
    """Check cache service health"""
    try:
        # Check if cache directory exists and is writable
        cache_dir_exists = service.cache_dir.exists()
        cache_dir_writable = service.cache_dir.is_dir() if cache_dir_exists else False

        # Count cache files
        cache_files_count = 0
        if cache_dir_exists:
            cache_files_count = len(list(service.cache_dir.glob("*.json")))

        healthy = cache_dir_exists and cache_dir_writable

        return {
            "healthy": healthy,
            "details": "Cache service operational"
            if healthy
            else "Cache directory issues",
            "cache_directory": str(service.cache_dir),
            "cache_files_count": cache_files_count,
            "directory_exists": cache_dir_exists,
            "directory_writable": cache_dir_writable,
        }
    except Exception as e:
        return {
            "healthy": False,
            "error": str(e),
            "details": "Cache service check failed",
        }


async def _check_deduplication_service(service: FaissDeduplicationService) -> dict:
    """Check deduplication service health (includes Qdrant connectivity)"""
    try:
        # Test Qdrant connectivity
        qdrant_healthy = False
        qdrant_details = {}

        try:
            collections = await asyncio.get_event_loop().run_in_executor(
                None, lambda: service.client.get_collections()
            )
            qdrant_healthy = True
            qdrant_details = {
                "collections_count": len(collections.collections),
                "connection": "successful",
            }
        except Exception as e:
            qdrant_details = {"connection": "failed", "error": str(e)}

        # Check model loading capability
        model_status = {
            "processor_loaded": service._processor is not None,
            "model_loaded": service._model is not None,
        }

        healthy = qdrant_healthy

        return {
            "healthy": healthy,
            "details": "Deduplication service operational"
            if healthy
            else "Service issues detected",
            "qdrant": qdrant_details,
            "models": model_status,
            "similarity_threshold": service.similarity_threshold,
        }
    except Exception as e:
        return {
            "healthy": False,
            "error": str(e),
            "details": "Deduplication service check failed",
        }


async def _check_query_service(service: QueryService) -> dict:
    """Check query service health (Qdrant connectivity)"""
    try:
        # Test Qdrant connectivity
        collections = await asyncio.get_event_loop().run_in_executor(
            None, lambda: service.client.get_collections()
        )

        # Look for fashion products collection
        fashion_collection_exists = any(
            col.name == service.fashion_products_collection
            for col in collections.collections
        )

        return {
            "healthy": True,
            "details": "Query service operational",
            "qdrant_url": service.qdrant_url,
            "collections_count": len(collections.collections),
            "fashion_products_collection": {
                "name": service.fashion_products_collection,
                "exists": fashion_collection_exists,
            },
        }
    except Exception as e:
        return {
            "healthy": False,
            "error": str(e),
            "details": "Query service check failed",
        }
