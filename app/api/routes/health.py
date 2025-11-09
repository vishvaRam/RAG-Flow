from fastapi import APIRouter
from app.services.search_service import SearchService, search_service
from app.core.config import get_settings

settings = get_settings()
router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    es_health = await search_service.health_check()

    return {
        "status": "healthy" if es_health["status"] == "healthy" else "degraded",
        "elasticsearch": es_health,
        "reranker": "local CrossEncoder",
        "langfuse": "enabled" if settings.LANGFUSE_PUBLIC_KEY else "disabled",
        "config": {
            "query_rewriting": settings.ENABLE_QUERY_REWRITING,
            "top_k_retrieval": settings.TOP_K_RETRIEVAL,
            "top_k_rerank": settings.TOP_K_RERANK
        }
    }
