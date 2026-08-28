from fastapi import APIRouter
from app.core.config import get_settings

settings = get_settings()
router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint"""

    return {
        "reranker": "local CrossEncoder",
        "langsmith": "enabled" if settings.LANGSMITH_API_KEY else "disabled",
        "config": {
            "query_rewriting": settings.ENABLE_QUERY_REWRITING,
            "top_k_retrieval": settings.TOP_K_RETRIEVAL,
            "top_k_rerank": settings.TOP_K_RERANK,
        },
    }
