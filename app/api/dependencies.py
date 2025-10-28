from typing import Optional
from app.services.embedding_service import EmbeddingService
from app.services.search_service import SearchService
from app.services.reranker_service import RerankerService
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.core.logging import logger


# ✅ Global singleton instances (initialized at startup, not per request)
_embedding_service: Optional[EmbeddingService] = None
_search_service: Optional[SearchService] = None
_reranker_service: Optional[RerankerService] = None
_llm_service: Optional[LLMService] = None
_rag_service: Optional[RAGService] = None


def init_services():
    """Initialize all services at startup (call this in lifespan)"""
    global _embedding_service, _search_service, _reranker_service, _llm_service, _rag_service
    
    logger.info("🔧 Initializing services...")
    
    _embedding_service = EmbeddingService()
    _search_service = SearchService()
    _reranker_service = RerankerService()
    _llm_service = LLMService()
    _rag_service = RAGService(
        embedding_service=_embedding_service,
        search_service=_search_service,
        reranker_service=_reranker_service,
        llm_service=_llm_service
    )
    
    logger.info("✅ All services initialized")


def get_embedding_service() -> EmbeddingService:
    """Get embedding service singleton"""
    if _embedding_service is None:
        raise RuntimeError("Services not initialized. Call init_services() first.")
    return _embedding_service


def get_search_service() -> SearchService:
    """Get search service singleton"""
    if _search_service is None:
        raise RuntimeError("Services not initialized. Call init_services() first.")
    return _search_service


def get_reranker_service() -> RerankerService:
    """Get reranker service singleton"""
    if _reranker_service is None:
        raise RuntimeError("Services not initialized. Call init_services() first.")
    return _reranker_service


def get_llm_service() -> LLMService:
    """Get LLM service singleton"""
    if _llm_service is None:
        raise RuntimeError("Services not initialized. Call init_services() first.")
    return _llm_service


def get_rag_service() -> RAGService:
    """Get RAG service singleton"""
    if _rag_service is None:
        raise RuntimeError("Services not initialized. Call init_services() first.")
    return _rag_service


def cleanup_services():
    """Cleanup services on shutdown"""
    global _search_service
    if _search_service:
        _search_service.close()
    logger.info("✅ Services cleaned up")
