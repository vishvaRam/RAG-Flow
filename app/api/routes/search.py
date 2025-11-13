from fastapi import APIRouter
from langfuse import observe, get_client

from app.models.schemas import SearchRequest, SearchResponse, SearchResult
from app.services.rag_service import rag_service
from app.core.logging import logger

router = APIRouter()

langfuse = get_client()

@router.post("/search", response_model=SearchResponse)
@observe()
async def search_endpoint(
    request: SearchRequest,
):
    logger.info(f"🔍 Search: '{request.query}' (top_k={request.top_k})")

    reranked, results = await rag_service.process_query(
            rewritten_query=request.query,
            subject_filter=request.subject_filter,
            topic_filter=request.topic_filter,
            top_k=request.top_k
        )
        
    search_results = [
        SearchResult(
                text=doc["text"][:500] + "..." if len(doc["text"]) > 500 else doc["text"],
                score=doc.get("rerank_score", doc["score"]),
                subject=doc["subject"],
                topic=doc["topic"],
                file_path=doc["file_path"],
                chunk_id=doc["chunk_id"]
            )
        for doc in reranked
    ]

    return SearchResponse(
        query=request.query,
        results=search_results,
        total_found=len(results)
    )
