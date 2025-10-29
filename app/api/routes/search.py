from fastapi import APIRouter, Depends
from langfuse import observe, get_client

from app.models.schemas import SearchRequest, SearchResponse, SearchResult
from app.api.dependencies import get_rag_service
from app.services.rag_service import RAGService
from app.core.logging import logger

router = APIRouter()

langfuse = get_client()

@router.post("/search", response_model=SearchResponse)
async def search_endpoint(
    request: SearchRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    with langfuse.start_as_current_span(
        name="search-endpoint",
        input={"query": request.query, "top_k": request.top_k}
    ) as span:
        logger.info(f"🔍 Search: '{request.query}' (top_k={request.top_k})")
        
        reranked, results = await rag_service.process_query(
            query=request.query,
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
        
        span.update(output={"total_results": len(search_results)})
        logger.info(f"✅ Returned {len(search_results)} results")
        
    langfuse.flush()
    
    return SearchResponse(
        query=request.query,
        results=search_results,
        total_found=len(results)
    )
