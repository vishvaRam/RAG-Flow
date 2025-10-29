import asyncio
from typing import List, Dict, Optional, Any
from elasticsearch import Elasticsearch
from langfuse import observe

from app.core.config import get_settings
from app.core.logging import logger
from app.utils.decorators import log_time

settings = get_settings()


class SearchService:
    """Service for Elasticsearch operations"""
    
    def __init__(self):
        # ✅ Fixed for Elasticsearch 9.x - removed maxsize parameter
        self.client = Elasticsearch(
            hosts=[f"http://{settings.ELASTICSEARCH_HOST}:{settings.ELASTICSEARCH_PORT}"],
            max_retries=settings.ELASTICSEARCH_MAX_RETRIES,
            retry_on_timeout=True,
            request_timeout=settings.ELASTICSEARCH_TIMEOUT
        )
        logger.info(f"✓ Elasticsearch connected: {settings.ELASTICSEARCH_HOST}:{settings.ELASTICSEARCH_PORT}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Elasticsearch health"""
        try:
            info = self.client.info()
            count = self.client.count(index=settings.ELASTICSEARCH_INDEX)
            return {
                "status": "healthy",
                "version": info['version']['number'],
                "document_count": count['count']
            }
        except Exception as e:
            logger.error(f"Elasticsearch health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    # @observe()
    @log_time("Hybrid search")
    async def hybrid_search(
        self,
        query: str,
        query_vector: List[float],
        top_k: int = 20,
        subject_filter: Optional[str] = None,
        topic_filter: Optional[str] = None,
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Perform hybrid BM25 + vector search"""
        
        filters = []
        if subject_filter:
            filters.append({"term": {"subject": subject_filter}})
        if topic_filter:
            filters.append({"term": {"topic": topic_filter}})
        
        search_body = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        {"match": {"text": {"query": query, "boost": 1 - alpha}}},
                        {
                            "script_score": {
                                "query": {"match_all": {}} if not filters else {"bool": {"filter": filters}},
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, 'dense_vector') + 1.0",
                                    "params": {"query_vector": query_vector}
                                },
                                "boost": alpha
                            }
                        }
                    ],
                    "filter": filters,
                    "minimum_should_match": 1
                }
            },
            "_source": ["text", "subject", "topic", "file_path", "chunk_id"]
        }
        
        try:
            response = await asyncio.to_thread(
                self.client.search,
                index=settings.ELASTICSEARCH_INDEX,
                body=search_body
            )
            
            results = [
                {
                    "text": hit['_source']['text'],
                    "score": hit['_score'],
                    "subject": hit['_source']['subject'],
                    "topic": hit['_source']['topic'],
                    "file_path": hit['_source']['file_path'],
                    "chunk_id": hit['_source']['chunk_id']
                }
                for hit in response['hits']['hits']
            ]
            
            logger.info(f"📊 Retrieved {len(results)} documents")
            return results
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}", exc_info=True)
            raise
    
    def close(self):
        """Close Elasticsearch connection"""
        self.client.close()
        logger.info("✓ Elasticsearch connection closed")
