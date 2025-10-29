import asyncio
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langfuse import observe

from app.core.config import get_settings
from app.core.logging import logger
from app.utils.decorators import log_time

settings = get_settings()


class EmbeddingService:
    """Service for generating embeddings"""
    
    def __init__(self):
        self.model = GoogleGenerativeAIEmbeddings(
            model=settings.GEMINI_EMBEDDING_MODEL,
            google_api_key=settings.GEMINI_API_KEY, # type: ignore
        )
        logger.info(f"✓ Embedding model loaded: {settings.GEMINI_EMBEDDING_MODEL}")
    
    # @observe()
    @log_time("Embedding generation")
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            embedding = await asyncio.to_thread(
                self.model.embed_query,
                text,
                output_dimensionality=settings.EMBEDDING_DIMS,
                task_type="retrieval_query"
            )
            logger.info(f"📊 Generated {len(embedding)}-dimensional embedding")
            return embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}", exc_info=True)
            raise
