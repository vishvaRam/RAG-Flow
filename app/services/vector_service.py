from typing import Any, List, Optional
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import get_settings
from app.core.logging import logger

settings = get_settings()


class VectorService:
    """Manages PGVector hybrid index, embedding calculations, and metadata filtering"""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.LLM_API_KEY,
            openai_api_base=settings.EMBEDDING_PROVIDER_URL,
            check_embedding_ctx_length=False,
            dimensions=settings.EMBEDDING_DIMENSIONS,
            chunk_size=90,
        )

        # Async PostgreSQL connection
        self.connection_string = (
            f"postgresql+asyncpg://{settings.DB_USER}:{settings.DB_PASSWORD}@"
            f"{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        )

        # Create async engine
        self.async_engine = create_async_engine(
            self.connection_string,
            pool_pre_ping=True,
        )

        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name=settings.COLLECTION_NAME,
            connection=self.async_engine,
            use_jsonb=True,
            embedding_length=settings.EMBEDDING_DIMENSIONS,
            create_extension=False,
        )

        logger.info("✓ PGVector store configured successfully")

    async def add_documents(
        self, documents: List[Document], batch_size: int = 90
    ) -> List[str]:
        """Persist chunked documents into PGVector safely in batches (<100)."""
        all_ids: List[str] = []
        try:
            total_docs = len(documents)
            for i in range(0, total_docs, batch_size):
                batch = documents[i : i + batch_size]
                ids = await self.vector_store.aadd_documents(batch)
                all_ids.extend(ids)
                logger.info(
                    f"Indexed batch {i // batch_size + 1}/{(total_docs + batch_size - 1) // batch_size} "
                    f"({len(batch)} chunks)"
                )

            logger.info(f"✓ Successfully indexed all {len(all_ids)} chunks into PGVector")
            return all_ids

        except Exception as e:
            logger.error(
                f"Error adding documents to PGVector: {e}",
                exc_info=True,
            )
            raise

    async def search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[dict[str, Any]] = None,
    ) -> List[Document]:
        """Vector search with JSONB metadata filtering"""
        try:
            results = await self.vector_store.asimilarity_search(
                query=query,
                k=k,
                filter=filter_dict,
            )
            return results
        except Exception as e:
            logger.error(
                f"Vector search failed: {e}",
                exc_info=True,
            )
            return []


vector_service = VectorService()