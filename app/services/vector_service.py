import asyncio
from typing import Any

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import get_settings
from app.core.logging import logger

settings = get_settings()


class VectorService:
    """Manages PGVector hybrid index, embedding calculations, and metadata filtering."""

    def __init__(self):
        # Configure smaller chunk_size and automatic retries for rate-limited endpoints
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.LLM_API_KEY,
            openai_api_base=settings.EMBEDDING_PROVIDER_URL,
            check_embedding_ctx_length=False,
            dimensions=settings.EMBEDDING_DIMENSIONS,
            chunk_size=40,  # Reduced from 90 to prevent payload/RPM exhaustion
            max_retries=5,
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
        self,
        documents: list[Document],
        batch_size: int = 40,
        rate_limit_delay: float = 1.0,
    ) -> list[str]:
        """Persist chunked documents into PGVector safely in micro-batches with rate-limit pacing."""
        all_ids: list[str] = []
        total_docs = len(documents)
        total_batches = (total_docs + batch_size - 1) // batch_size

        try:
            for idx, i in enumerate(range(0, total_docs, batch_size)):
                batch = documents[i : i + batch_size]

                # Retry loop per micro-batch with exponential backoff for 429 errors
                max_attempts = 4
                for attempt in range(max_attempts):
                    try:
                        ids = await self.vector_store.aadd_documents(batch)
                        all_ids.extend(ids)
                        logger.info(
                            f"Indexed batch {idx + 1}/{total_batches} ({len(batch)} chunks)"
                        )
                        break
                    except Exception as err:
                        if "429" in str(err) and attempt < max_attempts - 1:
                            wait_time = (2**attempt) * 2
                            logger.warning(
                                f"Rate limited on batch {idx + 1}. Sleeping {wait_time}s before retry..."
                            )
                            await asyncio.sleep(wait_time)
                        else:
                            raise

                # Non-blocking pause between batches to respect RPM limits
                if idx < total_batches - 1:
                    await asyncio.sleep(rate_limit_delay)

            logger.info(
                f"✓ Successfully indexed all {len(all_ids)} chunks into PGVector"
            )
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
        filter_dict: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Vector search with JSONB metadata filtering."""
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
