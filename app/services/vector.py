import asyncio
from typing import Any

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector

from app.core.config import get_settings
from app.core.database import db_manager
from app.core.logging import logger

settings = get_settings()


class VectorService:
    """Encapsulates PGVector store operations and embedding generation."""

    def __init__(self):
        self._store: PGVector | None = None

    @property
    def store(self) -> PGVector:
        if self._store is None:
            if not db_manager.engine:
                raise RuntimeError("Database engine not initialized.")

            embeddings = OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                openai_api_key=settings.LLM_API_KEY,
                openai_api_base=settings.EMBEDDING_PROVIDER_URL,
                check_embedding_ctx_length=False,
                dimensions=settings.EMBEDDING_DIMENSIONS,
                chunk_size=40,
                max_retries=5,
            )
            self._store = PGVector(
                embeddings=embeddings,
                collection_name=settings.COLLECTION_NAME,
                connection=db_manager.engine,
                use_jsonb=True,
                embedding_length=settings.EMBEDDING_DIMENSIONS,
                create_extension=False,
            )
        return self._store

    async def add_documents(
        self, documents: list[Document], batch_size: int = 40, delay: float = 0.5
    ) -> list[str]:
        all_ids: list[str] = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            for attempt in range(4):
                try:
                    ids = await self.store.aadd_documents(batch)
                    all_ids.extend(ids)
                    break
                except Exception as err:
                    if "429" in str(err) and attempt < 3:
                        wait = (2**attempt) * 1.5
                        logger.warning(
                            f"Embedding rate limit reached. Retrying in {wait}s..."
                        )
                        await asyncio.sleep(wait)
                    else:
                        raise
            if i + batch_size < len(documents):
                await asyncio.sleep(delay)
        return all_ids

    async def search(
        self, query: str, k: int | None = None, filters: dict[str, Any] | None = None
    ) -> list[Document]:
        limit = k or settings.TOP_K_RETRIEVAL
        try:
            return await self.store.asimilarity_search(
                query=query, k=limit, filter=filters
            )
        except Exception as err:
            logger.error(f"Vector search failed: {err}", exc_info=True)
            return []


vector_service = VectorService()
