import asyncio
from typing import Any

import httpx
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector

from app.core.config import get_settings
from app.core.database import db_manager
from app.core.logging import logger

settings = get_settings()


class VectorService:
    """Encapsulates PGVector store operations, embedding generation, and reranking."""

    def __init__(self):
        self._store: PGVector | None = None
        self._http_client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

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

    async def _rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int,
    ) -> list[Document]:
        """Reranks documents via OpenRouter's /api/v1/rerank endpoint."""
        if not documents:
            return []

        payload = {
            "model": settings.RERANKER_MODEL,
            "query": query,
            "documents": [doc.page_content for doc in documents],
            "top_n": min(top_n, len(documents)),
        }
        headers = {
            "Authorization": f"Bearer {settings.LLM_API_KEY}",
            "Content-Type": "application/json",
        }

        try:
            response = await self.client.post(
                settings.RERANKER_PROVIDER_URL + "/rerank",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

            reranked_docs: list[Document] = []
            for item in data.get("results", []):
                idx = item["index"]
                doc = documents[idx]
                # Attach relevance score directly to metadata
                doc.metadata["relevance_score"] = item.get("relevance_score")
                reranked_docs.append(doc)

            return reranked_docs

        except Exception as err:
            logger.error(
                f"Reranking failed, falling back to top-{top_n} vector results: {err}"
            )
            return documents[:top_n]

    async def search(
        self,
        query: str,
        k: int | None = None,
        filters: dict[str, Any] | None = None,
        rerank: bool = True,
        reranked_k: int = 8,
    ) -> list[Document]:
        """
        Performs similarity search, optionally reranking top results with Qwen.
        - initial_k: Number of candidate documents fetched from PGVector.
        - limit (k): Final number of highest-scoring documents returned.
        """

        try:
            candidates = await self.store.asimilarity_search(
                query=query, k=k, filter=filters
            )
        except Exception as err:
            logger.error(f"Vector search failed: {err}", exc_info=True)
            return []
        if rerank:
            return await self._rerank(
                query=query, documents=candidates, top_n=reranked_k
            )
        else:
            return candidates[:k]

    async def delete_by_filename(self, filename: str) -> int:
        """Deletes all chunks associated with a given filename from PGVector."""
        query = """
            DELETE FROM langchain_pg_embedding
            WHERE collection_id = (
                SELECT uuid FROM langchain_pg_collection WHERE name = $1 LIMIT 1
            )
            AND (
                cmetadata->>'source' = $2
                OR cmetadata->>'filename' = $2
            )
            RETURNING id;
        """
        async with db_manager.acquire_pg() as conn:
            deleted_rows = await conn.fetch(query, settings.COLLECTION_NAME, filename)
            count = len(deleted_rows)
            logger.info(f"Deleted {count} vector chunks for file '{filename}'.")
            return count


vector_service = VectorService()
