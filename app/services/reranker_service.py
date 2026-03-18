import asyncio
from typing import List, Dict, Any, Tuple

import numpy as np
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
from langchain_community.cross_encoders.base import BaseCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.documents import Document
from langsmith import traceable

from app.core.config import get_settings
from app.core.logging import logger
from app.utils.decorators import log_time

settings = get_settings()


class ONNXCrossEncoder(BaseCrossEncoder):
    def __init__(self, model_name: str, max_length: int = 512):
        logger.info(f"Loading ONNX cross-encoder: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = ORTModelForSequenceClassification.from_pretrained(
            model_name,
            file_name="model_O2.onnx",
            provider="CPUExecutionProvider",
        )
        self.max_length = max_length

    def score(self, text_pairs: List[Tuple[str, str]]) -> List[float]:
        features = self.tokenizer(
            text_pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",  # numpy instead of torch tensors
        )
        logits = self.model(**features).logits  # returns numpy array directly
        scores = logits.squeeze(-1).tolist()
        return [scores] if isinstance(scores, float) else scores


class RerankerService:
    """Service for reranking documents"""

    def __init__(self):
        logger.info(f"Loading reranker: {settings.RERANKER_MODEL}")
        cross_encoder = ONNXCrossEncoder(model_name=settings.RERANKER_MODEL)
        self.reranker = CrossEncoderReranker(
            model=cross_encoder, top_n=settings.TOP_K_RERANK
        )

    @traceable
    @log_time("Reranking")
    async def rerank(
        self, query: str, documents: List[Dict[str, Any]], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Rerank documents using CrossEncoder"""
        if not documents:
            return []

        try:
            langchain_docs = [
                Document(
                    page_content=doc["text"],
                    metadata={
                        "subject": doc["subject"],
                        "topic": doc["topic"],
                        "file_path": doc["file_path"],
                        "chunk_id": doc["chunk_id"],
                        "original_score": doc["score"],
                    },
                )
                for doc in documents
            ]

            reranked_docs = await asyncio.to_thread(
                self.reranker.compress_documents, documents=langchain_docs, query=query
            )

            results = [
                {
                    "text": doc.page_content,
                    "score": doc.metadata.get("original_score", 0.0),
                    "rerank_score": doc.metadata.get("relevance_score", 0.0),
                    "subject": doc.metadata.get("subject", ""),
                    "topic": doc.metadata.get("topic", ""),
                    "file_path": doc.metadata.get("file_path", ""),
                    "chunk_id": doc.metadata.get("chunk_id", ""),
                }
                for doc in reranked_docs[:top_k]
            ]

            logger.info(f"📊 Reranked to top {len(results)} documents")
            return results
        except Exception as e:
            logger.error(f"Reranking failed: {e}", exc_info=True)
            return documents[:top_k]


reranker_service = RerankerService()
