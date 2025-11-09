from typing import List, Dict, Optional
from app.services.embedding_service import EmbeddingService, embedding_service
from app.services.search_service import SearchService, search_service
from app.services.reranker_service import RerankerService, reranker_service
from app.services.llm_service import LLMService, llm_service
from app.services.db_service import PostgreSQLService, db_service
from app.core.config import get_settings

settings = get_settings()


class RAGService:
    """Main RAG orchestration service"""
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        search_service: SearchService,
        reranker_service: RerankerService,
        llm_service: LLMService,
        db_service: PostgreSQLService
    ):
        self.embedding = embedding_service
        self.search = search_service
        self.reranker = reranker_service
        self.llm = llm_service
        self.db = db_service
    
    @staticmethod
    def build_context(documents: List[Dict]) -> str:
        """Build context from documents"""
        if not documents:
            return "No relevant context found."
        
        return "\n\n".join([
            f"[Document {i}]\n"
            f"Subject: {doc['subject']} | Topic: {doc['topic']}\n"
            f"Source: {doc['chunk_id']}\n"
            f"Content: {doc['text']}"
            for i, doc in enumerate(documents, 1)
        ])
    
    @staticmethod
    def create_rag_prompt(query: str, context: str) -> str:
        """Create RAG prompt"""
        return f"""You are a JEE tutor. Answer using ONLY the context provided.

            Context:
            {context}

            Question: {query}

            Format your answer in markdown:
            - Use ## headers for sections
            - Use **bold** for key terms
            - Use $ for inline math (e.g., $F=ma$) and $$ for display equations
            - Show step-by-step: **Given** → **Formula** → **Solution** → **Answer**
            - Use SI units and JEE notation

            If context is insufficient, say "I don't have enough information to answer this."

            Answer:"""
            
            
    async def process_query(
        self,
        query: str,
        subject_filter: Optional[str] = None,
        topic_filter: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> tuple:
        """Process query through RAG pipeline"""
        
        # Step 1: Query rewriting
        rewritten_query = await self.llm.rewrite_query(query, subject_filter)
        
        # Step 2: Generate embedding
        query_vector = await self.embedding.generate_embedding(rewritten_query)
        
        # Step 3: Hybrid search
        results = await self.search.hybrid_search(
            query=rewritten_query,
            query_vector=query_vector,
            top_k=settings.TOP_K_RETRIEVAL,
            subject_filter=subject_filter,
            topic_filter=topic_filter,
            alpha=settings.HYBRID_ALPHA
        )
        
        # Step 4: Rerank
        top_k = top_k or settings.TOP_K_RERANK
        reranked = await self.reranker.rerank(query, results, top_k) if results else []
        
        return reranked, results
    
rag_service = RAGService(
    embedding_service=embedding_service,
    search_service=search_service,
    reranker_service=reranker_service,
    llm_service=llm_service,
    db_service=db_service
)