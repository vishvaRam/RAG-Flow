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
            f'<document id="{i}">\n'
            f" {doc['text']}"
            f'\n</document>\n\n'
            for i, doc in enumerate(documents, 1)
        ])
    
    @staticmethod
    def create_rag_prompt(query: str, context: str) -> str:
        """Create RAG prompt"""
        return f"""You are a JEE tutor. Answer using ONLY the context provided.

            <retrieved_context>
            {context}
            </retrieved_context>

            <question>
            {query}
            </question>

           <instructions>
            - Answer ONLY using information from the retrieved context above
            - If the context lacks relevant information, state: "I don't have enough information to answer this"
            - Do NOT mention documents, sources, or where information came from
            - Explain concepts as if from your own knowledge
            </instructions>\n\n"""            
            
    async def process_query(
        self,
        rewritten_query: str,
        subject_filter: Optional[str] = None,
        topic_filter: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> tuple:
        """Process query through RAG pipeline"""  
        
        # Step 1: Generate embedding
        query_vector = await self.embedding.generate_embedding(rewritten_query)
        
        # Step 2: Hybrid search
        results = await self.search.hybrid_search(
            query=rewritten_query,
            query_vector=query_vector,
            top_k=settings.TOP_K_RETRIEVAL,
            subject_filter=subject_filter,
            topic_filter=topic_filter,
            alpha=settings.HYBRID_ALPHA
        )
        
        # Step 2: Rerank
        top_k = top_k or settings.TOP_K_RERANK
        reranked = await self.reranker.rerank(rewritten_query, results, top_k) if results else []
        
        return reranked, results
    
rag_service = RAGService(
    embedding_service=embedding_service,
    search_service=search_service,
    reranker_service=reranker_service,
    llm_service=llm_service,
    db_service=db_service
)