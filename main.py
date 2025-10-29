import os
import logging
import json
import time
import asyncio
from typing import List, Dict, Optional, Any
from functools import wraps
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, PlainTextResponse
from elasticsearch import Elasticsearch

# Langfuse imports
from langfuse import get_client, observe
from langfuse.openai import AsyncOpenAI

# LangChain imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document

# Utils
from Utils.Config import config
from Utils.models import ChatRequest, SearchRequest, SearchResponse, SearchResult

# ============================================================================
# SETUP & INITIALIZATION
# ============================================================================

os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/rag_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize clients
langfuse = get_client()
es_client = Elasticsearch(
    [f"http://{config.ELASTICSEARCH_HOST}:{config.ELASTICSEARCH_PORT}"],
    max_retries=3,
    retry_on_timeout=True,
    # maxsize=25,
    request_timeout=30
)

openai_client = AsyncOpenAI(
    api_key=config.GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    timeout=20.0,
    max_retries=1
)

# Initialize embeddings model
embeddings_model = GoogleGenerativeAIEmbeddings(
    model=config.GEMINI_EMBEDDING_MODEL,
    google_api_key=config.GOOGLE_API_KEY, # type: ignore
)

# Initialize reranker
logger.info(f"Loading reranker: {config.RERANKER_MODEL}")
cross_encoder = HuggingFaceCrossEncoder(model_name=config.RERANKER_MODEL)
reranker = CrossEncoderReranker(model=cross_encoder, top_n=config.TOP_K_RERANK)
logger.info("✓ Reranker loaded")

# Query rewriting configuration
ENABLE_QUERY_REWRITING = config.ENABLE_QUERY_REWRITING
QUERY_REWRITE_TIMEOUT = config.QUERY_REWRITE_TIMEOUT

# ============================================================================
# UTILITIES
# ============================================================================

def log_time(operation_name: str):
    """Decorator to log execution time"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                logger.info(f"⏱️  {operation_name} took {(time.time() - start) * 1000:.2f}ms")
                return result
            except Exception as e:
                logger.error(f"⏱️  {operation_name} failed after {(time.time() - start) * 1000:.2f}ms: {e}")
                raise
        return wrapper
    return decorator

# ============================================================================
# CORE RAG FUNCTIONS - PLAIN ASYNC/AWAIT (NO SEMAPHORES)
# ============================================================================

@observe()
@log_time("Query rewriting")
async def rewrite_query(original_query: str, subject_filter: Optional[str] = None, enable: bool = True) -> str:
    """Rewrite user query for better retrieval (optional)"""
    
    if not enable or len(original_query.split()) < 3:
        logger.info("📝 Skipping query rewriting")
        return original_query
    
    rewrite_prompt = f"""Add technical terms to: "{original_query}"
        Subject: {subject_filter or "Physics/Chemistry/Mathematics"}
        Output ONLY the improved Hybrid RAG query."""

    try:
        async with asyncio.timeout(QUERY_REWRITE_TIMEOUT):
            response = await openai_client.chat.completions.create(
                reasoning_effort="none", # type: ignore
                model=config.QUERY_REWRITE_MODEL,
                messages=[{"role": "user", "content": rewrite_prompt}],
                max_tokens=config.QUERY_REWRITE_MAX_TOKENS,
                temperature=0.1
            )
        
        if not response.choices:
            logger.warning("No choices in response")
            return original_query
            
        content = response.choices[0].message.content
        
        if content and content.strip():
            rewritten = content.strip().strip('"').strip("'")
            logger.info(f"📝 Rewritten: '{original_query[:40]}' → '{rewritten[:40]}'")
            return rewritten
        else:
            return original_query
        
    except asyncio.TimeoutError:
        logger.warning(f"Query rewriting timeout after {QUERY_REWRITE_TIMEOUT}s")
        return original_query
    except Exception as e:
        logger.error(f"Query rewriting failed: {e}")
        return original_query

@observe()
@log_time("Embedding generation")
async def get_embedding(text: str) -> List[float]:
    """Generate query embedding - plain async"""
    try:
        # ✅ Just await - no semaphore
        embedding = await asyncio.to_thread(
            embeddings_model.embed_query,
            text,
            output_dimensionality=config.EMBEDDING_DIMS,
            task_type="retrieval_query"
        )
        logger.info(f"📊 Generated {len(embedding)}-dimensional embedding")
        return embedding
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}", exc_info=True)
        raise

@observe()
@log_time("Hybrid search")
async def hybrid_search(
    query: str,
    query_vector: List[float],
    top_k: int = 20,
    subject_filter: Optional[str] = None,
    topic_filter: Optional[str] = None,
    alpha: float = 0.5
) -> List[Dict[str, Any]]:
    """Perform hybrid search - plain async"""
    
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
                    {
                        "match": {
                            "text": {
                                "query": query,
                                "boost": 1 - alpha
                            }
                        }
                    },
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
        # ✅ Just await - no semaphore
        response = await asyncio.to_thread(es_client.search, index=config.ELASTICSEARCH_INDEX, body=search_body)
        
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

@observe()
@log_time("Reranking")
async def rerank_documents(query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """Rerank documents - plain async"""
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
                    "original_score": doc["score"]
                }
            )
            for doc in documents
        ]
        
        # ✅ Just await - no semaphore
        reranked_docs = await asyncio.to_thread(
            reranker.compress_documents,
            documents=langchain_docs,
            query=query
        )
        
        results = [
            {
                "text": doc.page_content,
                "score": doc.metadata.get("original_score", 0.0),
                "rerank_score": doc.metadata.get("relevance_score", 0.0),
                "subject": doc.metadata.get("subject", ""),
                "topic": doc.metadata.get("topic", ""),
                "file_path": doc.metadata.get("file_path", ""),
                "chunk_id": doc.metadata.get("chunk_id", "")
            }
            for doc in reranked_docs[:top_k]
        ]
        
        logger.info(f"📊 Reranked to top {len(results)} documents")
        return results
    except Exception as e:
        logger.error(f"Reranking failed: {e}", exc_info=True)
        return documents[:top_k]

def build_context(documents: List[Dict[str, Any]]) -> str:
    """Build context string from documents"""
    if not documents:
        return "No relevant context found."
    
    return "\n\n".join([
        f"[Document {i}]\n"
        f"Subject: {doc['subject']} | Topic: {doc['topic']}\n"
        f"Source: {doc['chunk_id']}\n"
        f"Content: {doc['text']}"
        for i, doc in enumerate(documents, 1)
    ])

def create_rag_prompt(query: str, context: str) -> str:
    """Create RAG prompt for JEE tutoring"""
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

SYSTEM_MESSAGE = {
    "role": "system", 
    "content": """You are a JEE examination tutor with expertise in Physics, Chemistry, and Mathematics. 
        Your role is to:
            - Explain concepts clearly with proper scientific reasoning
            - Break down complex problems into manageable steps
            - Use LaTeX notation for all mathematical expressions
            - Connect theory to JEE exam patterns and real-world applications
            - Maintain an encouraging, patient tone that builds student confidence
            - Focus on conceptual understanding over rote memorization
        """
}

# ============================================================================
# FASTAPI APP
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("Starting RAG API service...")
    logger.info(f"  - Elasticsearch: {config.ELASTICSEARCH_HOST}:{config.ELASTICSEARCH_PORT}")
    logger.info(f"  - Index: {config.ELASTICSEARCH_INDEX}")
    logger.info(f"  - Model: {config.GEMINI_MODEL}")
    logger.info(f"  - Reranker: {config.RERANKER_MODEL}")
    logger.info(f"  - Query rewriting: {'ENABLED' if ENABLE_QUERY_REWRITING else 'DISABLED'}")
    
    try:
        if langfuse.auth_check():
            logger.info("✓ Langfuse connected")
    except Exception as e:
        logger.warning(f"⚠ Langfuse unavailable: {e}")
    
    try:
        info = es_client.info()
        count = es_client.count(index=config.ELASTICSEARCH_INDEX)
        logger.info(f"✓ Elasticsearch {info['version']['number']} - {count['count']} documents")
    except Exception as e:
        logger.error(f"✗ Elasticsearch connection failed: {e}")
        raise
    
    logger.info("✓ Service started")
    yield
    
    langfuse.flush()
    es_client.close()
    logger.info("✓ Service stopped")

app = FastAPI(
    title="Hybrid RAG API",
    description="RAG system with Elasticsearch hybrid search and CrossEncoder reranking",
    version="2.0.0",
    lifespan=lifespan
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log request timing"""
    start = time.time()
    logger.info(f"🔵 {request.method} {request.url.path}")
    response = await call_next(request)
    duration = (time.time() - start) * 1000
    logger.info(f"🟢 {request.method} {request.url.path} - {response.status_code} - {duration:.2f}ms")
    response.headers["X-Process-Time"] = f"{duration:.2f}ms"
    return response

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Hybrid RAG API with Gemini",
        "version": "2.0.0",
        "model": config.GEMINI_MODEL,
        "embedding_model": config.GEMINI_EMBEDDING_MODEL,
        "reranker": config.RERANKER_MODEL,
        "concurrency": "plain async/await (no artificial limits)",
        "optimizations": {
            "query_rewriting": "disabled" if not ENABLE_QUERY_REWRITING else "enabled",
            "api_timeout": "20s",
            "max_retries": 2
        },
        "endpoints": {
            "/chat": "Chat with RAG",
            "/search": "Search documents",
            "/health": "Health check",
            "/docs": "API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check"""
    try:
        es_client.cluster.health()
        count = es_client.count(index=config.ELASTICSEARCH_INDEX)
        es_status = f"healthy ({count['count']} docs)"
    except Exception as e:
        es_status = f"unhealthy: {e}"
    
    return {
        "status": "healthy" if "healthy" in es_status else "degraded",
        "elasticsearch": es_status,
        "reranker": "local CrossEncoder",
        "langfuse": "enabled" if config.LANGFUSE_PUBLIC_KEY else "disabled"
    }

@app.post("/search", response_model=SearchResponse)
@observe()
async def search_endpoint(request: SearchRequest):
    """Search documents"""
    logger.info(f"🔍 Search: '{request.query}' (top_k={request.top_k})")

    rewritten_query = await rewrite_query(
        request.query, 
        request.subject_filter,
        enable=ENABLE_QUERY_REWRITING
    )

    query_vector = await get_embedding(rewritten_query)

    results = await hybrid_search(
        query=rewritten_query,
        query_vector=query_vector,
        top_k=config.TOP_K_RETRIEVAL,
        subject_filter=request.subject_filter,
        topic_filter=request.topic_filter,
        alpha=config.HYBRID_ALPHA
    )
    
    reranked = await rerank_documents(request.query, results, request.top_k) if results else []
    
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
    
    logger.info(f"✅ Returned {len(search_results)} results")
    return SearchResponse(query=request.query, results=search_results, total_found=len(results))

@app.post("/chat")
@observe()
async def chat_endpoint(request: ChatRequest):
    """Chat with RAG"""
    user_messages = [msg for msg in request.messages if msg.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")
    
    query = user_messages[-1].content
    logger.info(f"💬 Chat: '{query[:100]}...'")
    
    rewritten_query = await rewrite_query(
        query, 
        request.subject_filter,
        enable=ENABLE_QUERY_REWRITING
    )

    query_vector = await get_embedding(rewritten_query)
    results = await hybrid_search(
        query=rewritten_query,
        query_vector=query_vector,
        top_k=config.TOP_K_RETRIEVAL,
        subject_filter=request.subject_filter,
        topic_filter=request.topic_filter,
        alpha=config.HYBRID_ALPHA
    )
    
    top_k = request.top_k or config.TOP_K_RERANK
    reranked = await rerank_documents(query, results, top_k) if results else []
    
    context = build_context(reranked)
    rag_prompt = create_rag_prompt(query, context)
    
    conversation_messages = [SYSTEM_MESSAGE]
    for msg in request.messages[:-1]:
        conversation_messages.append({"role": msg.role, "content": msg.content})
    conversation_messages.append({"role": "user", "content": rag_prompt})
    
    temperature = request.temperature or config.TEMPERATURE
    
    if request.stream:
        async def generate_stream():
            try:
                start = time.time()
                stream = await openai_client.chat.completions.create(
                    model=config.GEMINI_MODEL,
                    reasoning_effort="none", # type: ignore
                    messages=conversation_messages, # type: ignore
                    max_tokens=config.MAX_TOKENS,
                    temperature=temperature,
                    stream=True,
                    stream_options={"include_usage": True}
                )
                
                usage_data = None
                
                async for chunk in stream:
                    if chunk.usage:
                        usage_data = chunk.usage
                    if chunk.choices[0].delta.content:
                        yield json.dumps({'content': chunk.choices[0].delta.content}) + '\n'
                        await asyncio.sleep(0)
                
                logger.info(f"⏱️  Streaming took {(time.time() - start) * 1000:.2f}ms")
                
                sources = [
                    {"chunk_id": doc["chunk_id"], "subject": doc["subject"], "topic": doc["topic"]}
                    for doc in reranked
                ]
                yield json.dumps({'sources': sources}) + '\n'
                # ✅ Send token usage (if available)
                if usage_data:
                    yield json.dumps({
                        'usage': {
                            'prompt_tokens': usage_data.prompt_tokens,
                            'completion_tokens': usage_data.completion_tokens,
                            'total_tokens': usage_data.total_tokens
                        }
                    }) + '\n'
                    logger.info(f"📊 Token usage: {usage_data.total_tokens} total tokens")
                
                yield json.dumps({'done': True}) + '\n'
            except Exception as e:
                logger.error(f"Streaming error: {e}", exc_info=True)
                yield json.dumps({'error': str(e)}) + '\n'

        return StreamingResponse(
            generate_stream(), 
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive"
            }
        )
    
    else:
        start = time.time()
        
        # ✅ Plain async call - no semaphore
        response = await openai_client.chat.completions.create(
            model=config.GEMINI_MODEL,
            reasoning_effort="none", # type: ignore
            messages=conversation_messages, # type: ignore
            max_tokens=config.MAX_TOKENS,
            temperature=temperature
        )
        
        logger.info(f"⏱️  LLM took {(time.time() - start) * 1000:.2f}ms")
        
        answer = response.choices[0].message.content
        logger.info(f"✅ Generated {len(answer)} characters") # type: ignore
        
        return {
            "answer": answer,
            "sources": [
                {
                    "chunk_id": doc["chunk_id"],
                    "subject": doc["subject"],
                    "topic": doc["topic"],
                    "relevance_score": doc.get("rerank_score", doc["score"])
                }
                for doc in reranked
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens, # type: ignore
                "completion_tokens": response.usage.completion_tokens, # type: ignore
                "total_tokens": response.usage.total_tokens # type: ignore
            }
        }

@app.post("/chat/markdown")
@observe()
async def chat_markdown_endpoint(request: ChatRequest):
    """Chat with RAG - returns clean markdown"""
    user_messages = [msg for msg in request.messages if msg.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")
    
    query = user_messages[-1].content
    logger.info(f"💬 Chat (Markdown): '{query[:100]}...'")
    
    query_vector = await get_embedding(query)
    results = await hybrid_search(
        query=query,
        query_vector=query_vector,
        top_k=config.TOP_K_RETRIEVAL,
        subject_filter=request.subject_filter,
        topic_filter=request.topic_filter,
        alpha=config.HYBRID_ALPHA
    )
    
    top_k = request.top_k or config.TOP_K_RERANK
    reranked = await rerank_documents(query, results, top_k) if results else []
    
    context = build_context(reranked)
    rag_prompt = create_rag_prompt(query, context)

    conversation_messages = [SYSTEM_MESSAGE]
    for msg in request.messages[:-1]:
        conversation_messages.append({"role": msg.role, "content": msg.content})
    conversation_messages.append({"role": "user", "content": rag_prompt})
    
    temperature = request.temperature or config.TEMPERATURE
    
    start = time.time()
    
    response = await openai_client.chat.completions.create(
        model=config.GEMINI_MODEL,
        messages=conversation_messages, # type: ignore
        max_tokens=config.MAX_TOKENS,
        temperature=temperature
    )
    
    logger.info(f"⏱️  LLM took {(time.time() - start) * 1000:.2f}ms")
    
    answer = response.choices[0].message.content
    logger.info(f"✅ Generated {len(answer)} characters") # type: ignore
    
    sources_md = "\n\n---\n\n## 📚 Sources\n\n"
    for i, doc in enumerate(reranked, 1):
        sources_md += f"{i}. **{doc['subject']}** - {doc['topic']} (Chunk: {doc['chunk_id']})\n"
    
    return PlainTextResponse(
        content=answer + sources_md, # type: ignore
        media_type="text/markdown",
        headers={
            "Content-Disposition": 'inline; filename="answer.md"'
        }
    )

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("RAG_PORT", "8000"))
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        workers=1,
        log_level="info"
    )
