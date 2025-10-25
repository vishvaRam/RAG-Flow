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
from fastapi.responses import StreamingResponse
from elasticsearch import Elasticsearch

# Langfuse imports
from langfuse import get_client, observe
from langfuse.openai import openai

# LangChain imports (minimal - only for embeddings and reranking)
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

# Create logs directory
os.makedirs('logs', exist_ok=True)

# Configure logging
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
es_client = Elasticsearch([f"http://{config.ELASTICSEARCH_HOST}:{config.ELASTICSEARCH_PORT}"])
openai_client = openai.OpenAI(
    api_key=config.GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Initialize embeddings model
embeddings_model = GoogleGenerativeAIEmbeddings(
    model=config.GEMINI_EMBEDDING_MODEL,
    google_api_key=config.GOOGLE_API_KEY,
)

# Initialize reranker (load once at startup)
logger.info(f"Loading reranker: {config.RERANKER_MODEL}")
cross_encoder = HuggingFaceCrossEncoder(model_name=config.RERANKER_MODEL)
reranker = CrossEncoderReranker(model=cross_encoder, top_n=config.TOP_K_RERANK)
logger.info("✓ Reranker loaded")


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
# CORE RAG FUNCTIONS
# ============================================================================

@observe()
@log_time("Embedding generation")
async def get_embedding(text: str) -> List[float]:
    """Generate query embedding"""
    embedding = await asyncio.to_thread(
        embeddings_model.embed_query,
        text,
        output_dimensionality=config.EMBEDDING_DIMS,
        task_type="retrieval_query"
    )
    logger.info(f"📊 Generated {len(embedding)}-dimensional embedding")
    return embedding


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
    """Perform hybrid search (BM25 + vector search)"""
    
    # Build filters
    filters = []
    if subject_filter:
        filters.append({"term": {"subject": subject_filter}})
    if topic_filter:
        filters.append({"term": {"topic": topic_filter}})
    
    # Hybrid search query
    search_body = {
        "size": top_k,
        "query": {
            "bool": {
                "should": [
                    # BM25 keyword search
                    {
                        "match": {
                            "text": {
                                "query": query,
                                "boost": 1 - alpha
                            }
                        }
                    },
                    # Vector similarity search
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
    
    response = await asyncio.to_thread(es_client.search, index=config.INDEX_NAME, body=search_body)
    
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


@observe()
@log_time("Reranking")
async def rerank_documents(query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """Rerank documents using CrossEncoder"""
    if not documents:
        return []
    
    # Convert to LangChain Document format
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
    
    # Rerank
    reranked_docs = await asyncio.to_thread(
        reranker.compress_documents,
        documents=langchain_docs,
        query=query
    )
    
    # Convert back to dict format
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
    return f"""You are an expert JEE (Joint Entrance Examination) tutor with deep knowledge of Physics, Chemistry, and Mathematics.

Context Information:
{context}

Student Question: {query}

Instructions:
- Answer ONLY based on the context provided
- If context is insufficient, state "I don't have enough information to answer this accurately"
- Use step-by-step explanations for problem-solving
- Include formulas, derivations, and calculations where applicable
- Use SI units and standard JEE notation
- Format chemical equations properly
- Show clear steps: Given → Formula → Substitution → Calculation → Final Answer
- Keep explanations clear, concise, and exam-oriented

Answer:"""


# ============================================================================
# FASTAPI APP
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("Starting RAG API service...")
    logger.info(f"  - Elasticsearch: {config.ELASTICSEARCH_HOST}:{config.ELASTICSEARCH_PORT}")
    logger.info(f"  - Index: {config.INDEX_NAME}")
    logger.info(f"  - Model: {config.GEMINI_MODEL}")
    logger.info(f"  - Reranker: {config.RERANKER_MODEL}")
    
    # Validate connections
    try:
        if langfuse.auth_check():
            logger.info(f"✓ Langfuse connected")
    except Exception as e:
        logger.warning(f"⚠ Langfuse unavailable: {e}")
    
    try:
        info = es_client.info()
        count = es_client.count(index=config.INDEX_NAME)
        logger.info(f"✓ Elasticsearch {info['version']['number']} - {count['count']} documents")
    except Exception as e:
        logger.error(f"✗ Elasticsearch connection failed: {e}")
        raise
    
    logger.info("✓ Service started")
    yield
    
    # Shutdown
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
        count = es_client.count(index=config.INDEX_NAME)
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
    
    # Get embedding
    query_vector = await get_embedding(request.query)
    
    # Hybrid search
    results = await hybrid_search(
        query=request.query,
        query_vector=query_vector,
        top_k=config.TOP_K_RETRIEVAL,
        subject_filter=request.subject_filter,
        topic_filter=request.topic_filter,
        alpha=config.HYBRID_ALPHA
    )
    
    # Rerank
    reranked = await rerank_documents(request.query, results, request.top_k) if results else []
    
    # Format response
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
    # Extract query
    user_messages = [msg for msg in request.messages if msg.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")
    
    query = user_messages[-1].content
    logger.info(f"💬 Chat: '{query[:100]}...'")
    
    # Get embedding and search
    query_vector = await get_embedding(query)
    results = await hybrid_search(
        query=query,
        query_vector=query_vector,
        top_k=config.TOP_K_RETRIEVAL,
        subject_filter=request.subject_filter,
        topic_filter=request.topic_filter,
        alpha=config.HYBRID_ALPHA
    )
    
    # Rerank
    top_k = request.top_k or config.TOP_K_RERANK
    reranked = await rerank_documents(query, results, top_k) if results else []
    
    # Build prompt
    context = build_context(reranked)
    rag_prompt = create_rag_prompt(query, context)
    
    messages = [{"role": "system", "content": "You are a helpful educational assistant."}]
    for msg in request.messages[:-1]:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": rag_prompt})
    
    temperature = request.temperature or config.TEMPERATURE
    
    # Streaming response
    if request.stream:
        async def generate_stream():
            try:
                start = time.time()
                stream = openai_client.chat.completions.create(
                    model=config.GEMINI_MODEL,
                    messages=messages,
                    max_tokens=config.MAX_TOKENS,
                    temperature=temperature,
                    stream=True
                )
                
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield f"data: {json.dumps({'content': chunk.choices[0].delta.content})}\n\n"
                
                logger.info(f"⏱️  Streaming took {(time.time() - start) * 1000:.2f}ms")
                
                sources = [
                    {"chunk_id": doc["chunk_id"], "subject": doc["subject"], "topic": doc["topic"]}
                    for doc in reranked
                ]
                yield f"data: {json.dumps({'sources': sources})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return StreamingResponse(generate_stream(), media_type="text/event-stream")
    
    # Non-streaming response
    else:
        start = time.time()
        response = openai_client.chat.completions.create(
            model=config.GEMINI_MODEL,
            messages=messages,
            max_tokens=config.MAX_TOKENS,
            temperature=temperature
        )
        logger.info(f"⏱️  LLM took {(time.time() - start) * 1000:.2f}ms")
        
        answer = response.choices[0].message.content
        logger.info(f"✅ Generated {len(answer)} characters")
        
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
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("RAG_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
