import os
import logging
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from elasticsearch import Elasticsearch
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
import json
import time
from functools import wraps

# Langfuse imports
from langfuse import get_client, observe
from langfuse.openai import openai

# LangChain and Utils
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from Utils.Config import config
from Utils.models import ChatRequest, SearchRequest, SearchResponse, SearchResult


# Initialize Langfuse client
langfuse = get_client()


# Create logs directory if it doesn't exist
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


# Timing decorator for async functions
def log_time(operation_name: str):
    """Decorator to log execution time of async functions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = (time.time() - start_time) * 1000
                logger.info(f"⏱️  {operation_name} took {duration:.2f}ms")
                return result
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                logger.error(f"⏱️  {operation_name} failed after {duration:.2f}ms: {str(e)}")
                raise
        return wrapper
    return decorator


# Validate required environment variables
if not config.GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")
if not config.GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is required")


# Initialize OpenAI client with Langfuse wrapper
openai_client = openai.OpenAI(
    api_key=config.GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


# Initialize Elasticsearch client
es_client = Elasticsearch([f"http://{config.ELASTICSEARCH_HOST}:{config.ELASTICSEARCH_PORT}"])


# Initialize Google Embeddings model
embeddings_model = GoogleGenerativeAIEmbeddings(
    model=config.GEMINI_EMBEDDING_MODEL,
    google_api_key=config.GOOGLE_API_KEY,
)


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("Starting RAG API service...")
    logger.info(f"Configuration loaded:")
    logger.info(f"  - Elasticsearch: {config.ELASTICSEARCH_HOST}:{config.ELASTICSEARCH_PORT}")
    logger.info(f"  - Index: {config.ELASTICSEARCH_INDEX}")
    logger.info(f"  - Gemini Model: {config.GEMINI_MODEL}")
    logger.info(f"  - Embedding Model: {config.GEMINI_EMBEDDING_MODEL}")
    logger.info(f"  - Embedding Dimensions: {config.EMBEDDING_DIMS}")
    logger.info(f"  - Reranker URL: {config.RERANKER_URL}")
    
    # Check Langfuse connection
    try:
        if langfuse.auth_check():
            logger.info(f"✓ Connected to Langfuse at {config.LANGFUSE_HOST}")
        else:
            logger.warning("⚠ Langfuse authentication failed - check credentials")
    except Exception as e:
        logger.error(f"✗ Langfuse connection failed: {e}")
        logger.warning("⚠ Continuing without Langfuse tracing")
    
    # Check Elasticsearch connection
    try:
        info = es_client.info()
        logger.info(f"✓ Connected to Elasticsearch {info['version']['number']}")
    except Exception as e:
        logger.error(f"✗ Elasticsearch connection failed: {e}")
        raise
    
    # Check reranker service
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{config.RERANKER_URL}/health", timeout=5.0)
            if response.status_code == 200:
                logger.info("✓ Reranker service is healthy")
            else:
                logger.warning("⚠ Reranker service returned non-200 status")
    except Exception as e:
        logger.error(f"✗ Reranker service connection failed: {e}")
    
    logger.info("✓ RAG API service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Flushing Langfuse traces...")
    try:
        langfuse.flush()
        logger.info("✓ Langfuse traces flushed")
    except Exception as e:
        logger.error(f"Error flushing Langfuse: {e}")
    
    logger.info("Shutting down RAG API service...")
    es_client.close()
    logger.info("✓ RAG API service shut down")


# Initialize FastAPI app
app = FastAPI(
    title="Hybrid RAG API with Gemini",
    description="RAG system using Elasticsearch hybrid search, cross-encoder reranking, and Gemini models",
    version="1.0.0",
    lifespan=lifespan
)


# Middleware to log request processing time
@app.middleware("http")
async def log_request_time(request: Request, call_next):
    """Log total request processing time"""
    start_time = time.time()
    logger.info(f"🔵 Request: {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    duration = (time.time() - start_time) * 1000
    logger.info(f"🟢 Response: {request.method} {request.url.path} - Status: {response.status_code} - Duration: {duration:.2f}ms")
    response.headers["X-Process-Time"] = f"{duration:.2f}ms"
    
    return response


# Helper functions with timing and observability
@observe()
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
@log_time("Embedding generation (query)")
async def get_embedding(text: str, task_type: str = "retrieval_query") -> List[float]:
    """Generate embedding using Google Generative AI Embeddings via LangChain"""
    try:
        if task_type == "retrieval_query":
            embedding = embeddings_model.embed_query(
                text,
                output_dimensionality=config.EMBEDDING_DIMS,
                task_type=task_type
            )
        else:
            embeddings = embeddings_model.embed_documents(
                [text],
                output_dimensionality=config.EMBEDDING_DIMS,
                task_type=task_type
            )
            embedding = embeddings[0]
        
        logger.info(f"📊 Generated {len(embedding)}-dimensional embedding")
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise


@observe()
@log_time("Elasticsearch hybrid search")
async def hybrid_search(
    query: str,
    query_vector: List[float],
    top_k: int = 20,
    subject_filter: Optional[str] = None,
    topic_filter: Optional[str] = None,
    alpha: float = 0.5
) -> List[Dict[str, Any]]:
    """Perform hybrid search combining BM25 and vector search"""
    
    filter_conditions = []
    if subject_filter:
        filter_conditions.append({"term": {"subject": subject_filter}})
    if topic_filter:
        filter_conditions.append({"term": {"topic": topic_filter}})
    
    must_clause = filter_conditions if filter_conditions else []
    
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
                            "query": {
                                "bool": {
                                    "must": must_clause if must_clause else [{"match_all": {}}]
                                }
                            },
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'dense_vector') + 1.0",
                                "params": {"query_vector": query_vector}
                            },
                            "boost": alpha
                        }
                    }
                ],
                "must": must_clause,
                "minimum_should_match": 1
            }
        },
        "_source": ["text", "subject", "topic", "file_path", "chunk_id"]
    }
    
    try:
        response = es_client.search(index=config.ELASTICSEARCH_INDEX, body=search_body)
        
        results = []
        for hit in response['hits']['hits']:
            results.append({
                "text": hit['_source']['text'],
                "score": hit['_score'],
                "subject": hit['_source']['subject'],
                "topic": hit['_source']['topic'],
                "file_path": hit['_source']['file_path'],
                "chunk_id": hit['_source']['chunk_id']
            })
        
        logger.info(f"📊 Retrieved {len(results)} documents from Elasticsearch")
        return results
    
    except Exception as e:
        logger.error(f"Hybrid search error: {e}")
        raise


@observe()
@log_time("Document reranking")
async def rerank_documents(query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """Rerank documents using cross-encoder"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{config.RERANKER_URL}/rerank",
                json={
                    "query": query,
                    "documents": [doc["text"] for doc in documents],
                    "top_k": top_k
                }
            )
            
            if response.status_code != 200:
                logger.warning(f"Reranker returned status {response.status_code}, using original ranking")
                return documents[:top_k]
            
            reranked_results = response.json()["results"]
            
            reranked_docs = []
            for result in reranked_results:
                original_doc = documents[result["index"]]
                reranked_docs.append({
                    **original_doc,
                    "rerank_score": result["score"]
                })
            
            logger.info(f"📊 Reranked to top {len(reranked_docs)} documents")
            return reranked_docs
    
    except Exception as e:
        logger.error(f"Reranking error: {e}, falling back to original ranking")
        return documents[:top_k]


def build_context(documents: List[Dict[str, Any]]) -> str:
    """Build context string from retrieved documents"""
    context_parts = []
    for i, doc in enumerate(documents, 1):
        context_parts.append(
            f"[Document {i}]\n"
            f"Subject: {doc['subject']} | Topic: {doc['topic']}\n"
            f"Source: {doc['chunk_id']}\n"
            f"Content: {doc['text']}\n"
        )
    return "\n\n".join(context_parts)


def create_rag_prompt(query: str, context: str) -> str:
    """Create RAG prompt with context for JEE exam preparation"""
    return f"""You are an expert JEE (Joint Entrance Examination) tutor with deep knowledge of Physics, Chemistry, and Mathematics. Your role is to provide accurate, exam-focused explanations based on standard JEE textbooks and syllabus.

    Context Information:
    {context}

    Student Question: {query}

    Instructions:
    - Provide accurate answers based ONLY on the information given in the context above
    - If the context doesn't contain sufficient information to answer the question, clearly state "I don't have enough information to answer this question accurately"
    - Use proper mathematical notation with LaTeX formatting for all formulas and equations for markdown rendering
    - Explain concepts step-by-step when solving problems
    - Include relevant formulas, derivations, and numerical calculations where applicable
    - Use SI units and standard notation as per JEE guidelines
    - Be precise with terminology, constants, and units
    - Format chemical equations properly
    - For problem-solving, show clear steps: Given → Formula → Substitution → Calculation → Final Answer
    - Do NOT make assumptions or add information not present in the context
    - Keep explanations clear, concise, and exam-oriented
    - Never reference the context directly in your answers

    Answer:"""


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Hybrid RAG API with Gemini",
        "version": "1.0.0",
        "model": config.GEMINI_MODEL,
        "embedding_model": config.GEMINI_EMBEDDING_MODEL,
        "embedding_dims": config.EMBEDDING_DIMS,
        "langfuse_enabled": bool(config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY),
        "endpoints": {
            "/chat": "Chat with RAG",
            "/search": "Search documents",
            "/health": "Health check",
            "/docs": "API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "elasticsearch": "unknown",
        "reranker": "unknown",
        "langfuse": "unknown",
        "config": {
            "index": config.ELASTICSEARCH_INDEX,
            "model": config.GEMINI_MODEL,
            "embedding_model": config.GEMINI_EMBEDDING_MODEL,
            "embedding_dims": config.EMBEDDING_DIMS
        }
    }
    
    try:
        es_client.cluster.health()
        health_status["elasticsearch"] = "healthy"
    except Exception as e:
        health_status["elasticsearch"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{config.RERANKER_URL}/health")
            if response.status_code == 200:
                health_status["reranker"] = "healthy"
            else:
                health_status["reranker"] = f"unhealthy: status {response.status_code}"
                health_status["status"] = "degraded"
    except Exception as e:
        health_status["reranker"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    try:
        if langfuse.auth_check():
            health_status["langfuse"] = "healthy"
        else:
            health_status["langfuse"] = "authentication failed"
    except Exception as e:
        health_status["langfuse"] = f"unhealthy: {str(e)}"
    
    return health_status


@app.post("/search", response_model=SearchResponse)
@observe()
async def search_endpoint(request: SearchRequest):
    """Search documents using hybrid search and reranking"""
    try:
        logger.info(f"🔍 Search query: '{request.query}' (top_k={request.top_k})")
        
        query_vector = await get_embedding(request.query, task_type="retrieval_query")
        
        results = await hybrid_search(
            query=request.query,
            query_vector=query_vector,
            top_k=config.TOP_K_RETRIEVAL,
            subject_filter=request.subject_filter,
            topic_filter=request.topic_filter,
            alpha=config.HYBRID_ALPHA
        )
        
        if results:
            reranked_results = await rerank_documents(
                query=request.query,
                documents=results,
                top_k=request.top_k
            )
        else:
            reranked_results = []
        
        search_results = [
            SearchResult(
                text=doc["text"][:500] + "..." if len(doc["text"]) > 500 else doc["text"],
                score=doc.get("rerank_score", doc["score"]),
                subject=doc["subject"],
                topic=doc["topic"],
                file_path=doc["file_path"],
                chunk_id=doc["chunk_id"]
            )
            for doc in reranked_results
        ]
        
        logger.info(f"✅ Search completed: returned {len(search_results)} results")
        
        return SearchResponse(
            query=request.query,
            results=search_results,
            total_found=len(results)
        )
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/chat")
@observe()
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint with RAG"""
    try:
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")
        
        query = user_messages[-1].content
        logger.info(f"💬 Chat query: '{query[:100]}...'")
        
        query_vector = await get_embedding(query, task_type="retrieval_query")
        
        results = await hybrid_search(
            query=query,
            query_vector=query_vector,
            top_k=config.TOP_K_RETRIEVAL,
            subject_filter=request.subject_filter,
            topic_filter=request.topic_filter,
            alpha=config.HYBRID_ALPHA
        )
        
        top_k = request.top_k or config.TOP_K_RERANK
        if results:
            reranked_results = await rerank_documents(
                query=query,
                documents=results,
                top_k=top_k
            )
        else:
            reranked_results = []
        
        context = build_context(reranked_results)
        rag_prompt = create_rag_prompt(query, context)
        
        gemini_messages = [
            {"role": "system", "content": "You are a helpful educational assistant with access to course materials."}
        ]
        
        for msg in request.messages[:-1]:
            gemini_messages.append({"role": msg.role, "content": msg.content})
        
        gemini_messages.append({"role": "user", "content": rag_prompt})
        
        temperature = request.temperature or config.TEMPERATURE
        
        if request.stream:
            async def generate_stream():
                try:
                    llm_start = time.time()
                    stream = openai_client.chat.completions.create(
                        model=config.GEMINI_MODEL,
                        messages=gemini_messages,
                        max_tokens=config.MAX_TOKENS,
                        temperature=temperature,
                        stream=True
                    )
                    
                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            yield f"data: {json.dumps({'content': chunk.choices[0].delta.content})}\n\n"
                    
                    llm_duration = (time.time() - llm_start) * 1000
                    logger.info(f"⏱️  LLM generation (streaming) took {llm_duration:.2f}ms")
                    
                    sources = [
                        {
                            "chunk_id": doc["chunk_id"],
                            "subject": doc["subject"],
                            "topic": doc["topic"]
                        }
                        for doc in reranked_results
                    ]
                    yield f"data: {json.dumps({'sources': sources})}\n\n"
                    yield "data: [DONE]\n\n"
                
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        
        else:
            llm_start = time.time()
            response = openai_client.chat.completions.create(
                model=config.GEMINI_MODEL,
                messages=gemini_messages,
                max_tokens=config.MAX_TOKENS,
                temperature=temperature
            )
            llm_duration = (time.time() - llm_start) * 1000
            logger.info(f"⏱️  LLM generation took {llm_duration:.2f}ms")
            
            answer = response.choices[0].message.content
            
            logger.info(f"✅ Chat completed: generated {len(answer)} characters")
            
            return {
                "answer": answer,
                "sources": [
                    {
                        "chunk_id": doc["chunk_id"],
                        "subject": doc["subject"],
                        "topic": doc["topic"],
                        "relevance_score": doc.get("rerank_score", doc["score"])
                    }
                    for doc in reranked_results
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


if __name__ == "__main__":
    port = int(os.getenv("RAG_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
