import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.logging import logger
from app.api.routes import chat, search, health
from app.api.dependencies import init_services, cleanup_services, get_search_service

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    # ✅ Initialize all services ONCE at startup
    init_services()
    
    # Health check
    search_svc = get_search_service()
    es_health = await search_svc.health_check()
    logger.info(f"✓ Elasticsearch: {es_health['status']} ({es_health.get('document_count', 0)} documents)")
    
    logger.info("✓ Service started and ready to accept requests")
    yield
    
    # ✅ Cleanup on shutdown
    cleanup_services()
    logger.info("✓ Service stopped")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="RAG system with Elasticsearch hybrid search and CrossEncoder reranking",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    logger.info(f"🔵 {request.method} {request.url.path}")
    response = await call_next(request)
    duration = (time.time() - start) * 1000
    logger.info(f"🟢 {request.method} {request.url.path} - {response.status_code} - {duration:.2f}ms")
    response.headers["X-Process-Time"] = f"{duration:.2f}ms"
    return response

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(search.router, tags=["Search"])
app.include_router(chat.router, tags=["Chat"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "model": settings.GEMINI_MODEL,
        "embedding_model": settings.GEMINI_EMBEDDING_MODEL,
        "reranker": settings.RERANKER_MODEL,
        "endpoints": {
            "/chat": "Chat with RAG",
            "/search": "Search documents",
            "/health": "Health check",
            "/docs": "API documentation"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        log_level=settings.LOG_LEVEL.lower()
    )
