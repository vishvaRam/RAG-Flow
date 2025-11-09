import os
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Settings
    APP_NAME: str = "Hybrid RAG API"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 4545
    
    # Elasticsearch
    ELASTICSEARCH_HOST: str = "localhost"
    ELASTICSEARCH_PORT: int = 9200
    ELASTICSEARCH_INDEX: str = "documents"
    ELASTICSEARCH_TIMEOUT: int = 30
    ELASTICSEARCH_MAX_RETRIES: int = 3
    # ELASTICSEARCH_POOL_SIZE: int = 25
    
    # Gemini/OpenAI Configuration
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-2.5-flash-lite"
    GEMINI_EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    GEMINI_TIMEOUT: float = 20.0
    GEMINI_MAX_RETRIES: int = 1
    
    # Embeddings
    EMBEDDING_DIMS: int = 768
    
    # Query Rewriting
    ENABLE_QUERY_REWRITING: bool = True
    QUERY_REWRITE_MODEL: str = "gemini-2.5-flash-lite"
    QUERY_REWRITE_MAX_TOKENS: int = 256
    QUERY_REWRITE_TIMEOUT: float = 5.0
    
    # Reranker
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L2-v2"
    
    # RAG Settings
    TOP_K_RETRIEVAL: int = 15
    TOP_K_RERANK: int = 5
    HYBRID_ALPHA: float = 0.5
    
    # LLM Settings
    MAX_TOKENS: int = 2048
    TEMPERATURE: float = 0.7
    
    # Langfuse
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    LANGFUSE_DEBUG: bool = False
    
    DB_HOST: str = "localhost"
    DB_PORT: int = 5433
    DB_NAME: str = 'rag_database'
    DB_USER: str = 'postgres'
    DB_PASSWORD: str = 'admin'
    DB_MIN_CONNECTIONS: int = 2
    DB_MAX_CONNECTIONS: int = 10
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/rag_api.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton"""
    return Settings()
