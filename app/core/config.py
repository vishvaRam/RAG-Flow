from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from app.utils.prompts import JEE_CONTEXT_PROMPT, JEE_SYS_PROMPT

load_dotenv()


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # API Settings
    APP_NAME: str = "Hybrid RAG API"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 4545

    # Prompts
    JEE_SYSTEM_PROMPT: str = JEE_SYS_PROMPT
    JEE_CONTEXT_PROMPT: str = JEE_CONTEXT_PROMPT

    # Elasticsearch
    ELASTICSEARCH_HOST: str = "localhost"
    ELASTICSEARCH_PORT: int = 9200
    ELASTICSEARCH_INDEX: str = "documents"
    ELASTICSEARCH_TIMEOUT: int = 30
    ELASTICSEARCH_MAX_RETRIES: int = 3

    # Gemini/OpenAI Configuration
    LLM_PROVIDER_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    LLM_API_KEY: str
    LLM_MODEL: str = "gemini-2.5-flash"
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    LLM_TIMEOUT: float = 20.0
    LLM_MAX_RETRIES: int = 2

    # Embeddings
    EMBEDDING_DIMS: int = 768

    # Query Rewriting
    ENABLE_QUERY_REWRITING: bool = True
    QUERY_REWRITE_MODEL: str = "gemini-2.5-flash-lite"
    QUERY_REWRITE_MAX_TOKENS: int = 256
    QUERY_REWRITE_TIMEOUT: float = 5.0
    QUERY_REWRITE_WITH_HISTORY_COUNT: int = 5

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
    LANGFUSE_TRACING_ENABLED: bool = True

    # Database Settings
    DB_HOST: str = "localhost"
    DB_PORT: int = 5433
    DB_NAME: str = "rag_database"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "admin"
    SUMMARY_TABLE: str = "chat_session_summaries"
    HISTORY_TABLE: str = "chat_messages_history"
    DB_MIN_CONNECTIONS: int = 2
    DB_MAX_CONNECTIONS: int = 10

    # History and Summary Settings
    MAX_HISTORY_MESSAGES: int = 10
    SUMMARY_INTERVAL: int = 10
    SUMMARY_MODEL: str = "gemini-2.5-flash-lite"
    SUMMARY_MAX_TOKENS: int = 512

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
