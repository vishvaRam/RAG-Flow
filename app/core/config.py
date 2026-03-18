from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings
from app.utils.prompts import JEE_CONTEXT_PROMPT, JEE_SYS_PROMPT


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    ############################
    # 🚀 App / API Settings
    ############################
    APP_NAME: str = "Hybrid RAG API"
    APP_VERSION: str = "3.0.0"
    DEBUG: bool = False
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 4545

    ############################
    # 🧠 Prompts
    ############################
    JEE_SYSTEM_PROMPT: str = JEE_SYS_PROMPT
    JEE_CONTEXT_PROMPT: str = JEE_CONTEXT_PROMPT

    ############################
    # 🔍 Elasticsearch
    ############################
    ELASTICSEARCH_HOST: str = "localhost"
    ELASTICSEARCH_PORT: int = 9200
    ELASTICSEARCH_INDEX: str = "documents"
    ELASTICSEARCH_TIMEOUT: int = 30
    ELASTICSEARCH_MAX_RETRIES: int = 3

    ############################
    # 🤖 LLM Configuration
    ############################
    LLM_PROVIDER_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    LLM_API_KEY: str  # Required
    LLM_MODEL: str = "gemini-3.1-flash-lite-preview"

    MAX_TOKENS: int = 4096
    TEMPERATURE: float = 0.4

    LLM_TIMEOUT: float = 20.0
    LLM_MAX_RETRIES: int = 2

    ############################
    # 🔗 Embeddings
    ############################
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    EMBEDDING_DIMS: int = 768

    ############################
    # 🔄 Query Rewriting
    ############################
    ENABLE_QUERY_REWRITING: bool = True
    QUERY_REWRITE_MODEL: str = "gemini-2.5-flash-lite"
    QUERY_REWRITE_MAX_TOKENS: int = 512
    QUERY_REWRITE_TIMEOUT: float = 5.0
    QUERY_REWRITE_WITH_HISTORY_COUNT: int = 5

    ############################
    # 🧠 Reranker
    ############################
    RERANKER_ENABLE: bool = False
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L2-v2"
    TOP_K_RERANK: int = 3

    ############################
    # 📚 RAG Settings
    ############################
    TOP_K_RETRIEVAL: int = 8
    HYBRID_ALPHA: float = 0.5

    ############################
    # 📊 LangSmith
    ############################
    LANGSMITH_API_KEY: Optional[str] = None
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_PROJECT: str = "Dream-RAG"
    LANGSMITH_TRACING: bool = True

    ############################
    # 🗄️ Database
    ############################
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "rag_database"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "admin"

    DB_MIN_CONNECTIONS: int = 2
    DB_MAX_CONNECTIONS: int = 10

    HISTORY_TABLE: str = "chat_messages_history"
    SUMMARY_TABLE: str = "chat_session_summaries"

    ############################
    # 🧾 History & Summarization
    ############################
    MAX_HISTORY_MESSAGES: int = 10
    SUMMARY_INTERVAL: int = 5

    SUMMARY_MODEL: str = "gemini-2.5-flash-lite"
    SUMMARY_MAX_TOKENS: int = 512

    ############################
    # 🪵 Logging
    ############################
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/rag_api.log"

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton instance of settings"""
    return Settings()
