import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.utils.prompts import JEE_CONTEXT_PROMPT, JEE_SYS_PROMPT


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )

    # App / API Settings
    APP_NAME: str = "LangGraph RAG API"
    APP_VERSION: str = "4.0.0"
    DEBUG: bool = False
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 4545

    # Prompts
    JEE_SYSTEM_PROMPT: str = JEE_SYS_PROMPT
    JEE_CONTEXT_PROMPT: str = JEE_CONTEXT_PROMPT

    # LLM Configuration
    LLM_PROVIDER_URL: str = "https://openrouter.ai/api/v1"
    LLM_API_KEY: str
    LLM_MODEL: str = "qwen/qwen3.8-flash"
    MAX_TOKENS: int = 8096
    TEMPERATURE: float = 0.4
    LLM_TIMEOUT: float = 30.0

    # Embeddings
    EMBEDDING_PROVIDER_URL: str = (
        "https://openrouter.ai/api/v1"
    )
    EMBEDDING_MODEL: str = "google/gemini-embedding-001"
    EMBEDDING_DIMENSIONS: int = 1024

    # Reranker
    RERANKER_ENABLE: bool = True
    RERANKER_PROVIDER_URL: str = "https://openrouter.ai/api/v1"
    RERANKER_MODEL: str = "qwen/qwen3-reranker-0.6b"
    RERANK_TOP_K: int = 8

    # RAG Settings
    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 250
    TOP_K_RETRIEVAL: int = 25
    COLLECTION_NAME: str = "pdf_knowledge_base"

    # Database
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "rag_database"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "admin"
    DB_MIN_CONNECTIONS: int = 2
    DB_MAX_CONNECTIONS: int = 10

    HISTORY_TABLE: str = "chat_messages_history"
    SUMMARY_TABLE: str = "chat_session_summaries"

    # LangSmith
    LANGFUSE_SECRET_KEY: str
    LANGFUSE_PUBLIC_KEY: str
    LANGFUSE_BASE_URL: str = "https://cloud.langfuse.com"
    LANGFUSE_TRACING: bool = True

    # History & Summarization
    MAX_HISTORY_MESSAGES: int = 20
    SUMMARY_INTERVAL: int = 5
    SUMMARY_MODEL: str = "google/gemini-3.1-flash-lite"
    SUMMARY_MAX_TOKENS: int = 2048

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/rag_api.log"

    @property
    def asyncpg_url(self) -> str:
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @property
    def sqlalchemy_async_url(self) -> str:
        return f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    def configure_langfuse(self) -> None:
        if self.LANGFUSE_TRACING and self.LANGFUSE_SECRET_KEY:
            os.environ["LANGFUSE_SECRET_KEY"] = self.LANGFUSE_SECRET_KEY
            os.environ["LANGFUSE_PUBLIC_KEY"] = self.LANGFUSE_PUBLIC_KEY
            # LANGFUSE_HOST is the current SDK variable; keep BASE_URL for
            # compatibility with older Langfuse integrations.
            os.environ["LANGFUSE_HOST"] = self.LANGFUSE_BASE_URL
            os.environ["LANGFUSE_BASE_URL"] = self.LANGFUSE_BASE_URL
            os.environ["LANGFUSE_TRACING"] = "true"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
