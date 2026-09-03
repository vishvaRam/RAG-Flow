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
    LLM_PROVIDER_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    LLM_API_KEY: str
    LLM_MODEL: str = "gemini-3.5-flash-lite"
    MAX_TOKENS: int = 4096
    TEMPERATURE: float = 0.4
    LLM_TIMEOUT: float = 30.0

    # Embeddings
    EMBEDDING_PROVIDER_URL: str = (
        "https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    EMBEDDING_DIMENSIONS: int = 1024

    # RAG Settings
    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 250
    TOP_K_RETRIEVAL: int = 6
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
    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_PROJECT: str = "LangGraph-RAG"
    LANGSMITH_TRACING: bool = True

    # History & Summarization
    MAX_HISTORY_MESSAGES: int = 20
    SUMMARY_INTERVAL: int = 5
    SUMMARY_MODEL: str = "gemini-3.1-flash-lite"
    SUMMARY_MAX_TOKENS: int = 512

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/rag_api.log"

    @property
    def asyncpg_url(self) -> str:
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @property
    def sqlalchemy_async_url(self) -> str:
        return f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    def configure_langsmith(self) -> None:
        if self.LANGSMITH_TRACING and self.LANGSMITH_API_KEY:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.LANGSMITH_API_KEY
            os.environ["LANGCHAIN_PROJECT"] = self.LANGSMITH_PROJECT
            os.environ["LANGCHAIN_ENDPOINT"] = self.LANGSMITH_ENDPOINT


@lru_cache()
def get_settings() -> Settings:
    return Settings()
