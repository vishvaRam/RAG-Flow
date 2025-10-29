import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Load environment variables with type conversion
def get_env_int(key: str, default: int) -> int:
    """Get environment variable as integer"""
    value = os.getenv(key)
    return int(value) if value else default


def get_env_float(key: str, default: float) -> float:
    """Get environment variable as float"""
    value = os.getenv(key)
    return float(value) if value else default


def get_env_bool(key: str, default: bool) -> bool:
    """Get environment variable as boolean"""
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() in ('true', '1', 'yes', 'on')


# Configuration from environment variables
class Config:
    # Elasticsearch
    ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "localhost")
    ELASTICSEARCH_PORT = get_env_int("ELASTICSEARCH_PORT", 9200)
    ELASTICSEARCH_INDEX = os.getenv("ELASTICSEARCH_INDEX", "documents")
    
    # Gemini via OpenAI (for LLM only)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    
    # Google Embeddings via LangChain
    GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
    EMBEDDING_DIMS = get_env_int("EMBEDDING_DIMS", 768)
    
    # Query Rewriting
    ENABLE_QUERY_REWRITING = get_env_bool("ENABLE_QUERY_REWRITING", True)
    QUERY_REWRITE_MODEL = os.getenv("QUERY_REWRITE_MODEL", "gemini-2.5-flash-lite")
    QUERY_REWRITE_MAX_TOKENS = get_env_int("QUERY_REWRITE_MAX_TOKENS", 512)
    QUERY_REWRITE_TIMEOUT = get_env_float("QUERY_REWRITE_TIMEOUT", 5.0)
    
    # Reranker (LangChain CrossEncoder model)
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2")
    
    # RAG settings
    TOP_K_RETRIEVAL = get_env_int("TOP_K_RETRIEVAL", 15)
    TOP_K_RERANK = get_env_int("TOP_K_RERANK", 5)
    HYBRID_ALPHA = get_env_float("HYBRID_ALPHA", 0.5)
    
    # API settings
    MAX_TOKENS = get_env_int("MAX_TOKENS", 2048)
    TEMPERATURE = get_env_float("TEMPERATURE", 0.7)
    
    # LangFuse settings
    LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
    LANGFUSE_DEBUG: bool = get_env_bool("LANGFUSE_DEBUG", False)


config = Config()
