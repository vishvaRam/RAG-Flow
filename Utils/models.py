
from pydantic import BaseModel, Field
from typing import List, Optional

# Pydantic models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="Conversation history")
    stream: bool = Field(default=False, description="Enable streaming response")
    subject_filter: Optional[str] = Field(None, description="Filter by subject")
    topic_filter: Optional[str] = Field(None, description="Filter by topic")
    top_k: Optional[int] = Field(None, description="Override top_k reranking")
    temperature: Optional[float] = Field(None, description="Override temperature")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=5, description="Number of results")
    subject_filter: Optional[str] = None
    topic_filter: Optional[str] = None


class SearchResult(BaseModel):
    text: str
    score: float
    subject: str
    topic: str
    file_path: str
    chunk_id: str


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_found: int