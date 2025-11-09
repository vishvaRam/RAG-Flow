from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime


class ChatMessage(BaseModel):
    """Chat message schema"""
    role: str = Field(..., description="Role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Chat request schema"""
    messages: List[ChatMessage] = Field(..., description="Conversation history")
    stream: bool = Field(default=False, description="Enable streaming response")
    subject_filter: Optional[str] = Field(None, description="Filter by subject")
    topic_filter: Optional[str] = Field(None, description="Filter by topic")
    top_k: Optional[int] = Field(None, description="Override top_k reranking")
    temperature: Optional[float] = Field(None, description="Override temperature")


class ChatRequestSession(BaseModel):
    """Chat request schema with session"""
    session_id: str = Field(..., description="Unique session ID")
    user_id: str = Field(..., description="Unique user ID")
    assistant_id: str = Field(..., description="Unique assistant ID")
    messages: List[ChatMessage] = Field(..., description="Conversation history")
    stream: bool = Field(default=False, description="Enable streaming response")
    subject_filter: Optional[str] = Field(None, description="Filter by subject")
    topic_filter: Optional[str] = Field(None, description="Filter by topic")
    top_k: Optional[int] = Field(None, description="Override top_k reranking")
    temperature: Optional[float] = Field(None, description="Override temperature")


class SearchRequest(BaseModel):
    """Search request schema"""
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=5, description="Number of results")
    subject_filter: Optional[str] = None
    topic_filter: Optional[str] = None


class SearchResult(BaseModel):
    """Search result schema"""
    text: str
    score: float
    subject: str
    topic: str
    file_path: str
    chunk_id: str


class SearchResponse(BaseModel):
    """Search response schema"""
    query: str
    results: List[SearchResult]
    total_found: int


class ChatResponse(BaseModel):
    """Chat response schema"""
    answer: str
    sources: List[dict]
    usage: dict


# ==================== Database Models ====================

class ChatMessageDB(BaseModel):
    """Base database model for chat messages"""
    session_id: str
    sender_id: str
    sender_type: Literal["user", "assistant"]
    message: str


class ChatMessageCreateDB(ChatMessageDB):
    """Database model for creating new chat messages"""
    pass


class UserMessageCreateDB(BaseModel):
    """Database model specifically for creating user messages"""
    session_id: str
    sender_id: str
    message: str
    
    @property
    def sender_type(self) -> str:
        return "user"


class AssistantMessageCreateDB(BaseModel):
    """Database model specifically for creating assistant messages"""
    session_id: str
    sender_id: str
    message: str
    
    @property
    def sender_type(self) -> str:
        return "assistant"


class ChatMessageReadDB(ChatMessageDB):
    """Database model for reading chat messages from database"""
    id: str  # Changed from int to str for custom IDs
    created_at: int  # Unix timestamp
    updated_at: Optional[int] = None
    deleted_at: Optional[int] = None
    
    @property
    def created_at_datetime(self) -> datetime:
        """Convert Unix timestamp to datetime"""
        return datetime.fromtimestamp(self.created_at)
    
    @property
    def updated_at_datetime(self) -> Optional[datetime]:
        """Convert Unix timestamp to datetime"""
        return datetime.fromtimestamp(self.updated_at) if self.updated_at else None
    
    class Config:
        from_attributes = True


class ChatMessageUpdateDB(BaseModel):
    """Database model for updating chat messages"""
    message: str
    updated_at: Optional[int] = None
