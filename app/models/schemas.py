from datetime import datetime
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# ==================== API Request & Response Models ====================


class ChatMessagePayload(BaseModel):
    """Individual message object within the conversation history array."""

    role: Literal["user", "assistant", "system"] = Field(
        ..., description="Message author role"
    )
    content: str = Field(..., description="Text content of the message")


class AgentChatRequest(BaseModel):
    """Payload for communicating with the LangGraph ReAct agent."""

    session_id: str = Field(..., description="Unique thread/session ID")
    user_id: str = Field(..., description="Unique user identifier")
    assistant_id: str = Field(..., description="Unique assistant identifier")
    messages: List[ChatMessagePayload] = Field(
        ..., min_length=1, description="List of messages in the conversation"
    )
    exam: Optional[str] = Field(
        default=None, description="Target exam context (e.g. 'JEE_MAIN', 'NEET', 'UPSC')"
    )
    stream: bool = Field(default=False, description="Enable SSE token streaming")


class DocumentUploadResponse(BaseModel):
    """Response payload after PDF parsing and vector indexing."""

    filename: str
    chunks_indexed: int
    message: str


# ==================== PostgreSQL Frontend History Models ====================


class ChatMessageCreateDB(BaseModel):
    """Payload for inserting a message into the PostgreSQL history table."""

    session_id: str
    sender_id: str
    sender_type: Literal["user", "assistant"]
    message: str


class ChatMessageReadDB(BaseModel):
    """Payload for returning recorded chat history to the frontend UI."""

    id: str
    session_id: str
    sender_id: str
    sender_type: str
    message: str
    created_at: int
    updated_at: Optional[int] = None
    deleted_at: Optional[int] = None

    @property
    def created_at_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.created_at)

    @property
    def updated_at_datetime(self) -> Optional[datetime]:
        return datetime.fromtimestamp(self.updated_at) if self.updated_at else None

    class Config:
        from_attributes = True


class SessionSummaryDB(BaseModel):
    """Payload for session summary storage and retrieval."""

    id: str
    session_id: str
    summary: str
    messages_count: int
    created_at: int
    updated_at: Optional[int] = None

    @property
    def created_at_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.created_at)

    class Config:
        from_attributes = True