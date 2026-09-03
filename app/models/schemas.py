from typing import Literal

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class AgentChatRequest(BaseModel):
    session_id: str
    user_id: str
    messages: list[ChatTurn]
    stream: bool = False
    exam: str | None = None
    assistant_id: str | None = "assistant"


class ChatMessageCreateDB(BaseModel):
    session_id: str
    sender_id: str
    sender_type: Literal["user", "assistant"]
    message: str


class ChatMessageReadDB(ChatMessageCreateDB):
    id: str
    created_at: int
    updated_at: int | None = None
    deleted_at: int | None = None


class SessionSummaryOutput(BaseModel):
    summary: str = Field(
        ...,
        description="A concise, compressed academic summary of the conversation including topics, key concepts, formulas, resolved doubts, and student progress.",
    )


class SessionSummaryDB(BaseModel):
    id: str
    session_id: str
    summary: str
    messages_count: int
    created_at: int
    updated_at: int | None = None


class DocumentUploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    message: str


class RetrievalFilterInput(BaseModel):
    query: str = Field(..., description="Target search query or core academic topic.")
    subject: str | None = Field(
        None, description="Academic subject (e.g., Physics, Chemistry, Mathematics)."
    )
