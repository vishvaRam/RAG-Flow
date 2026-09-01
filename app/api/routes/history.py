from fastapi import APIRouter

from app.services.db_service import db_service

router = APIRouter(prefix="/history", tags=["History"])


@router.get("/{session_id}")
async def get_session_history(session_id: str, limit: int = 50, offset: int = 0):
    """Retrieve UI chat history for a session from PostgreSQL."""
    history = await db_service.get_chat_history(session_id, limit=limit, offset=offset)
    history.reverse()  # Return chronological order
    return {
        "session_id": session_id,
        "count": len(history),
        "messages": [
            {
                "id": msg.id,
                "sender_type": msg.sender_type,
                "message": msg.message,
                "created_at": msg.created_at,
            }
            for msg in history
        ],
    }


@router.get("/{session_id}/summary")
async def get_session_summary(session_id: str):
    """Get session summary if available."""
    summary = await db_service.get_session_summary(session_id)
    if not summary:
        return {"session_id": session_id, "has_summary": False, "summary": None}
    return {
        "session_id": session_id,
        "has_summary": True,
        "summary": summary.summary,
        "messages_count": summary.messages_count,
    }