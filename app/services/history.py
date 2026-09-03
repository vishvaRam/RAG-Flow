import secrets
import string

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.core.database import db_manager
from app.core.logging import logger
from app.models.schemas import (
    ChatMessageCreateDB,
    ChatMessageReadDB,
    SessionSummaryDB,
    SessionSummaryOutput,
)

settings = get_settings()

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert academic conversation summarizer and context compression specialist.
Your task is to synthesize the conversation into a highly compact, dense, and structured text summary.

Compression Guidelines:
- Extract and preserve key academic topics, concepts, theorems, and formulas discussed.
- Note specific student queries, misconceptions, and resolutions.
- Eliminate conversational filler, pleasantries, and redundant remarks.
- Integrate with previous summary seamlessly without losing earlier context.""",
        ),
        (
            "human",
            """Previous Summary Context:
{existing_summary}

Recent Messages to Integrate:
{new_messages}

Generate the updated compressed summary.""",
        ),
    ]
)


def generate_id(length: int = 14) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


class HistoryService:
    """Manages chat persistence, context formulation, and periodic conversation compression."""

    def __init__(self):
        self.base_llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_PROVIDER_URL,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
            timeout=settings.LLM_TIMEOUT,
        )
        self.summary_chain = SUMMARY_PROMPT | self.base_llm.with_structured_output(
            SessionSummaryOutput
        )

    async def insert_message(self, message: ChatMessageCreateDB) -> ChatMessageReadDB:
        msg_id = generate_id()
        query = f"""
            INSERT INTO {settings.HISTORY_TABLE} 
            (id, session_id, sender_id, sender_type, message, created_at)
            VALUES ($1, $2, $3, $4, $5, EXTRACT(EPOCH FROM NOW())::bigint)
            RETURNING id, session_id, sender_id, sender_type, message, created_at, updated_at, deleted_at
        """
        async with db_manager.acquire_pg() as conn:
            row = await conn.fetchrow(
                query,
                msg_id,
                message.session_id,
                message.sender_id,
                message.sender_type,
                message.message,
            )
            return ChatMessageReadDB(**dict(row))

    async def get_history(
        self, session_id: str, limit: int = 50, offset: int = 0
    ) -> list[ChatMessageReadDB]:
        query = f"""
            SELECT id, session_id, sender_id, sender_type, message, created_at, updated_at, deleted_at
            FROM {settings.HISTORY_TABLE}
            WHERE session_id = $1 AND deleted_at IS NULL
            ORDER BY created_at DESC
            LIMIT $2 OFFSET $3
        """
        async with db_manager.acquire_pg() as conn:
            rows = await conn.fetch(query, session_id, limit, offset)
            return [ChatMessageReadDB(**dict(r)) for r in rows]

    async def get_summary(self, session_id: str) -> SessionSummaryDB | None:
        query = f"""
            SELECT id, session_id, summary, messages_count, created_at, updated_at
            FROM {settings.SUMMARY_TABLE}
            WHERE session_id = $1
            ORDER BY created_at DESC LIMIT 1
        """
        async with db_manager.acquire_pg() as conn:
            row = await conn.fetchrow(query, session_id)
            return SessionSummaryDB(**dict(row)) if row else None

    async def load_context(self, session_id: str) -> str:
        summary_rec = await self.get_summary(session_id)
        existing_summary = summary_rec.summary if summary_rec else ""

        history = await self.get_history(
            session_id, limit=settings.MAX_HISTORY_MESSAGES
        )
        if not history and not existing_summary:
            return ""

        dialogue = "\n".join(
            f"{'Student' if m.sender_type == 'user' else 'Tutor'}: {m.message}"
            for m in reversed(history)
        )

        blocks = []
        if existing_summary:
            blocks.append(f"Summary of Earlier Context:\n{existing_summary}")
        if dialogue:
            blocks.append(f"Recent Conversation:\n{dialogue}")
        return "\n\n".join(blocks)

    async def summarize_if_needed(self, session_id: str) -> None:
        async with db_manager.acquire_pg() as conn:
            count_row = await conn.fetchrow(
                f"SELECT COUNT(*) as cnt FROM {settings.HISTORY_TABLE} WHERE session_id = $1 AND deleted_at IS NULL",
                session_id,
            )
            total = count_row["cnt"] if count_row else 0

        latest = await self.get_summary(session_id)
        summarized_count = latest.messages_count if latest else 0
        unsummarized = total - summarized_count

        if unsummarized >= settings.SUMMARY_INTERVAL:
            logger.info(
                f"Triggering auto-summarization for session {session_id} ({unsummarized} pending messages)"
            )
            records = await self.get_history(session_id, limit=unsummarized)
            dialogue = "\n".join(
                f"{'Student' if m.sender_type == 'user' else 'Tutor'}: {m.message}"
                for m in reversed(records)
            )

            result: SessionSummaryOutput = await self.summary_chain.ainvoke(
                {
                    "existing_summary": latest.summary if latest else "None",
                    "new_messages": dialogue,
                }
            )
            summary_text = (
                result.summary.strip()
                if hasattr(result, "summary")
                else str(result).strip()
            )

            summary_id = generate_id()
            query = f"""
                INSERT INTO {settings.SUMMARY_TABLE} (id, session_id, summary, messages_count, created_at)
                VALUES ($1, $2, $3, $4, EXTRACT(EPOCH FROM NOW())::bigint)
            """
            async with db_manager.acquire_pg() as conn:
                await conn.execute(query, summary_id, session_id, summary_text, total)


history_service = HistoryService()
