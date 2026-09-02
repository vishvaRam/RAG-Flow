from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import get_settings
from app.core.logging import logger
from app.services.db_service import db_service

settings = get_settings()

SUMMARY_PROMPT = PromptTemplate.from_template(
    """You are an expert conversation summarizer for an educational assistant.
Given the existing summary (if any) and a set of new messages between the Student and the Tutor, create a concise, factual updated summary.
Preserve key topics studied, unresolved questions, doubts, and core academic concepts discussed.

Existing Summary:
{existing_summary}

New Conversation Messages:
{new_messages}

Updated Comprehensive Summary:"""
)


class MemoryService:
    """Manages chat context reconstruction and asynchronous progressive summarization."""

    def __init__(self):
        self.summary_llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            google_api_key=settings.LLM_API_KEY,
            temperature=0.0,
            max_output_tokens=4000,
            timeout=30,
        )

    async def load_session_context(self, session_id: str) -> str:
        """Reconstructs session context combining the latest summary and recent raw dialogue turns."""
        summary_record = await db_service.get_session_summary(session_id)
        existing_summary = summary_record.summary if summary_record else ""

        db_records = await db_service.get_chat_history(
            session_id=session_id,
            limit=settings.MAX_HISTORY_MESSAGES,
            offset=0,
        )

        if not db_records and not existing_summary:
            return ""

        chronological_records = list(reversed(db_records))

        formatted_lines = []
        for record in chronological_records:
            role = (
                "Student"
                if record.sender_type.lower() in ("user", "human")
                else "Tutor"
            )
            formatted_lines.append(f"{role}: {record.message}")

        recent_dialogue = "\n".join(formatted_lines)

        context_blocks = []
        if existing_summary:
            context_blocks.append(f"Summary of Earlier Context:\n{existing_summary}")
        if recent_dialogue:
            context_blocks.append(f"Recent Conversation:\n{recent_dialogue}")

        return "\n\n".join(context_blocks)

    async def summarize_session_if_needed(self, session_id: str, threshold: int = 10):
        """Checks total messages and generates an updated summary if unsummarized messages meet threshold."""
        total_count = await db_service.count_session_messages(session_id)
        latest_summary_record = await db_service.get_session_summary(session_id)

        already_summarized_count = (
            latest_summary_record.messages_count if latest_summary_record else 0
        )
        unsummarized_count = total_count - already_summarized_count

        if unsummarized_count >= threshold:
            logger.info(
                f"🔄 Triggering history summarization for session {session_id} "
                f"({unsummarized_count} unsummarized messages, total {total_count})"
            )

            new_records = await db_service.get_chat_history(
                session_id=session_id,
                limit=unsummarized_count,
                offset=0,
            )
            chronological_new = list(reversed(new_records))

            new_lines = [
                f"{'Student' if r.sender_type.lower() in ('user', 'human') else 'Tutor'}: {r.message}"
                for r in chronological_new
            ]
            new_messages_text = "\n".join(new_lines)

            existing_summary = (
                latest_summary_record.summary if latest_summary_record else "None"
            )

            summary_chain = SUMMARY_PROMPT | self.summary_llm
            summary_response = await summary_chain.ainvoke(
                {
                    "existing_summary": existing_summary,
                    "new_messages": new_messages_text,
                }
            )

            updated_summary_text = (
                summary_response.content
                if isinstance(summary_response.content, str)
                else str(summary_response.content)
            )

            await db_service.save_session_summary(
                session_id=session_id,
                summary=updated_summary_text.strip(),
                messages_count=total_count,
            )
            logger.info(
                f"✓ Summary persisted for session {session_id} up to message #{total_count}"
            )


memory_service = MemoryService()
