from typing import List, Optional, Tuple
from app.services.db_service import db_service
from app.models.schemas import ChatMessageReadDB, ConversationContext
from app.core.logging import logger


class ConversationMemoryService:
    """
    Service for managing conversation memory with progressive summarization.
    """
    
    def __init__(self, window_size: int = 10, summarize_threshold: int = 10):
        """
        Initialize memory service.
        
        Args:
            window_size: Number of recent messages to keep in full
            summarize_threshold: Trigger summarization after this many messages
        """
        self.window_size = window_size
        self.summarize_threshold = summarize_threshold
    
    async def get_context_for_llm(self, session_id: str) -> ConversationContext:
        """
        Get conversation context optimized for LLM consumption.
        
        Returns:
            ConversationContext with summary and recent messages
        """
        summary, recent_messages, total_count = await db_service.get_conversation_context(
            session_id, 
            self.window_size
        )
        
        # Format recent messages for LLM
        formatted_messages = [
            {
                "role": msg.sender_type,
                "content": msg.message,
                "timestamp": msg.created_at
            }
            for msg in recent_messages
        ]
        
        return ConversationContext(
            summary=summary,
            recent_messages=formatted_messages,
            total_messages=total_count
        )
    
    async def should_summarize(self, session_id: str) -> bool:
        """Check if conversation should be summarized."""
        total_count = await db_service.count_session_messages(session_id)
        summary_obj = await db_service.get_session_summary(session_id)
        
        if summary_obj is None:
            # First summarization after threshold
            return total_count >= self.summarize_threshold
        else:
            # Summarize every N new messages after last summary
            messages_since_summary = total_count - summary_obj.messages_count
            return messages_since_summary >= self.summarize_threshold
    
    async def generate_summary_prompt(self, session_id: str) -> str:
        """
        Generate prompt for LLM to create conversation summary.
        """
        summary_obj = await db_service.get_session_summary(session_id)
        existing_summary = summary_obj.summary if summary_obj else None
        last_summary_count = summary_obj.messages_count if summary_obj else 0
        
        # Get messages to summarize (messages after last summary)
        total_count = await db_service.count_session_messages(session_id)
        messages_to_summarize_count = total_count - last_summary_count
        
        # Get the messages that need summarization
        messages = await db_service.get_chat_history(
            session_id,
            limit=messages_to_summarize_count,
            offset=0
        )
        messages.reverse()  # Chronological order
        
        # Build conversation text
        conversation_text = "\n".join([
            f"{msg.sender_type.upper()}: {msg.message}"
            for msg in messages
        ])
        
        if existing_summary:
            prompt = f"""You are summarizing an ongoing conversation. Here is the previous summary:

                PREVIOUS SUMMARY:
                {existing_summary}

                NEW CONVERSATION SINCE LAST SUMMARY:
                {conversation_text}

                Please create an updated summary that:
                1. Integrates the previous summary with new information
                2. Maintains key topics, decisions, and context
                3. Removes redundant information
                4. Keeps the summary concise (max 300 words)
                5. Preserves important details

                Updated Summary:"""
        else:
            prompt = f"""You are summarizing a conversation. Here is the conversation:

                CONVERSATION:
                {conversation_text}

                Please create a summary that:
                1. Captures key topics and main points
                2. Preserves important context and decisions
                3. Keeps the summary concise (max 300 words)
                4. Includes important details

                Summary:"""
        
        return prompt
    
    def build_context_prompt(self, context: ConversationContext, current_query: str) -> str:
        """
        Build a context-aware prompt for the LLM.
        
        Args:
            context: Conversation context with summary and recent messages
            current_query: Current user query
            
        Returns:
            Formatted prompt with context
        """
        parts = []
        
        if context.summary:
            parts.append(f"CONVERSATION HISTORY SUMMARY:\n{context.summary}\n")
        
        if context.recent_messages:
            parts.append("RECENT CONVERSATION:")
            for msg in context.recent_messages[-self.window_size:]:
                parts.append(f"{msg['role'].upper()}: {msg['content']}")
            parts.append("")
        
        parts.append(f"CURRENT QUERY: {current_query}")
        
        return "\n".join(parts)


# Singleton instance
memory_service = ConversationMemoryService(
    window_size=10,
    summarize_threshold=10
)
