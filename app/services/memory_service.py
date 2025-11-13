from app.services.db_service import db_service
from app.models.schemas import ConversationContext
from app.core.logging import logger
from app.core.config import get_settings
import time
from typing import Optional


settings = get_settings()


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
        logger.info(f"🧠 ConversationMemoryService initialized:")
        logger.info(f"   • Window size: {window_size} messages")
        logger.info(f"   • Summarization threshold: {summarize_threshold} messages")
    
    async def get_context_for_llm(self, session_id: str) -> ConversationContext:
        """
        Get conversation context optimized for LLM consumption.
        
        Returns:
            ConversationContext with summary and recent messages
        """
        start_time = time.time()
        logger.info(f"📚 Fetching conversation context for session: {session_id}")
        
        try:
            summary, recent_messages, total_count = await db_service.get_conversation_context(
                session_id, 
                self.window_size
            )
            
            # Format recent messages for LLM
            formatted_messages = [
                {
                    "role": msg.sender_type,
                    "content": msg.message
                }
                for msg in recent_messages
            ]
            
            duration = (time.time() - start_time) * 1000
            
            logger.info(f"✅ Context retrieved for session {session_id}:")
            logger.info(f"   • Total messages: {total_count}")
            logger.info(f"   • Recent messages: {len(formatted_messages)}")
            logger.info(f"   • Has summary: {'Yes' if summary else 'No'}")
            if summary:
                logger.info(f"   • Summary length: {len(summary)} chars")
            logger.info(f"   ⏱️  Retrieved in {duration:.2f}ms")
            
            return ConversationContext(
                summary=summary,
                recent_messages=formatted_messages,
                total_messages=total_count
            )
        
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            logger.error(f"❌ Failed to get context for session {session_id}: {e}", exc_info=True)
            logger.error(f"   ⏱️  Failed after {duration:.2f}ms")
            
            # Return empty context on error
            return ConversationContext(
                summary=None,
                recent_messages=[],
                total_messages=0
            )
    
    async def should_summarize(self, session_id: str) -> bool:
        """Check if conversation should be summarized."""
        start_time = time.time()
        logger.debug(f"🔍 Checking if summarization needed for session: {session_id}")
        
        try:
            total_count = await db_service.count_session_messages(session_id)
            summary_obj = await db_service.get_session_summary(session_id)
            
            if summary_obj is None:
                # First summarization after threshold
                should_summarize = total_count >= self.summarize_threshold
                duration = (time.time() - start_time) * 1000
                
                if should_summarize:
                    logger.info(f"🔔 Summarization triggered for session {session_id}:")
                else:
                    logger.debug(f"⏸️  Summarization not needed: {total_count}/{self.summarize_threshold} messages")
                
                return should_summarize
            else:
                # Summarize every N new messages after last summary
                messages_since_summary = total_count - summary_obj.messages_count
                should_summarize = messages_since_summary >= self.summarize_threshold
                duration = (time.time() - start_time) * 1000
                
                if should_summarize:
                    logger.info(f"🔔 Summarization triggered for session {session_id}:")
                else:
                    logger.debug(f"⏸️  Summarization not needed: {messages_since_summary}/{self.summarize_threshold} new messages")
                
                return should_summarize
        
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            logger.error(f"❌ Error checking summarization for session {session_id}: {e}", exc_info=True)
            logger.error(f"   ⏱️  Failed after {duration:.2f}ms")
            return False
    
    async def generate_summary_prompt(self, session_id: str) -> str:
        """
        Generate prompt for LLM to create conversation summary.
        """
        start_time = time.time()
        logger.info(f"📝 Generating summary prompt for session: {session_id}")
        
        try:
            summary_obj = await db_service.get_session_summary(session_id)
            existing_summary = summary_obj.summary if summary_obj else None
            last_summary_count = summary_obj.messages_count if summary_obj else 0
            
            # Get messages to summarize (messages after last summary)
            total_count = await db_service.count_session_messages(session_id)
            messages_to_summarize_count = total_count - last_summary_count
            
            logger.info(f"📊 Summary prompt details:")
            logger.info(f"   • Total messages: {total_count}")
            logger.info(f"   • Last summary at: {last_summary_count} messages")
            logger.info(f"   • Messages to summarize: {messages_to_summarize_count}")
            logger.info(f"   • Has existing summary: {'Yes' if existing_summary else 'No'}")
            
            # Get the messages that need summarization
            messages = await db_service.get_chat_history(
                session_id,
                limit=messages_to_summarize_count,
                offset=0
            )
            messages.reverse()  # Chronological order
            
            logger.info(f"   • Retrieved {len(messages)} messages for summarization")
            
            # Build conversation text
            conversation_text = "\n".join([
                f"{msg.sender_type.upper()}: {msg.message}"
                for msg in messages
            ])
            
            conversation_length = len(conversation_text)
            logger.info(f"   • Conversation text length: {conversation_length} chars")
            
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
                logger.info(f"   • Prompt type: Update existing summary")
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
                logger.info(f"   • Prompt type: Create first summary")
            
            duration = (time.time() - start_time) * 1000
            prompt_length = len(prompt)
            logger.info(f"✅ Summary prompt generated:")
            logger.info(f"   • Prompt length: {prompt_length} chars")
            logger.info(f"   ⏱️  Generated in {duration:.2f}ms")
            
            return prompt
        
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            logger.error(f"❌ Error generating summary prompt for session {session_id}: {e}", exc_info=True)
            logger.error(f"   ⏱️  Failed after {duration:.2f}ms")
            raise
    
    def build_context_prompt(
        self, 
        context: ConversationContext, 
        current_query: str
    ) -> str:
        """
        Build a context-aware prompt for the LLM.
        
        Args:
            context: Conversation context with summary and recent messages
            current_query: Current user query
            
        Returns:
            Formatted prompt with context
        """
        start_time = time.time()
        logger.debug(f"🔨 Building context prompt for query: '{current_query[:50]}...'")
        
        try:
            parts = []
            
            if context.summary:
                parts.append(f"CONVERSATION HISTORY SUMMARY:\n{context.summary}\n")
                logger.debug(f"   • Added summary: {len(context.summary)} chars")
            
            if context.recent_messages:
                messages_to_include = context.recent_messages[-self.window_size:]
                parts.append("RECENT CONVERSATION:")
                for msg in messages_to_include:
                    parts.append(f"{msg['role'].upper()}: {msg['content']}")
                parts.append("")
                logger.debug(f"   • Added {len(messages_to_include)} recent messages")
            
            parts.append(f"CURRENT QUERY: {current_query}")
            
            prompt = "\n".join(parts)
            duration = (time.time() - start_time) * 1000
            
            logger.info(f"✅ Context prompt built:")
            logger.info(f"   • Total length: {len(prompt)} chars")
            logger.info(f"   • Components: {'Summary' if context.summary else ''}"
                       f"{' + ' if context.summary and context.recent_messages else ''}"
                       f"{'Recent messages' if context.recent_messages else ''}")
            logger.info(f"   ⏱️  Built in {duration:.2f}ms")
            
            return prompt
        
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            logger.error(f"❌ Error building context prompt: {e}", exc_info=True)
            logger.error(f"   ⏱️  Failed after {duration:.2f}ms")
            
            # Return minimal prompt on error
            return f"CURRENT QUERY: {current_query}"
    
    def get_statistics(self) -> dict:
        """
        Get service statistics and configuration.
        
        Returns:
            Dictionary with service stats
        """
        stats = {
            "window_size": self.window_size,
            "summarize_threshold": self.summarize_threshold,
            "service_status": "active"
        }
        logger.info(f"📊 Memory service statistics: {stats}")
        return stats


# Singleton instance
memory_service = ConversationMemoryService(
    window_size=settings.MAX_HISTORY_MESSAGES,
    summarize_threshold=settings.SUMMARY_INTERVAL
)
