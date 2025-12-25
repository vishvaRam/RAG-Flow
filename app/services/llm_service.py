import asyncio
from typing import List, Dict, Optional
from langfuse.openai import AsyncOpenAI # type: ignore
from langfuse import observe

from app.core.config import get_settings
from app.core.logging import logger
from app.models.schemas import ConversationContext
from app.services.db_service import db_service
from app.utils.decorators import log_time
from app.services.memory_service import memory_service

settings = get_settings()


class LLMService:
    """Service for LLM operations"""
    
    SYSTEM_MESSAGE = {
        "role": "system",
        "content": """
            You are a JEE examination tutor with expertise in Physics, Chemistry, and Mathematics. 
                STRICT RULES:
                    - Do NOT mention documents, passages, sources, references, page numbers, or phrases like
                    "according to the context", "from the document", or "the text says".
                    - Do NOT cite or refer to how or where the information was obtained.
                    - Write the answer as if it is your own explanation.
                Your role is to:
                    - Explain concepts clearly with proper scientific reasoning
                    - Break down complex problems into manageable steps
                    - Use LaTeX notation for all mathematical expressions
                    - Connect theory to JEE exam patterns and real-world applications
                    - Maintain an encouraging, patient tone that builds student confidence
                    - Focus on conceptual understanding over rote memorization\n\n
                """
    }
    
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_PROVIDER_URL,
            timeout=settings.LLM_TIMEOUT,
            max_retries=settings.LLM_MAX_RETRIES
        )
    
    @observe()
    @log_time("Query rewriting")
    async def rewrite_query(
        self,
        query: str,
        subject_filter: Optional[str] = None
    ) -> str:
        """Rewrite query for better retrieval"""
        if not settings.ENABLE_QUERY_REWRITING or len(query.split()) < 3:
            logger.info("📝 Skipping query rewriting")
            return query
        
        rewrite_prompt = f"""Add technical terms to: "{query}"
            Subject: {subject_filter or "Physics/Chemistry/Mathematics"}
            Output ONLY the improved Hybrid RAG query."""
        
        try:
            async with asyncio.timeout(settings.QUERY_REWRITE_TIMEOUT):
                response = await self.client.chat.completions.create(
                    model=settings.QUERY_REWRITE_MODEL,
                    reasoning_effort="none",
                    messages=[{"role": "user", "content": rewrite_prompt}],
                    max_tokens=settings.QUERY_REWRITE_MAX_TOKENS,
                    temperature=0.01,
                )
            
            if response.choices and response.choices[0].message.content:
                rewritten = response.choices[0].message.content.strip().strip('"').strip("'")
                logger.info(f"📝 Rewritten: '{query[:40]}' → '{rewritten[:40]}'")
                return rewritten
            return query
        
        except asyncio.TimeoutError:
            logger.warning(f"Query rewriting timeout after {settings.QUERY_REWRITE_TIMEOUT}s")
            return query
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            return query
        
    @observe()
    @log_time("Query rewriting with history")
    async def rewrite_query_with_history(
        self,
        query: str,
        context: Optional[ConversationContext] = None,
        num_previous_messages: int = settings.MAX_HISTORY_MESSAGES,
        subject_filter: Optional[str] = None
    ) -> str:
        """
        Rewrite query for better retrieval using conversation context.
        
        Args:
            query: User's current query
            context: ConversationContext with summary and recent messages
            num_previous_messages: Number of recent messages to include (default: 5)
            subject_filter: Subject area to focus on
            
        Returns:
            Rewritten query optimized for hybrid RAG retrieval
        """
        if not settings.ENABLE_QUERY_REWRITING or len(query.split()) < 3:
            logger.info("📝 Skipping query rewriting (disabled or query too short)")
            return query
        
        # Build context string from ConversationContext
        context_str = ""
        if context and context.recent_messages:
            try:
                # Get the specified number of recent messages
                messages_to_use = context.recent_messages[-num_previous_messages:]
                
                context_parts = [
                    f"{msg['role']}: {msg['content'][:100]}" 
                    for msg in messages_to_use
                ]
                context_str = " | ".join(context_parts)
                
                logger.info(f"📚 Using {len(messages_to_use)} messages for context")
                
            except Exception as e:
                logger.warning(f"⚠️  Failed to build context string: {e}")
                context_str = ""
        
        # Add summary if available
        summary_str = ""
        if context and context.summary:
            summary_str = f"\nConversation summary: {context.summary[:150]}"
        
        # Build rewrite prompt
        rewrite_prompt = f"""Rewrite this query into a better search query for semantic search and reranking:

            Query: "{query}"
            Recent conversation: {context_str or "None"}{summary_str}
            Subject: {subject_filter or "Physics/Chemistry/Mathematics"}

            Instructions:
            - Replace pronouns (it, this, that, these) with specific nouns from conversation
            - Add relevant technical terms and keywords
            - Make it standalone and self-contained
            - Keep it concise
            - Focus on concepts for document retrieval

            Output only the rewritten query:"""
        
        try:
            async with asyncio.timeout(settings.QUERY_REWRITE_TIMEOUT):
                response = await self.client.chat.completions.create(
                    model=settings.QUERY_REWRITE_MODEL,
                    reasoning_effort="none",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at reformulating conversational queries into standalone search queries. Output ONLY the rewritten query."
                        },
                        {
                            "role": "user",
                            "content": rewrite_prompt
                        }
                    ],
                    max_tokens=settings.QUERY_REWRITE_MAX_TOKENS,
                    temperature=0.1
                )
            
            if response.choices and response.choices[0].message.content:
                rewritten = response.choices[0].message.content.strip().strip('"').strip("'")
                logger.info("📝 Query rewriting:")
                logger.info(f"   Original:  '{query}'")
                logger.info(f"   Rewritten: '{rewritten}'")
                return rewritten
            
            logger.warning("⚠️  Empty response from query rewriter, using original")
            return query
        
        except asyncio.TimeoutError:
            logger.warning(f"⏱️  Query rewriting timeout after {settings.QUERY_REWRITE_TIMEOUT}s")
            return query
        except Exception as e:
            logger.error(f"❌ Query rewriting failed: {e}", exc_info=True)
            return query

    
    @observe()
    @log_time("LLM generation")
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        stream: bool = False
    ):
        """Generate LLM response"""
        try:
            response = await self.client.chat.completions.create(
                model=settings.LLM_MODEL,
                reasoning_effort="none",
                messages=messages, # type: ignore
                max_tokens=settings.MAX_TOKENS,
                temperature=temperature or settings.TEMPERATURE,
                stream=stream,
                stream_options={"include_usage": True} if stream else {}
            ) # type: ignore
            return response
        except Exception as e:
            logger.error(f"LLM generation failed: {e}", exc_info=True)
            raise
    
    @observe()
    @log_time("Chat summary generation")
    async def generate_summary(
        self,
        session_id: str,
    ):
        """Generate chat summary using LLM"""
        
        try:
            # Generate summary prompt
            summary_prompt = await memory_service.generate_summary_prompt(session_id)
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates concise conversation summaries."},
                {"role": "user", "content": summary_prompt}
            ]
            logger.info(f"🔄 Starting background summarization for session {session_id}")
      
            response = await self.client.chat.completions.create(
                model=settings.SUMMARY_MODEL,
                reasoning_effort="none",
                messages=messages, # type: ignore
                max_tokens=settings.SUMMARY_MAX_TOKENS,
                temperature=0.3,
                stream=False
            ) # type: ignore
            
            summary = response.choices[0].message.content.strip()
        
            # Save summary to database
            total_count = await db_service.count_session_messages(session_id)
            await db_service.save_session_summary(session_id, summary, total_count)
            
            logger.info(f"✅ Background summarization completed for session {session_id}")
            logger.info(f"📝 Summary: {summary[:100]}...")
        
        except Exception as e:
            logger.error(f"Chat summary generation failed: {e}", exc_info=True)
        
llm_service = LLMService()