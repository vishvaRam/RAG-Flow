import asyncio
from typing import List, Dict, Optional
from langfuse.openai import AsyncOpenAI
from langfuse import observe

from app.core.config import get_settings
from app.core.logging import logger
from app.utils.decorators import log_time

settings = get_settings()


class LLMService:
    """Service for LLM operations"""
    
    SYSTEM_MESSAGE = {
        "role": "system",
        "content": """You are a JEE examination tutor with expertise in Physics, Chemistry, and Mathematics. 
                Your role is to:
                    - Explain concepts clearly with proper scientific reasoning
                    - Break down complex problems into manageable steps
                    - Use LaTeX notation for all mathematical expressions
                    - Connect theory to JEE exam patterns and real-world applications
                    - Maintain an encouraging, patient tone that builds student confidence
                    - Focus on conceptual understanding over rote memorization
                """
    }
    
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.GEMINI_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            timeout=settings.GEMINI_TIMEOUT,
            max_retries=settings.GEMINI_MAX_RETRIES
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
                    messages=[{"role": "user", "content": rewrite_prompt}],
                    max_tokens=settings.QUERY_REWRITE_MAX_TOKENS,
                    temperature=0.1
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
                model=settings.GEMINI_MODEL,
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
