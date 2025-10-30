import json
import time
import asyncio
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from langfuse import observe

from app.models.schemas import ChatRequest
from app.services.rag_service import rag_service
from app.core.config import get_settings
from app.core.logging import logger

settings = get_settings()
router = APIRouter()


@router.post("/chat")
@observe()
async def chat_endpoint(
    request: ChatRequest,
):
    """Chat with RAG - returns JSON or streaming response"""
    user_messages = [msg for msg in request.messages if msg.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")
    
    query = user_messages[-1].content
    logger.info(f"💬 Chat: '{query[:100]}...'")
    
    # Process RAG pipeline
    reranked, _ = await rag_service.process_query(
        query=query,
        subject_filter=request.subject_filter,
        topic_filter=request.topic_filter,
        top_k=request.top_k
    )
    
    # Build prompt
    context = rag_service.build_context(reranked)
    rag_prompt = rag_service.create_rag_prompt(query, context)
    
    # Build messages
    conversation_messages = [rag_service.llm.SYSTEM_MESSAGE]
    for msg in request.messages[:-1]:
        conversation_messages.append({"role": msg.role, "content": msg.content})
    conversation_messages.append({"role": "user", "content": rag_prompt})
    
    temperature = request.temperature or settings.TEMPERATURE
    
    if request.stream:
        async def generate_stream():
            try:
                start = time.time()
                
                stream = await rag_service.llm.generate(
                    messages=conversation_messages,
                    temperature=temperature,
                    stream=True
                )
                
                usage_data = None
                async for chunk in stream:
                    if chunk.usage:
                        usage_data = chunk.usage
                    if chunk.choices[0].delta.content:
                        yield json.dumps({'content': chunk.choices[0].delta.content}) + '\n'
                        await asyncio.sleep(0)
                
                duration = (time.time() - start) * 1000
                logger.info(f"⏱️  Streaming took {duration:.2f}ms")
                
                # Send sources
                sources = [
                    {"chunk_id": doc["chunk_id"], "subject": doc["subject"], "topic": doc["topic"]}
                    for doc in reranked
                ]
                yield json.dumps({'sources': sources}) + '\n'
                
                # Send usage
                if usage_data:
                    yield json.dumps({
                        'usage': {
                            'prompt_tokens': usage_data.prompt_tokens,
                            'completion_tokens': usage_data.completion_tokens,
                            'total_tokens': usage_data.total_tokens
                        }
                    }) + '\n'
                
                yield json.dumps({'done': True}) + '\n'
            except Exception as e:
                logger.error(f"Streaming error: {e}", exc_info=True)
                yield json.dumps({'error': str(e)}) + '\n'
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive"
            }
        )
    
    else:
        start = time.time()
        
        response = await rag_service.llm.generate(
            messages=conversation_messages,
            temperature=temperature,
            stream=False
        )
        
        duration = (time.time() - start) * 1000
        logger.info(f"⏱️  LLM took {duration:.2f}ms")
        
        answer = response.choices[0].message.content
        logger.info(f"✅ Generated {len(answer)} characters")
        
        return {
            "answer": answer,
            "sources": [
                {
                    "chunk_id": doc["chunk_id"],
                    "subject": doc["subject"],
                    "topic": doc["topic"],
                    "relevance_score": doc.get("rerank_score", doc["score"])
                }
                for doc in reranked
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
