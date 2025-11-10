import json
import time
import asyncio
from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from fastapi.responses import StreamingResponse
from langfuse import observe, propagate_attributes

from app.models.schemas import ChatRequest, ChatRequestSession, UserMessageCreateDB, AssistantMessageCreateDB
from app.services.rag_service import rag_service
from app.services.db_service import db_service
from app.core.config import get_settings
from app.core.logging import logger
from app.services.memory_service import memory_service

settings = get_settings()
router = APIRouter()


async def summarize_conversation_background(session_id: str):
    """Background task to summarize conversation."""
    try:
        logger.info(f"🔄 Starting background summarization for session {session_id}")
        
        # Generate summary prompt
        summary_prompt = await memory_service.generate_summary_prompt(session_id)
        
        # Use LLM to generate summary
        response = await rag_service.llm.generate(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates concise conversation summaries."},
                {"role": "user", "content": summary_prompt}
            ],
            temperature=0.3,
            stream=False
        )
        
        summary = response.choices[0].message.content.strip()
        
        # Save summary to database
        total_count = await db_service.count_session_messages(session_id)
        await db_service.save_session_summary(session_id, summary, total_count)
        
        logger.info(f"✅ Background summarization completed for session {session_id}")
        logger.info(f"📝 Summary: {summary[:100]}...")
        
    except Exception as e:
        logger.error(f"❌ Error in background summarization: {e}", exc_info=True)



@router.post("/chat")
@observe()
async def chat_endpoint(
    request: ChatRequest,
):
    """
    Chat with RAG without session persistence - stateless conversation.
    Use this endpoint for one-off queries or when you don't need conversation history.
    """
    user_messages = [msg for msg in request.messages if msg.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")
    
    query = user_messages[-1].content
    logger.info(f"💬 Chat (stateless): '{query[:100]}...'")
    
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

@router.post("/chat/session")
@observe()
async def chat_endpoint_with_history(
    request: ChatRequestSession,
    background_tasks: BackgroundTasks
):
    """
    Chat with RAG with session persistence and intelligent memory management.
    Uses sliding window + progressive summarization for long conversations.
    """
    with propagate_attributes(user_id=request.user_id, session_id=request.session_id):
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")
        
        query = user_messages[-1].content
        logger.info(f"💬 Chat (session {request.session_id}): '{query[:100]}...'")
        
        # Get conversation context with memory management
        try:
            context = await memory_service.get_context_for_llm(request.session_id)
            logger.info(f"📊 Context: {context.total_messages} total messages, "
                    f"{'with' if context.summary else 'without'} summary")
        except Exception as e:
            logger.error(f"Error retrieving context: {e}", exc_info=True)
            context = None
        
        # Store user message in database
        try:
            user_msg = UserMessageCreateDB(
                session_id=request.session_id,
                sender_id=request.user_id,
                message=query
            )
            user_result = await db_service.insert_user_message(user_msg)
            logger.info(f"✅ Stored user message with ID: {user_result.id}")
        except Exception as e:
            logger.error(f"Error storing user message: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to store user message")
        
        # Check if we should trigger summarization (after response is sent)
        should_summarize = await memory_service.should_summarize(request.session_id)
        if should_summarize:
            logger.info(f"🔔 Scheduling background summarization for session {request.session_id}")
        
        # Process RAG pipeline
        reranked, _ = await rag_service.process_query(
            query=query,
            subject_filter=request.subject_filter,
            topic_filter=request.topic_filter,
            top_k=request.top_k
        )
        
        # Build RAG context
        rag_context = rag_service.build_context(reranked)
        rag_prompt = rag_service.create_rag_prompt(query, rag_context)
        
        # Build conversation messages with memory context
        conversation_messages = [rag_service.llm.SYSTEM_MESSAGE]
        
        # Add conversation history context if available
        if context and (context.summary or context.recent_messages):
            context_prompt = memory_service.build_context_prompt(context, query)
            conversation_messages.append({
                "role": "system",
                "content": f"Conversation Context:\n{context_prompt}"
            })
        
        # Add RAG-enhanced query
        conversation_messages.append({"role": "user", "content": rag_prompt})
        
        temperature = request.temperature or settings.TEMPERATURE
        
        if request.stream:
            async def generate_stream():
                try:
                    start = time.time()
                    full_response = ""
                    
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
                            content = chunk.choices[0].delta.content
                            full_response += content
                            yield json.dumps({'content': content}) + '\n'
                            await asyncio.sleep(0)
                    
                    duration = (time.time() - start) * 1000
                    logger.info(f"⏱️  Streaming took {duration:.2f}ms")
                    
                    # Store assistant message
                    try:
                        assistant_msg = AssistantMessageCreateDB(
                            session_id=request.session_id,
                            sender_id=request.assistant_id,
                            message=full_response
                        )
                        assistant_result = await db_service.insert_assistant_message(assistant_msg)
                        logger.info(f"✅ Stored assistant message with ID: {assistant_result.id}")
                    except Exception as e:
                        logger.error(f"Error storing assistant message: {e}", exc_info=True)
                    
                    # Trigger summarization in background if needed
                    if should_summarize:
                        background_tasks.add_task(
                            summarize_conversation_background,
                            request.session_id
                        )
                    
                    # Send metadata
                    yield json.dumps({
                        'sources': [
                            {"chunk_id": doc["chunk_id"], "subject": doc["subject"], "topic": doc["topic"]}
                            for doc in reranked
                        ]
                    }) + '\n'
                    
                    if usage_data:
                        yield json.dumps({
                            'usage': {
                                'prompt_tokens': usage_data.prompt_tokens,
                                'completion_tokens': usage_data.completion_tokens,
                                'total_tokens': usage_data.total_tokens
                            }
                        }) + '\n'
                    
                    yield json.dumps({
                        'context_info': {
                            'has_summary': context.summary is not None if context else False,
                            'total_messages': context.total_messages if context else 0,
                            'will_summarize': should_summarize
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
            
            # Store assistant message
            try:
                assistant_msg = AssistantMessageCreateDB(
                    session_id=request.session_id,
                    sender_id=request.assistant_id,
                    message=answer
                )
                assistant_result = await db_service.insert_assistant_message(assistant_msg)
                logger.info(f"✅ Stored assistant message with ID: {assistant_result.id}")
            except Exception as e:
                logger.error(f"Error storing assistant message: {e}", exc_info=True)
            
            # Trigger summarization in background if needed
            if should_summarize:
                background_tasks.add_task(
                    summarize_conversation_background,
                    request.session_id
                )
            
            return {
                "answer": answer,
                "session_id": request.session_id,
                "context_info": {
                    "has_summary": context.summary is not None if context else False,
                    "total_messages": context.total_messages if context else 0,
                    "will_summarize": should_summarize
                },
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


@router.get("/chat/session/{session_id}/summary")
@observe()
async def get_session_summary(session_id: str):
    """Get the current summary for a session."""
    try:
        summary = await db_service.get_session_summary(session_id)
        if not summary:
            return {
                "session_id": session_id,
                "has_summary": False,
                "message": "No summary available yet"
            }
        
        return {
            "session_id": session_id,
            "has_summary": True,
            "summary": summary.summary,
            "messages_count": summary.messages_count,
            "created_at": summary.created_at,
            "created_at_iso": summary.created_at_datetime.isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch summary")



@router.get("/chat/session/{session_id}/history")
@observe()
async def get_session_history(
    session_id: str,
    limit: int = 50
):
    """
    Retrieve chat history for a specific session.
    
    Args:
        session_id: The session identifier
        limit: Maximum number of messages to retrieve (default: 50)
    
    Returns:
        List of chat messages ordered by creation time
    """
    try:
        logger.info(f"📜 Fetching history for session {session_id}")
        history = await db_service.get_chat_history(session_id, limit)
        
        # Reverse to get chronological order (oldest first)
        history.reverse()
        
        return {
            "session_id": session_id,
            "message_count": len(history),
            "messages": [
                {
                    "id": msg.id,
                    "sender_type": msg.sender_type,
                    "message": msg.message,
                    "created_at": msg.created_at,
                    "created_at_iso": msg.created_at_datetime.isoformat()
                }
                for msg in history
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching session history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch session history")
