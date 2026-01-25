import json
import time
import asyncio
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from langfuse import observe, propagate_attributes

from app.models.schemas import (
    ChatRequest,
    ChatRequestSession,
    UserMessageCreateDB,
    AssistantMessageCreateDB,
)
from app.services.rag_service import rag_service
from app.services.db_service import db_service
from app.services.llm_service import llm_service
from app.core.config import get_settings
from app.core.logging import logger
from app.services.memory_service import memory_service

settings = get_settings()
router = APIRouter()


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

    # Step 1: Query rewriting
    rewritten_query = await llm_service.rewrite_query(query, request.subject_filter)

    # Process RAG pipeline
    reranked, _ = await rag_service.process_query(
        rewritten_query=rewritten_query,
        subject_filter=request.subject_filter,
        topic_filter=request.topic_filter,
        top_k=request.top_k,
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
                    messages=conversation_messages, temperature=temperature, stream=True
                )

                usage_data = None
                async for chunk in stream:
                    if chunk.usage:
                        usage_data = chunk.usage
                    if chunk.choices[0].delta.content:
                        yield (
                            json.dumps({"content": chunk.choices[0].delta.content})
                            + "\n"
                        )
                        await asyncio.sleep(0)

                duration = (time.time() - start) * 1000
                logger.info(f"⏱️  Streaming took {duration:.2f}ms")

                # Send sources
                sources = [
                    {
                        "chunk_id": doc["chunk_id"],
                        "subject": doc["subject"],
                        "topic": doc["topic"],
                    }
                    for doc in reranked
                ]
                yield json.dumps({"sources": sources}) + "\n"

                # Send usage
                if usage_data:
                    yield (
                        json.dumps(
                            {
                                "usage": {
                                    "prompt_tokens": usage_data.prompt_tokens,
                                    "completion_tokens": usage_data.completion_tokens,
                                    "total_tokens": usage_data.total_tokens,
                                }
                            }
                        )
                        + "\n"
                    )

                yield json.dumps({"done": True}) + "\n"
            except Exception as e:
                logger.error(f"Streaming error: {e}", exc_info=True)
                yield json.dumps({"error": str(e)}) + "\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    else:
        start = time.time()

        response = await rag_service.llm.generate(
            messages=conversation_messages, temperature=temperature, stream=False
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
                    "relevance_score": doc.get("rerank_score", doc["score"]),
                }
                for doc in reranked
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }


@router.post("/chat/session")
@observe()
async def chat_endpoint_with_history(
    request: ChatRequestSession, background_tasks: BackgroundTasks
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
            logger.info(
                f"📊 Context: {context.total_messages} total messages, "
                f"{'with' if context.summary else 'without'} summary"
            )
        except Exception as e:
            logger.error(f"Error retrieving context: {e}", exc_info=True)
            context = None

        # Store user message in database
        try:
            user_msg = UserMessageCreateDB(
                session_id=request.session_id, sender_id=request.user_id, message=query
            )
            background_tasks.add_task(db_service.insert_user_message, user_msg)
        except Exception as e:
            logger.error(f"Error storing user message: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to store user message")

        # Check if we should trigger summarization (after response is sent)
        should_summarize = await memory_service.should_summarize(request.session_id)
        if should_summarize:
            logger.info(
                f"🔔 Scheduling background summarization for session {request.session_id}"
            )

        # Query rewriting
        rewritten_query = await llm_service.rewrite_query_with_history(
            query,
            context,
            num_previous_messages=settings.QUERY_REWRITE_WITH_HISTORY_COUNT,
            subject_filter=request.subject_filter,
        )

        # Process RAG pipeline
        reranked, _ = await rag_service.process_query(
            rewritten_query=rewritten_query,
            subject_filter=request.subject_filter,
            topic_filter=request.topic_filter,
            top_k=request.top_k,
        )

        # Build RAG context
        rag_context = rag_service.build_context(reranked)

        # ===== FIX: Build system message properly =====
        system_content_parts = [rag_service.llm.SYSTEM_MESSAGE["content"]]

        # Optionally add conversation summary to system (keep it brief!)
        if context and context.summary:
            system_content_parts.append(
                f"\n\nPrevious Conversation Summary:\n{context.summary}"
            )

        unified_system_message = {
            "role": "system",
            "content": "\n".join(system_content_parts),
        }

        conversation_messages = [unified_system_message]

        # ===== FIX: Add recent conversation history (all of them) =====
        if context and context.recent_messages:
            for msg in context.recent_messages:  # Include ALL recent messages
                conversation_messages.append(
                    {
                        "role": "user" if msg["role"] == "user" else "assistant",
                        "content": msg["content"],
                    }
                )

        # ===== FIX: Proper formatting for current user message =====
        current_user_message = rag_service.create_rag_prompt(query, rag_context)

        conversation_messages.append({"role": "user", "content": current_user_message})

        temperature = request.temperature or settings.TEMPERATURE

        if request.stream:

            async def generate_stream():
                try:
                    start = time.time()
                    full_response = ""

                    stream = await rag_service.llm.generate(
                        messages=conversation_messages,
                        temperature=temperature,
                        stream=True,
                    )

                    usage_data = None
                    async for chunk in stream:
                        if chunk.usage:
                            usage_data = chunk.usage
                        if chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            full_response += content
                            yield json.dumps({"content": content}) + "\n"
                            await asyncio.sleep(0)

                    duration = (time.time() - start) * 1000
                    logger.info(f"⏱️  Streaming took {duration:.2f}ms")

                    # Store assistant message
                    try:
                        assistant_msg = AssistantMessageCreateDB(
                            session_id=request.session_id,
                            sender_id=request.assistant_id,
                            message=full_response,
                        )
                        background_tasks.add_task(
                            db_service.insert_assistant_message, assistant_msg
                        )
                    except Exception as e:
                        logger.error(
                            f"Error storing assistant message: {e}", exc_info=True
                        )

                    # Trigger summarization in background if needed
                    if should_summarize:
                        background_tasks.add_task(
                            llm_service.generate_summary, request.session_id
                        )

                    # Send metadata
                    yield (
                        json.dumps(
                            {
                                "sources": [
                                    {
                                        "chunk_id": doc["chunk_id"],
                                        "subject": doc["subject"],
                                        "topic": doc["topic"],
                                    }
                                    for doc in reranked
                                ]
                            }
                        )
                        + "\n"
                    )

                    if usage_data:
                        yield (
                            json.dumps(
                                {
                                    "usage": {
                                        "prompt_tokens": usage_data.prompt_tokens,
                                        "completion_tokens": usage_data.completion_tokens,
                                        "total_tokens": usage_data.total_tokens,
                                    }
                                }
                            )
                            + "\n"
                        )

                    yield (
                        json.dumps(
                            {
                                "context_info": {
                                    "has_summary": context.summary is not None
                                    if context
                                    else False,
                                    "total_messages": context.total_messages
                                    if context
                                    else 0,
                                    "will_summarize": should_summarize,
                                }
                            }
                        )
                        + "\n"
                    )

                    yield json.dumps({"done": True}) + "\n"
                except Exception as e:
                    logger.error(f"Streaming error: {e}", exc_info=True)
                    yield json.dumps({"error": str(e)}) + "\n"

            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                },
            )

        else:
            start = time.time()

            response = await rag_service.llm.generate(
                messages=conversation_messages, temperature=temperature, stream=False
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
                    message=answer,
                )
                background_tasks.add_task(
                    db_service.insert_assistant_message, assistant_msg
                )
            except Exception as e:
                logger.error(f"Error storing assistant message: {e}", exc_info=True)

            # Trigger summarization in background if needed
            if should_summarize:
                background_tasks.add_task(
                    llm_service.generate_summary, request.session_id
                )

            return {
                "answer": answer,
                "session_id": request.session_id,
                "context_info": {
                    "has_summary": context.summary is not None if context else False,
                    "total_messages": context.total_messages if context else 0,
                    "will_summarize": should_summarize,
                },
                "sources": [
                    {
                        "chunk_id": doc["chunk_id"],
                        "subject": doc["subject"],
                        "topic": doc["topic"],
                        "relevance_score": doc.get("rerank_score", doc["score"]),
                    }
                    for doc in reranked
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
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
                "message": "No summary available yet",
            }

        return {
            "session_id": session_id,
            "has_summary": True,
            "summary": summary.summary,
            "messages_count": summary.messages_count,
            "created_at": summary.created_at,
            "created_at_iso": summary.created_at_datetime.isoformat(),
        }
    except Exception as e:
        logger.error(f"Error fetching summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch summary")


@router.get("/chat/session/{session_id}/history")
@observe()
async def get_session_history(session_id: str, limit: int = 50):
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
                    "created_at_iso": msg.created_at_datetime.isoformat(),
                }
                for msg in history
            ],
        }
    except Exception as e:
        logger.error(f"Error fetching session history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch session history")
