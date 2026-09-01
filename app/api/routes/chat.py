import asyncio
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import get_settings
from app.models.schemas import AgentChatRequest, ChatMessageCreateDB
from app.services.agent_service import agent_service
from app.services.db_service import db_service
from app.services.memory_service import memory_service

settings = get_settings()
router = APIRouter(prefix="/agent", tags=["Agent"])


def extract_text_content(content: Any) -> str:
    """
    Safely extract plain text from LangChain message content.
    Handles plain string or list of content blocks.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            elif isinstance(part, str):
                text_parts.append(part)
            elif hasattr(part, "text"):
                text_parts.append(getattr(part, "text"))
            else:
                text_parts.append(str(part))
        return "".join(text_parts).strip()
    return str(content).strip()


@router.post("/chat")
async def chat_with_agent(
    request: AgentChatRequest,
    background_tasks: BackgroundTasks,
):
    """
    Interact with the LangGraph ReAct agent.
    - Reconstructs history and persistent summaries via memory_service.
    - Supports SSE streaming and standard request/response.
    """
    # 1. Extract the latest user query
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(
            status_code=400,
            detail="No user message found in the messages array.",
        )
    latest_user_message = user_messages[-1].content

    # 2. Persist incoming user query to PostgreSQL in background
    background_tasks.add_task(
        db_service.insert_message,
        ChatMessageCreateDB(
            session_id=request.session_id,
            sender_id=request.user_id,
            sender_type="user",
            message=latest_user_message,
        ),
    )

    # 3. Load previous history + existing summary context from memory_service
    past_context = await memory_service.load_session_context(request.session_id)

    # 4. Build dynamic system prompt
    system_prompt = settings.JEE_SYSTEM_PROMPT
    if request.exam:
        system_prompt += (
            f"\n\nContext Directive: The student is preparing for the '{request.exam.upper()}' exam. "
            f"When invoking 'retrieve_study_material', prioritize exam='{request.exam.upper()}'."
        )

    # Inject history context
    system_prompt += (
        f"\n\n--- PREVIOUS CONVERSATION CONTEXT ---\n"
        f"{past_context}\n"
        f"------------------------------------"
    )

    # 5. Assemble input messages
    input_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=latest_user_message),
    ]

    config = {"configurable": {"thread_id": request.session_id}}

    # ==================== STREAMING MODE ====================
    if request.stream:

        async def token_generator():
            accumulated_response = ""
            try:
                async for event in agent_service.agent.astream_events(
                    {"messages": input_messages},
                    config=config,
                    version="v2",
                ):
                    if (
                        event["event"] == "on_chat_model_stream"
                        and event.get("metadata", {}).get("langgraph_node") == "agent"
                    ):
                        raw_chunk = event["data"]["chunk"].content
                        chunk_content = extract_text_content(raw_chunk)
                        if chunk_content:
                            accumulated_response += chunk_content
                            yield f"data: {chunk_content}\n\n"
                            await asyncio.sleep(0)

            except Exception as stream_err:
                error_msg = f"\n[Generation Error: {stream_err!s}]"
                accumulated_response += error_msg
                yield f"data: {error_msg}\n\n"

            finally:
                clean_assistant_text = accumulated_response.strip()
                if clean_assistant_text:
                    background_tasks.add_task(
                        db_service.insert_message,
                        ChatMessageCreateDB(
                            session_id=request.session_id,
                            sender_id=request.assistant_id or "assistant",
                            sender_type="assistant",
                            message=clean_assistant_text,
                        ),
                    )
                    # Trigger progressive summarization check
                    background_tasks.add_task(
                        memory_service.summarize_session_if_needed,
                        request.session_id,
                    )

        return StreamingResponse(
            token_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # ==================== NON-STREAMING MODE ====================
    result = await agent_service.agent.ainvoke(
        {"messages": input_messages},
        config=config,
    )

    final_message = result["messages"][-1]
    final_answer = extract_text_content(final_message.content)

    # Persist assistant response
    background_tasks.add_task(
        db_service.insert_message,
        ChatMessageCreateDB(
            session_id=request.session_id,
            sender_id=request.assistant_id or "assistant",
            sender_type="assistant",
            message=final_answer,
        ),
    )

    # Trigger progressive summarization check
    background_tasks.add_task(
        memory_service.summarize_session_if_needed,
        request.session_id,
    )

    return {
        "session_id": request.session_id,
        "answer": final_answer,
    }