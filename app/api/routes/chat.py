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
    """Safely extract plain text from LangChain message content."""
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
    return str(content).strip() if content is not None else ""


@router.post("/chat")
async def chat_with_agent(
    request: AgentChatRequest,
    background_tasks: BackgroundTasks,
):
    """
    Interact with the LangGraph ReAct agent.
    - Preserves internal LangGraph thought signatures.
    - Prevents duplicate SystemMessages in MemorySaver checkpoints.
    - Supports SSE streaming and synchronous request/response.
    """
    # 1. Extract the latest user query
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(
            status_code=400,
            detail="No user message found in the messages array.",
        )
    latest_user_message = user_messages[-1].content

    # 2. Persist incoming user query
    background_tasks.add_task(
        db_service.insert_message,
        ChatMessageCreateDB(
            session_id=request.session_id,
            sender_id=request.user_id,
            sender_type="user",
            message=latest_user_message,
        ),
    )

    config = {"configurable": {"thread_id": request.session_id}}

    # 3. Verify if graph state already exists for this thread
    state_snapshot = await agent_service.agent.aget_state(config)
    existing_messages = (
        state_snapshot.values.get("messages", []) if state_snapshot else []
    )

    input_messages = []

    # Inject System Prompt ONLY on the initial turn for this thread
    if not existing_messages:
        past_context = await memory_service.load_session_context(request.session_id)
        system_prompt = settings.JEE_SYSTEM_PROMPT
        if request.exam:
            system_prompt += (
                f"\n\nContext Directive: The student is preparing for the '{request.exam.upper()}' exam. "
                f"When invoking 'retrieve_study_material', prioritize exam='{request.exam.upper()}'."
            )

        if past_context:
            system_prompt += (
                f"\n\n--- PREVIOUS CONVERSATION CONTEXT ---\n"
                f"{past_context}\n"
                f"------------------------------------"
            )
        input_messages.append(SystemMessage(content=system_prompt))

    # Add the current human input turn
    input_messages.append(HumanMessage(content=latest_user_message))

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
                        chunk = event["data"]["chunk"]
                        # Skip tool call chunks during token generation
                        if getattr(chunk, "tool_call_chunks", None):
                            continue

                        chunk_content = extract_text_content(chunk.content)
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

    background_tasks.add_task(
        db_service.insert_message,
        ChatMessageCreateDB(
            session_id=request.session_id,
            sender_id=request.assistant_id or "assistant",
            sender_type="assistant",
            message=final_answer,
        ),
    )

    background_tasks.add_task(
        memory_service.summarize_session_if_needed,
        request.session_id,
    )

    return {
        "session_id": request.session_id,
        "answer": final_answer,
    }
