import asyncio
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.config import get_settings
from app.models.schemas import AgentChatRequest, ChatMessageCreateDB
from app.services.agent_service import agent_service
from app.services.db_service import db_service

settings = get_settings()
router = APIRouter(prefix="/agent", tags=["Agent"])


@router.post("/chat")
async def chat_with_agent(
    request: AgentChatRequest,
    background_tasks: BackgroundTasks,
):
    """Interact with the LangGraph ReAct agent with optional exam filtering context."""
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(
            status_code=400, detail="No user message found in the messages array."
        )

    latest_user_message = user_messages[-1].content

    # Background log user message to Postgres for UI history
    background_tasks.add_task(
        db_service.insert_message,
        ChatMessageCreateDB(
            session_id=request.session_id,
            sender_id=request.user_id,
            sender_type="user",
            message=latest_user_message,
        ),
    )

    # Build system instructions including exam context
    system_prompt = settings.JEE_SYSTEM_PROMPT
    if request.exam:
        system_prompt += (
            f"\n\nContext: The student is preparing for the '{request.exam.upper()}' exam. "
            f"When using the retrieval tool, always supply exam='{request.exam.upper()}' if relevant."
        )

    input_messages = [SystemMessage(content=system_prompt)]

    for msg in request.messages:
        if msg.role == "user":
            input_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            input_messages.append(AIMessage(content=msg.content))
        elif msg.role == "system":
            input_messages.append(SystemMessage(content=msg.content))

    config = {"configurable": {"thread_id": request.session_id}}

    if request.stream:

        async def token_generator():
            accumulated_response = ""
            async for event in agent_service.agent.astream_events(
                {"messages": input_messages},
                config=config,
                version="v2",
            ):
                if event["event"] == "on_chat_model_stream":
                    chunk_content = event["data"]["chunk"].content
                    if chunk_content and isinstance(chunk_content, str):
                        accumulated_response += chunk_content
                        yield chunk_content
                        await asyncio.sleep(0)

            # Persist assistant response to Postgres in background
            background_tasks.add_task(
                db_service.insert_message,
                ChatMessageCreateDB(
                    session_id=request.session_id,
                    sender_id=request.assistant_id,
                    sender_type="assistant",
                    message=accumulated_response,
                ),
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

    # Non-streaming invocation
    result = await agent_service.agent.ainvoke(
        {"messages": input_messages},
        config=config,
    )
    final_answer = result["messages"][-1].content

    # Persist assistant response to Postgres in background
    background_tasks.add_task(
        db_service.insert_message,
        ChatMessageCreateDB(
            session_id=request.session_id,
            sender_id=request.assistant_id,
            sender_type="assistant",
            message=final_answer,
        ),
    )

    return {
        "session_id": request.session_id,
        "answer": final_answer,
    }