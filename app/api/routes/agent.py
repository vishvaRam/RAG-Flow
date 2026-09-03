import asyncio
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import get_settings
from app.models.schemas import AgentChatRequest, ChatMessageCreateDB
from app.services.agent import agent_service
from app.services.history import history_service

settings = get_settings()
router = APIRouter(prefix="/agent", tags=["Agent"])


def parse_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        ).strip()
    return str(content).strip() if content else ""


@router.post("/chat")
async def chat(request: AgentChatRequest, background_tasks: BackgroundTasks):
    user_turns = [m.content for m in request.messages if m.role == "user"]
    if not user_turns:
        raise HTTPException(status_code=400, detail="Missing user prompt.")
    user_query = user_turns[-1]

    # Save user message to database
    await history_service.insert_message(
        ChatMessageCreateDB(
            session_id=request.session_id,
            sender_id=request.user_id,
            sender_type="user",
            message=user_query,
        )
    )

    thread_config = {
        "configurable": {
            "thread_id": request.session_id,
            "exam": request.exam,
        }
    }
    state = await agent_service.graph.aget_state(
        {"configurable": {"thread_id": request.session_id}}
    )
    messages = []

    # Inject system prompt on initial turn
    if not (state and state.values.get("messages")):
        sys_prompt = settings.JEE_SYSTEM_PROMPT

        if request.exam:
            sys_prompt += f"\n\n{settings.JEE_CONTEXT_PROMPT.format(exam=request.exam.strip().upper())}"

        past_context = await history_service.load_context(request.session_id)
        if past_context:
            sys_prompt += f"\n\n--- PREVIOUS SESSION SUMMARY ---\n{past_context}\n---------------------------------"

        messages.append(SystemMessage(content=sys_prompt))

    messages.append(HumanMessage(content=user_query))

    # Streaming mode
    if request.stream:

        async def stream_generator():
            accumulated = ""
            try:
                async for event in agent_service.graph.astream_events(
                    {"messages": messages}, config=thread_config, version="v2"
                ):
                    if (
                        event["event"] == "on_chat_model_stream"
                        and event.get("metadata", {}).get("langgraph_node") == "agent"
                    ):
                        chunk = event["data"]["chunk"]
                        if getattr(chunk, "tool_call_chunks", None):
                            continue
                        text = parse_content(chunk.content)
                        if text:
                            accumulated += text
                            yield f"data: {text}\n\n"
                            await asyncio.sleep(0)
            finally:
                final_response = accumulated.strip()
                if final_response:
                    await history_service.insert_message(
                        ChatMessageCreateDB(
                            session_id=request.session_id,
                            sender_id=request.assistant_id or "assistant",
                            sender_type="assistant",
                            message=final_response,
                        )
                    )
                    asyncio.create_task(
                        history_service.summarize_if_needed(request.session_id)
                    )

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    # Non-streaming mode
    result = await agent_service.graph.ainvoke(
        {"messages": messages}, config=thread_config
    )
    final_text = parse_content(result["messages"][-1].content)

    await history_service.insert_message(
        ChatMessageCreateDB(
            session_id=request.session_id,
            sender_id=request.assistant_id or "assistant",
            sender_type="assistant",
            message=final_text,
        )
    )
    background_tasks.add_task(history_service.summarize_if_needed, request.session_id)

    return {"session_id": request.session_id, "answer": final_text}
