import asyncio
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langfuse.langchain import CallbackHandler

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

    # Attach Langfuse at the graph boundary so model, tool, retriever, embedding,
    # and reranking work is represented in one trace.
    callbacks = [CallbackHandler()] if settings.LANGFUSE_TRACING else []
    thread_config = {
        "configurable": {
            "thread_id": request.session_id,
            "exam": request.exam,
            "session_id": request.session_id,
            "user_id": request.user_id,
        },
        "callbacks": callbacks,
        "metadata": {
            "langfuse_session_id": request.session_id,
            "langfuse_user_id": request.user_id,
            "langfuse_trace_name": "agent-chat",
        },
        "tags": ["agent", f"exam:{request.exam or 'unknown'}"],
    }

    input_state = {
        "messages": [HumanMessage(content=user_query)],
        "exam": request.exam,
        "user_id": request.user_id,
        "session_id": request.session_id,
    }

    # Streaming mode
    if request.stream:

        async def stream_generator():
            accumulated = ""
            try:
                async for event in agent_service.graph.astream_events(
                    input_state, config=thread_config, version="v2"
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
                        history_service.summarize_if_needed(
                            request.session_id, request.user_id
                        )
                    )

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    # Non-streaming mode
    result = await agent_service.graph.ainvoke(input_state, config=thread_config)
    final_text = parse_content(result["messages"][-1].content)

    await history_service.insert_message(
        ChatMessageCreateDB(
            session_id=request.session_id,
            sender_id=request.assistant_id or "assistant",
            sender_type="assistant",
            message=final_text,
        )
    )
    background_tasks.add_task(
        history_service.summarize_if_needed, request.session_id, request.user_id
    )

    return {"session_id": request.session_id, "answer": final_text}
