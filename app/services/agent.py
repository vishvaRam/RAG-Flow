from langchain_core.messages import AnyMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.core.config import get_settings
from app.models.schemas import AgentState
from app.services.history import history_service
from app.services.tools import retrieve_study_material

settings = get_settings()


class AgentService:
    """LangGraph execution engine for ReAct agent workflows."""

    def __init__(self):
        self.tools = [retrieve_study_material]

        # Initialize LLM based on provider configuration
        self.base_llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            base_url=settings.LLM_PROVIDER_URL,
            api_key=settings.LLM_API_KEY,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
            timeout=settings.LLM_TIMEOUT,
            extra_body={"reasoning": {"effort": "minimal"}},
        )
        self.bound_llm = self.base_llm.bind_tools(self.tools)
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    async def _init_context(self, state: AgentState) -> dict[str, AnyMessage]:
        """Ensures system prompt and context summary are injected on first turn."""
        messages = state.get("messages", [])
        has_system = any(isinstance(m, SystemMessage) for m in messages)
        if not has_system:
            sys_prompt = settings.JEE_SYSTEM_PROMPT
            exam = state.get("exam")
            session_id = state.get("session_id")

            if exam:
                sys_prompt += f"\n\n{settings.JEE_CONTEXT_PROMPT.format(exam=exam.strip().upper())}"

            if session_id:
                past_context = await history_service.load_context(session_id)
                if past_context:
                    sys_prompt += f"\n\n--- PREVIOUS SESSION SUMMARY ---\n{past_context}\n---------------------------------"

            return {"messages": [SystemMessage(content=sys_prompt)]}
        return {}

    async def _call_model(self, state: AgentState) -> dict[str, list[AnyMessage]]:
        response = await self.bound_llm.ainvoke(
            state["messages"],
        )
        return {"messages": [response]}

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("init_context", self._init_context)
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", ToolNode(self.tools, handle_tool_errors=True))

        workflow.add_edge(START, "init_context")
        workflow.add_edge("init_context", "agent")
        workflow.add_conditional_edges(
            "agent", tools_condition, {"tools": "tools", END: END}
        )
        workflow.add_edge("tools", "agent")
        return workflow.compile(checkpointer=self.checkpointer)


agent_service = AgentService()
