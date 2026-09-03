from langchain_core.messages import AnyMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.core.config import get_settings
from app.services.tools import retrieve_study_material

settings = get_settings()


class AgentService:
    """LangGraph execution engine for ReAct agent workflows."""

    def __init__(self):
        self.tools = [retrieve_study_material]
        self.base_llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            google_api_key=settings.LLM_API_KEY,
            temperature=settings.TEMPERATURE,
            max_output_tokens=settings.MAX_TOKENS,
            timeout=settings.LLM_TIMEOUT,
        )
        self.bound_llm = self.base_llm.bind_tools(self.tools)
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    async def _call_model(self, state: MessagesState) -> dict[str, list[AnyMessage]]:
        response = await self.bound_llm.ainvoke(state["messages"])
        return {"messages": [response]}

    def _build_graph(self):
        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", ToolNode(self.tools, handle_tool_errors=True))

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent", tools_condition, {"tools": "tools", END: END}
        )
        workflow.add_edge("tools", "agent")
        return workflow.compile(checkpointer=self.checkpointer)


agent_service = AgentService()
