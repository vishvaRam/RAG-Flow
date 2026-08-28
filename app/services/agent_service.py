from typing import Any, Dict, List, Optional
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.logging import logger
from app.services.vector_service import vector_service

settings = get_settings()


# ==================== Educational Retrieval Tool Schema ====================


class EducationalRetrievalInput(BaseModel):
    query: str = Field(
        ..., description="The search query for document retrieval"
    )
    exam: Optional[str] = Field(
        None, description="Target exam filter (e.g. 'JEE_MAIN', 'JEE_ADVANCED', 'NEET', 'UPSC')"
    )
    subject: Optional[str] = Field(
        None, description="Academic subject (e.g. 'Physics', 'Chemistry', 'Biology', 'History')"
    )
    grade: Optional[str] = Field(
        None, description="Class/standard (e.g. 'Class_11', 'Class_12')"
    )
    topic: Optional[str] = Field(
        None, description="Specific topic or unit (e.g. 'Thermodynamics', 'Organic_Chemistry')"
    )
    doc_type: Optional[str] = Field(
        None, description="Document type filter (e.g. 'NCERT', 'PYQ', 'Notes')"
    )
    year: Optional[int] = Field(
        None, description="Specific exam or publication year"
    )


# ==================== Vector Retrieval Tool ====================


@tool("retrieve_study_material", args_schema=EducationalRetrievalInput)
async def retrieve_study_material(
    query: str,
    exam: Optional[str] = None,
    subject: Optional[str] = None,
    grade: Optional[str] = None,
    topic: Optional[str] = None,
    doc_type: Optional[str] = None,
    year: Optional[int] = None,
) -> str:
    """Retrieve relevant study material, definitions, formulas, and past papers with optional metadata filters."""
    filter_dict: Dict[str, Any] = {}
    if exam:
        filter_dict["exam"] = exam.strip().upper()
    if subject:
        filter_dict["subject"] = subject.strip().capitalize()
    if grade:
        filter_dict["grade"] = grade.strip()
    if topic:
        filter_dict["topic"] = topic.strip()
    if doc_type:
        filter_dict["doc_type"] = doc_type.strip().upper()
    if year:
        filter_dict["year"] = year

    logger.info(f"🔍 Educational Query: '{query}' | Filters: {filter_dict}")

    docs = await vector_service.search(
        query=query,
        k=settings.TOP_K_RETRIEVAL,
        filter_dict=filter_dict if filter_dict else None,
    )

    if not docs:
        return "No relevant study materials found matching the query and filters."

    formatted_docs = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        source_info = (
            f"Source: {meta.get('source', 'Unknown')} | "
            f"Exam: {meta.get('exam', 'General')} | "
            f"Subject: {meta.get('subject', 'General')} | "
            f"Topic: {meta.get('topic', 'General')}"
        )
        formatted_docs.append(
            f'<document id="{i}" {source_info}>\n{doc.page_content}\n</document>'
        )

    return "\n\n".join(formatted_docs)


# ==================== Custom LangGraph Agent Service ====================


class AgentService:
    """LangGraph agent compiled with in-memory checkpointer."""

    def __init__(self):
        self.tools = [retrieve_study_material]

        # Fix Gemini OpenAI-proxy 400 thought_signature by disabling reasoning_effort
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            reasoning_effort="minimal",
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_PROVIDER_URL,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
            timeout=settings.LLM_TIMEOUT,
        ).bind_tools(self.tools)

        self.checkpointer = MemorySaver()
        self.agent = self._build_graph()
        logger.info("✓ Educational Agent StateGraph compiled with MemorySaver")

    async def _call_model(self, state: MessagesState) -> Dict[str, List[AnyMessage]]:
        messages = state["messages"]
        response = await self.llm.ainvoke(messages)
        return {"messages": [response]}

    def _build_graph(self):
        workflow = StateGraph(MessagesState)

        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", ToolNode(self.tools))

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {"tools": "tools", END: END},
        )
        workflow.add_edge("tools", "agent")

        return workflow.compile(checkpointer=self.checkpointer)


agent_service = AgentService()