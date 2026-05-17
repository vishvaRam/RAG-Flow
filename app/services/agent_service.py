from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

# Import your existing services and configurations
from app.core.config import get_settings
from app.core.logging import logger
from app.services.rag_service import rag_service
from app.utils.prompts import AGENT_ROUTER_PROMPT, PROBLEM_SOLVING_PROMPT

settings = get_settings()


# 1. Define the Pydantic model for Structured Routing (2 options only)
class RouterDecision(BaseModel):
    """Decide the routing path for JEE examination queries."""

    route: Literal["rag_pipeline", "direct_llm"] = Field(
        ...,
        description=(
            "Choose 'rag_pipeline' for general JEE information, syllabi, exam dates, college cutoffs, "
            "or descriptive, high-level textbook conceptual definitions. "
            "Choose 'direct_llm' for all analytical problem-solving, calculations, equations, MCQs, "
            "as well as any casual greetings, chitchat, or topics completely unrelated to JEE."
        ),
    )
    reasoning: str = Field(
        ...,
        description="A clear, short reason detailing why the query falls into this specific path.",
    )


# 2. Define the Graph State
class AgentState(TypedDict):
    query: str
    rewritten_query: str
    context: Any  # Memory service context output
    subject_filter: Optional[str]
    topic_filter: Optional[str]
    top_k: int
    route: Optional[str]
    reranked_docs: List[Dict[str, Any]]
    conversation_messages: List[Dict[str, Any]]


class AgentService:
    def __init__(self):
        # Initializing Gemini using langchain-google-genai
        self._base_model = ChatGoogleGenerativeAI(
            model=getattr(settings, "LLM_MODEL", "gemini-3.1-flash-lite-preview"),
            google_api_key=getattr(settings, "LLM_API_KEY", None),
            temperature=0,
        )

        # Bind the Pydantic structured output router for Gemini
        self.router_llm = self._base_model.with_structured_output(RouterDecision)
        self.graph = self._build_graph()

    async def route_query_node(self, state: AgentState) -> Dict[str, Any]:
        """Node that uses Pydantic structured output to classify the incoming query path."""
        rewritten_query = state["rewritten_query"]
        prompt = AGENT_ROUTER_PROMPT.format(rewritten_query=rewritten_query)

        try:
            decision = await self.router_llm.ainvoke(prompt)
            logger.info(
                f"🤖 LangGraph Router choice: [{decision.route.upper()}] | Reason: {decision.reasoning}"  # type: ignore
            )
            return {"route": decision.route}  # type: ignore

        except Exception as e:
            logger.error(f"Routing node failure, defaulting to RAG: {e}", exc_info=True)
            return {"route": "rag_pipeline"}

    async def run_rag_pipeline_node(self, state: AgentState) -> Dict[str, Any]:
        """Node executing your classic Hybrid RAG Pipeline tracking."""
        logger.info("⚡ Executing Hybrid RAG pipeline execution route...")

        reranked, _ = await rag_service.process_query(
            rewritten_query=state["rewritten_query"],
            subject_filter=state["subject_filter"],
            topic_filter=state["topic_filter"],
            top_k=state["top_k"],
        )

        rag_context = rag_service.build_context(reranked)
        current_user_message = rag_service.create_rag_prompt(
            state["query"], rag_context
        )
        conversation_messages = self._assemble_messages(state, current_user_message)

        return {
            "reranked_docs": reranked,
            "conversation_messages": conversation_messages,
        }

    async def run_direct_llm_node(self, state: AgentState) -> Dict[str, Any]:
        """Node specialized for both math challenges and direct off-scope guardrail filtering."""
        logger.info("🧠 Skipping RAG. Executing direct processing path...")

        problem_prompt = PROBLEM_SOLVING_PROMPT.format(state=state)
        conversation_messages = self._assemble_messages(state, problem_prompt)

        return {
            "reranked_docs": [],
            "conversation_messages": conversation_messages,
        }

    def _assemble_messages(
        self, state: AgentState, current_user_content: str
    ) -> List[Dict[str, Any]]:
        """Helper to unify historical items, summaries, system prompts, and the generated request layout."""
        context = state["context"]
        system_content_parts = [rag_service.llm.SYSTEM_MESSAGE["content"]]

        if context and context.summary:
            system_content_parts.append(
                f"\n\nPrevious Conversation Summary:\n{context.summary}"
            )

        unified_system_message = {
            "role": "system",
            "content": "\n".join(system_content_parts),
        }

        conversation_messages = [unified_system_message]

        if context and context.recent_messages:
            for msg in context.recent_messages:
                conversation_messages.append(
                    {
                        "role": "user" if msg["role"] == "user" else "assistant",
                        "content": msg["content"],
                    }
                )

        conversation_messages.append({"role": "user", "content": current_user_content})
        return conversation_messages

    def _build_graph(self):
        """Compiles the operational LangGraph design."""
        workflow = StateGraph(AgentState)

        # Add Nodes
        workflow.add_node("route_query", self.route_query_node)
        workflow.add_node("run_rag_pipeline", self.run_rag_pipeline_node)
        workflow.add_node("run_direct_llm", self.run_direct_llm_node)

        # Set Entry Point
        workflow.set_entry_point("route_query")

        # Define Router Conditional Connections
        workflow.add_conditional_edges(
            "route_query",
            lambda state: state["route"],
            {
                "rag_pipeline": "run_rag_pipeline",
                "direct_llm": "run_direct_llm",
            },
        )

        # End nodes
        workflow.add_edge("run_rag_pipeline", END)
        workflow.add_edge("run_direct_llm", END)

        return workflow.compile()

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Main interface function to execute the LangGraph workflow runtime."""
        return await self.graph.ainvoke(inputs)


# Global singleton instance
agent_service = AgentService()
