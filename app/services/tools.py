from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.logging import logger
from app.services.vector import vector_service

settings = get_settings()


class EducationalRetrievalInput(BaseModel):
    query: str = Field(
        ...,
        description=(
            "Focused keyword-rich search query containing core scientific concepts, "
            "formulas, theorems, or specific question text to match in textbooks and PYQs."
        ),
    )
    subject: str | None = Field(
        None,
        description=(
            "Academic subject category to narrow down search: 'Physics', 'Chemistry', "
            "'Mathematics', 'Biology'. Leave null if the topic crosses multiple subjects."
        ),
    )


@tool("retrieve_study_material", args_schema=EducationalRetrievalInput)
async def retrieve_study_material(
    query: str,
    config: RunnableConfig,
    subject: str | None = None,
) -> str:
    """
    Search and retrieve authoritative academic study materials, textbook excerpts,
    derivations, formulas, and past year questions (PYQs) from the vector knowledge base.

    WHEN TO CALL:
    - Whenever solving or explaining academic concepts, formulas, definitions, or derivations.
    - When answering textbook questions, standard curriculum problems, or previous year exam questions.
    - To verify syllabus facts, theorems, or specific numerical problems.

    HOW TO QUERY:
    - Extract essential keywords, concepts, or problem statements instead of conversational phrases.
    - Pass `subject` if explicitly known to avoid cross-subject ambiguity.
    - Target examination filter is automatically applied from system session configuration.

    RETURNS:
    - XML-formatted matching document excerpts with metadata (source, subject, topic).
    """
    exam = config.get("configurable", {}).get("exam")
    filter_dict: dict[str, Any] = {}
    if exam:
        filter_dict["exam"] = str(exam).strip().upper()
    if subject:
        filter_dict["subject"] = subject.strip().capitalize()

    logger.info(f"🔍 Executing vector retrieval: '{query}' | Filters: {filter_dict}")

    try:
        docs = await vector_service.search(
            query=query,
            k=settings.TOP_K_RETRIEVAL,
            filters=filter_dict if filter_dict else None,
        )

        if not docs:
            return (
                "NO_DOCUMENTS_FOUND: No exact matches found for the query in the knowledge base. "
                "Try rephrasing with alternative technical keywords or removing the subject filter."
            )

        formatted_docs = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata or {}
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

    except Exception as exc:
        logger.error(f"Error during retrieval: {exc}", exc_info=True)
        return f"RETRIEVAL_ERROR: Failed to retrieve study material: {exc!s}. Try rephrasing or clearing filters."
