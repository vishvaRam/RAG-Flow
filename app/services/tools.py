from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.logging import logger
from app.services.vector_service import vector_service

settings = get_settings()


class EducationalRetrievalInput(BaseModel):
    query: str = Field(
        ...,
        description="Search query containing core academic concepts, keywords, or question text.",
    )
    exam: str | None = Field(
        None,
        description="Target examination filter. Allowed values: 'JEE', 'NEET', 'UPSC', 'BOARDS'.",
    )
    subject: str | None = Field(
        None,
        description="Academic subject name (e.g., 'Physics', 'Chemistry', 'Mathematics', 'Biology').",
    )
    grade: str | None = Field(
        None,
        description="Target class or standard (e.g., 'Class_11', 'Class_12').",
    )
    topic: str | None = Field(
        None,
        description="Specific academic chapter/topic (e.g., 'Thermodynamics', 'Electrostatics', 'Calculus').",
    )
    doc_type: str | None = Field(
        None,
        description="Type of document to fetch: 'NCERT', 'PYQ' (Previous Year Questions), 'Notes', or 'FormulaSheet'.",
    )
    year: int | None = Field(
        None,
        description="Specific exam year for PYQs (e.g., 2021, 2022, 2023, 2024).",
    )


@tool("retrieve_study_material", args_schema=EducationalRetrievalInput)
async def retrieve_study_material(
    query: str,
    exam: str | None = None,
    subject: str | None = None,
    grade: str | None = None,
    topic: str | None = None,
    doc_type: str | None = None,
    year: int | None = None,
) -> str:
    """
    Retrieve relevant academic study materials from the vector index.

    Args:
        query: Core academic question, concepts, or keywords to search.
        exam: Filter by exam: JEE, NEET, UPSC, or BOARDS.
        subject: Filter by subject, e.g. Physics, Chemistry, Maths, Biology.
        grade: Filter by class/standard, e.g. Class_11 or Class_12.
        topic: Filter by specific chapter or topic, e.g. Thermodynamics.
        doc_type: Filter by document type: NCERT, PYQ, Notes, or FormulaSheet.
        year: Filter by specific PYQ exam year, e.g. 2021 or 2024.

    Use filters only when they are relevant to the user's query.
    Returns the most relevant matching documents from the vector index.
    """
    filter_dict: dict[str, Any] = {}
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

    logger.info(f"🔍 Executing vector retrieval: '{query}' | Filters: {filter_dict}")

    try:
        docs = await vector_service.search(
            query=query,
            k=settings.TOP_K_RETRIEVAL,
            filter_dict=filter_dict if filter_dict else None,
        )

        if not docs:
            return (
                "NO_DOCUMENTS_FOUND: No exact matches found for the provided filters. "
                "Consider removing narrow filters (like specific year or doc_type) and retry."
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