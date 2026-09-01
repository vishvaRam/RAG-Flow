from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.models.schemas import DocumentUploadResponse
from app.services.document_service import document_service

router = APIRouter(prefix="/documents", tags=["Documents"])


def sanitize_input(value: str | None) -> str | None:
    """Helper to clean form input and strip Swagger/UI defaults."""
    if value is None:
        return None
    val = value.strip()
    if not val or val.lower() in {"string", "none", "null"}:
        return None
    return val


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    exam: str | None = Form(None, description="Target Exam: JEE, NEET, UPSC, CBSE"),
    subject: str | None = Form(
        None, description="Subject: Physics, Chemistry, Biology, History"
    ),
    grade: str | None = Form(
        None, description="Class/Level: Class_11, Class_12, General"
    ),
    topic: str | None = Form(
        None, description="Chapter/Unit: Electrostatics, Polity, Organic_Chemistry"
    ),
    doc_type: str | None = Form(None, description="Type: NCERT, PYQ, Notes, Reference"),
    year: int | None = Form(None, description="Publication or Question Paper Year"),
):
    """Upload and parse PDF with educational metadata taxonomy into PGVector."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only PDF files are supported.",
        )

    # Sanitize and build clean metadata dictionary
    metadata = {}

    clean_exam = sanitize_input(exam)
    if clean_exam:
        metadata["exam"] = clean_exam.upper()

    clean_subject = sanitize_input(subject)
    if clean_subject:
        metadata["subject"] = clean_subject.capitalize()

    clean_grade = sanitize_input(grade)
    if clean_grade:
        metadata["grade"] = clean_grade

    clean_topic = sanitize_input(topic)
    if clean_topic:
        metadata["topic"] = clean_topic

    clean_doc_type = sanitize_input(doc_type)
    if clean_doc_type:
        metadata["doc_type"] = clean_doc_type.upper()

    if year:
        metadata["year"] = int(year)

    chunks_count = await document_service.process_and_index_pdf(file, metadata)

    return DocumentUploadResponse(
        filename=file.filename,
        chunks_indexed=chunks_count,
        message="Document indexed into PGVector with educational metadata.",
    )
