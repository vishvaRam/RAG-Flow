from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from app.models.schemas import DocumentUploadResponse
from app.services.document_service import document_service

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    exam: Optional[str] = Form(None, description="Target Exam: JEE, NEET, UPSC, CBSE"),
    subject: Optional[str] = Form(None, description="Subject: Physics, Chemistry, Biology, History"),
    grade: Optional[str] = Form(None, description="Class/Level: Class_11, Class_12, General"),
    topic: Optional[str] = Form(None, description="Chapter/Unit: Electrostatics, Polity, Organic_Chemistry"),
    doc_type: Optional[str] = Form(None, description="Type: NCERT, PYQ, Notes, Reference"),
    year: Optional[int] = Form(None, description="Publication or Question Paper Year"),
):
    """Upload and parse PDF with educational metadata taxonomy into PGVector."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only PDF files are supported.",
        )

    # Build clean metadata dictionary (excluding None values)
    metadata = {}
    if exam:
        metadata["exam"] = exam.strip().upper()
    if subject:
        metadata["subject"] = subject.strip().capitalize()
    if grade:
        metadata["grade"] = grade.strip()
    if topic:
        metadata["topic"] = topic.strip()
    if doc_type:
        metadata["doc_type"] = doc_type.strip().upper()
    if year:
        metadata["year"] = int(year)

    chunks_count = await document_service.process_and_index_pdf(file, metadata)

    return DocumentUploadResponse(
        filename=file.filename,
        chunks_indexed=chunks_count,
        message="Document indexed into PGVector with educational metadata.",
    )