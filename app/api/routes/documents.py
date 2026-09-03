from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.models.schemas import DocumentUploadResponse
from app.services.document import document_service

router = APIRouter(prefix="/documents", tags=["Documents"])


def sanitize(val: str | None) -> str | None:
    if not val:
        return None
    cleaned = val.strip()
    return None if cleaned.lower() in {"none", "null", "string", ""} else cleaned


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    exam: str | None = Form(None, description="Target Exam: JEE, NEET, UPSC, CBSE"),
    subject: str | None = Form(
        None, description="Subject: Physics, Chemistry, Mathematics, Biology"
    ),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Invalid file format. Only PDF files are supported."
        )

    meta = {}
    if c_exam := sanitize(exam):
        meta["exam"] = c_exam.upper()
    if c_subj := sanitize(subject):
        meta["subject"] = c_subj.capitalize()

    count = await document_service.process_and_index_pdf(file, meta)
    return DocumentUploadResponse(
        filename=file.filename,
        chunks_indexed=count,
        message="Successfully parsed and indexed document.",
    )
