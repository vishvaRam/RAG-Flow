import os
import tempfile
from typing import Any, Dict
from fastapi import UploadFile
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from liteparse import LiteParse

from app.core.config import get_settings
from app.core.logging import logger
from app.services.vector_service import vector_service

settings = get_settings()


class DocumentService:
    """Extracts Markdown from PDFs via liteparse and chunks into PGVector."""

    def __init__(self):
        # Configure LiteParse specifically to render Markdown directly on result.text
        self.parser = LiteParse(
            output_format="markdown",
            image_mode="placeholder",
            extract_links=True,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
        )

    async def process_and_index_pdf(
        self,
        file: UploadFile,
        metadata: Dict[str, Any],
    ) -> int:
        """Write PDF to temporary storage, parse to Markdown, split, and index into PGVector."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            file_bytes = await file.read()
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            logger.info(f"📄 Parsing PDF with liteparse: {file.filename}")
            parse_result = self.parser.parse(tmp_path)
            
            # Markdown output is provided directly on result.text when output_format="markdown"
            markdown_content = parse_result.text or ""
            logger.info(f"📊 Extracted clean Markdown: {len(markdown_content):,} chars")

            chunks = self.splitter.split_text(markdown_content)
            logger.info(f"Generated {len(chunks)} chunks from {file.filename}")

            documents = [
                Document(
                    page_content=chunk,
                    metadata={
                        **metadata,
                        "source": file.filename,
                        "chunk_index": idx,
                    },
                )
                for idx, chunk in enumerate(chunks)
            ]

            await vector_service.add_documents(documents)
            return len(documents)

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


document_service = DocumentService()