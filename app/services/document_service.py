import asyncio
import os
import tempfile
from typing import Any

from fastapi import UploadFile
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

from app.core.config import get_settings
from app.core.logging import logger
from app.services.vector_service import vector_service

settings = get_settings()


class DocumentService:
    """Extracts Markdown from PDFs via Marker in a worker thread and chunks into PGVector."""

    def __init__(self):
        marker_config = {
            "mode": "fast",
            "disable_ocr": True,
            "disable_image_extraction": True,
            "output_format": "markdown",
        }
        config_parser = ConfigParser(marker_config)

        self.converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
        )

    def _sync_parse_and_chunk(self, file_path: str) -> list[str]:
        """Synchronous CPU-bound parsing and chunking to run inside worker thread."""
        rendered = self.converter(file_path)
        markdown_content, _, _ = text_from_rendered(rendered)
        markdown_content = markdown_content or ""
        return self.splitter.split_text(markdown_content)

    async def process_and_index_pdf(
        self,
        file: UploadFile,
        metadata: dict[str, Any],
    ) -> int:
        """Write PDF, parse in a separate thread pool to prevent event-loop blocking, and index."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            file_bytes = await file.read()
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            logger.info(
                f"📄 Parsing PDF with Marker (offloaded to thread): {file.filename}"
            )

            # Offload synchronous CPU work to a separate thread
            chunks = await asyncio.to_thread(self._sync_parse_and_chunk, tmp_path)
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
