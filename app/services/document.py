import asyncio
import os
import tempfile
from typing import Any

from fastapi import UploadFile
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.core.logging import logger
from app.services.vector import vector_service

settings = get_settings()


class DocumentService:
    """PDF text extractor using Marker with chunking and vector storage."""

    def __init__(self):
        self._converter = None
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
        )

    def _get_converter(self):
        if self._converter is None:
            from marker.config.parser import ConfigParser
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict

            config_parser = ConfigParser(
                {
                    "mode": "fast",
                    "disable_ocr": True,
                    "disable_image_extraction": True,
                    "output_format": "markdown",
                }
            )
            self._converter = PdfConverter(
                config=config_parser.generate_config_dict(),
                artifact_dict=create_model_dict(),
                processor_list=config_parser.get_processors(),
                renderer=config_parser.get_renderer(),
            )
        return self._converter

    def _sync_parse(self, file_path: str) -> list[str]:
        from marker.output import text_from_rendered

        converter = self._get_converter()
        rendered = converter(file_path)
        markdown, _, _ = text_from_rendered(rendered)
        return self.splitter.split_text(markdown or "")

    async def process_and_index_pdf(
        self, file: UploadFile, metadata: dict[str, Any]
    ) -> int:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        try:
            logger.info(f"Parsing PDF via Marker thread pool: {file.filename}")
            chunks = await asyncio.to_thread(self._sync_parse, tmp_path)
            documents = [
                Document(
                    page_content=chunk,
                    metadata={**metadata, "source": file.filename, "chunk_index": i},
                )
                for i, chunk in enumerate(chunks)
            ]
            await vector_service.add_documents(documents)
            return len(documents)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


document_service = DocumentService()
