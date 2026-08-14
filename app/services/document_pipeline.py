"""Orchestrates the upload -> extract -> clean -> chunk -> index pipeline.

Each step is its own tested service (TextExtractionService,
TextCleaningService, ChunkingService, IndexingService); this class only
wires them together in order. IndexingService.index_document already
handles embedding + Qdrant upsert internally.
"""

from app.domain.chunk import ChunkingConfig
from app.domain.document_summary import IngestionResult
from app.services.chunking import ChunkingService
from app.services.file_validation import FileValidator
from app.services.filename_sanitizer import sanitize_filename
from app.services.indexing import IndexingService
from app.services.text_cleaning import TextCleaningService
from app.services.text_extraction import TextExtractionService


class DocumentIngestionService:
    def __init__(
        self,
        validator: FileValidator,
        indexing_service: IndexingService,
        chunking_config: ChunkingConfig,
        extraction_service: TextExtractionService | None = None,
        cleaning_service: TextCleaningService | None = None,
        chunking_service: ChunkingService | None = None,
    ) -> None:
        self._extraction_service = extraction_service or TextExtractionService(validator)
        self._cleaning_service = cleaning_service or TextCleaningService()
        self._chunking_service = chunking_service or ChunkingService()
        self._chunking_config = chunking_config
        self._indexing_service = indexing_service

    def ingest(self, filename: str, content: bytes) -> IngestionResult:
        safe_filename = sanitize_filename(filename)
        extraction_result = self._extraction_service.extract(safe_filename, content)
        cleaned_pages = self._cleaning_service.clean(extraction_result.pages)
        chunks = self._chunking_service.chunk_document(
            extraction_result.document, cleaned_pages, self._chunking_config
        )
        document_text = "\n\n".join(page.text for page in cleaned_pages)
        indexing_result = self._indexing_service.index_document(
            extraction_result.document, chunks, document_text=document_text
        )

        return IngestionResult(
            document_id=extraction_result.document.document_id,
            filename=extraction_result.document.filename,
            chunk_count=len(chunks),
            # From IndexingService's own result, not recomputed here — it's
            # the one place that already knows whether hybrid mode is on
            # (see IndexingService._collection_name). Recomputing it
            # independently was a real bug caught by testing this live: it
            # silently disagreed with the collection IndexingService
            # actually wrote to whenever hybrid search was enabled.
            collection_name=indexing_result.collection_name,
            stale_chunks_removed=indexing_result.stale_chunks_removed,
        )
