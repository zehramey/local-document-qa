"""End-to-end demo: extraction -> chunking -> embedding -> Qdrant indexing.

Uses FakeEmbeddingProvider and an in-memory Qdrant instance (no real model
download, no external server) so it runs as a fast, deterministic part of
the default test suite. Prints the resulting chunk/point counts for a
small sample document, per the phase's acceptance requirement.
"""

from app.domain.chunk import ChunkingConfig
from app.repositories.qdrant_chunk_repository import QdrantChunkRepository
from app.services.chunking import ChunkingService
from app.services.embedding_provider import FakeEmbeddingProvider
from app.services.file_validation import FileValidationConfig, FileValidator
from app.services.indexing import IndexingService
from app.services.text_extraction import TextExtractionService
from qdrant_client import QdrantClient

_SAMPLE_DOCUMENT = """Introduction

Local document question answering keeps every step on the user's own \
machine wherever possible. This section introduces the overall goal.

Architecture

The system extracts text from PDF or TXT files, splits it into chunks, \
embeds each chunk locally, and stores the vectors in Qdrant alongside the \
source metadata needed to cite an answer.

Limitations

Scanned PDFs without extractable text are rejected rather than silently \
producing an empty answer, since this phase does not implement OCR.
"""


def test_small_document_end_to_end_chunk_and_point_counts(capsys) -> None:  # noqa: ANN001
    validator = FileValidator(FileValidationConfig(max_size_bytes=1024 * 1024))
    extraction_service = TextExtractionService(validator)
    extraction_result = extraction_service.extract("demo.txt", _SAMPLE_DOCUMENT.encode("utf-8"))

    chunking_service = ChunkingService()
    config = ChunkingConfig(max_tokens=60, overlap_tokens=10)
    chunks = chunking_service.chunk_document(
        extraction_result.document, extraction_result.pages, config
    )

    provider = FakeEmbeddingProvider(vector_dimension=16, model_id="demo-fake-model")
    repository = QdrantChunkRepository(QdrantClient(location=":memory:"))
    indexing_service = IndexingService(provider, repository)

    result = indexing_service.index_document(extraction_result.document, chunks)

    print(
        f"\n[Faz 4 demo] '{extraction_result.document.filename}' -> "
        f"{len(chunks)} chunk, Qdrant collection '{result.collection_name}' -> "
        f"{result.total_points_for_document} point"
    )

    assert len(chunks) > 0
    assert result.chunks_indexed == len(chunks)
    assert result.total_points_for_document == len(chunks)
    assert repository.count(result.collection_name) == len(chunks)
