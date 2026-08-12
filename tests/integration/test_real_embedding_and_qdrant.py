"""Verifies the real embedding model (BAAI/bge-m3) and a real Qdrant server.

Excluded from the default test run (see pyproject `addopts`) because it
downloads a real model (first run only, then cached) and requires
`pip install -e ".[embeddings]"` plus a reachable Qdrant instance
(e.g. `docker compose up -d qdrant`).

Run explicitly with: pytest -m integration
"""

import pytest
from app.domain.chunk import Chunk
from app.domain.document import Document, DocumentType
from app.repositories.qdrant_chunk_repository import QdrantChunkRepository
from app.services.embedding_models import BGE_M3
from app.services.indexing import IndexingService
from app.services.sentence_transformer_provider import SentenceTransformerEmbeddingProvider
from qdrant_client import QdrantClient

pytestmark = pytest.mark.integration


def test_real_bge_m3_embeddings_are_indexed_in_real_qdrant() -> None:
    provider = SentenceTransformerEmbeddingProvider(
        model_id=BGE_M3.model_id,
        device="cpu",
        query_instruction=BGE_M3.query_instruction,
        passage_instruction=BGE_M3.passage_instruction,
    )
    assert provider.model_info.vector_dimension > 0

    repository = QdrantChunkRepository(QdrantClient(host="localhost", port=6333))
    service = IndexingService(provider, repository)

    document = Document(
        document_id="integration-test-doc",
        filename="integration.txt",
        document_type=DocumentType.TXT,
        size_bytes=42,
    )
    chunk = Chunk(
        chunk_id="c" * 64,
        document_id=document.document_id,
        filename=document.filename,
        page_start=1,
        page_end=1,
        chunk_index=0,
        section_title=None,
        token_count=5,
        text="Local document question answering system.",
        chunking_strategy="structural-heading-paragraph-sentence-token-v1",
        chunking_version="1",
    )

    result = service.index_document(document, [chunk])

    assert result.chunks_indexed == 1
    assert result.total_points_for_document == 1

    query_vector = provider.embed_query("What does this system do?")
    assert len(query_vector) == provider.model_info.vector_dimension

    service.delete_document(document.document_id)
