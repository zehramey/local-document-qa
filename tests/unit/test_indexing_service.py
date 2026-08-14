from app.domain.chunk import Chunk
from app.domain.document import Document, DocumentType
from app.repositories.qdrant_chunk_repository import QdrantChunkRepository, collection_name_for
from app.services.contextual_chunking import FakeChunkContextGenerator
from app.services.embedding_provider import FakeEmbeddingProvider
from app.services.indexing import IndexingService
from qdrant_client import QdrantClient


def _make_document(document_id: str = "doc-1") -> Document:
    return Document(
        document_id=document_id,
        filename="report.pdf",
        document_type=DocumentType.PDF,
        size_bytes=100,
    )


def _make_chunk(chunk_id: str, document_id: str, chunk_index: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        filename="report.pdf",
        page_start=1,
        page_end=1,
        chunk_index=chunk_index,
        section_title=None,
        token_count=3,
        text=text,
        chunking_strategy="structural-heading-paragraph-sentence-token-v1",
        chunking_version="1",
    )


def _make_service() -> tuple[IndexingService, FakeEmbeddingProvider, QdrantChunkRepository]:
    provider = FakeEmbeddingProvider(vector_dimension=8)
    repository = QdrantChunkRepository(QdrantClient(location=":memory:"))
    return IndexingService(provider, repository), provider, repository


def test_index_document_creates_expected_number_of_points() -> None:
    service, _, repository = _make_service()
    document = _make_document()
    chunks = [
        _make_chunk("a" * 64, document.document_id, 0, "first chunk"),
        _make_chunk("b" * 64, document.document_id, 1, "second chunk"),
        _make_chunk("c" * 64, document.document_id, 2, "third chunk"),
    ]

    result = service.index_document(document, chunks)

    assert result.chunks_indexed == 3
    assert result.total_points_for_document == 3
    assert repository.count(result.collection_name) == 3


def test_reindexing_same_document_with_same_chunks_does_not_duplicate() -> None:
    service, _, repository = _make_service()
    document = _make_document()
    chunks = [_make_chunk("a" * 64, document.document_id, 0, "first chunk")]

    service.index_document(document, chunks)
    result = service.index_document(document, chunks)

    assert result.total_points_for_document == 1


def test_reindexing_with_fewer_chunks_removes_stale_points() -> None:
    service, _, repository = _make_service()
    document = _make_document()
    original_chunks = [
        _make_chunk("a" * 64, document.document_id, 0, "first chunk"),
        _make_chunk("b" * 64, document.document_id, 1, "second chunk"),
    ]
    service.index_document(document, original_chunks)

    new_chunks = [_make_chunk("c" * 64, document.document_id, 0, "only chunk now")]
    result = service.index_document(document, new_chunks)

    assert result.stale_chunks_removed == 2
    assert result.total_points_for_document == 1
    assert repository.count(result.collection_name) == 1


def test_delete_document_removes_all_its_points() -> None:
    service, _, repository = _make_service()
    document = _make_document()
    chunks = [_make_chunk("a" * 64, document.document_id, 0, "chunk text")]
    result = service.index_document(document, chunks)

    service.delete_document(document.document_id)

    assert repository.count(result.collection_name) == 0


def test_different_documents_do_not_interfere() -> None:
    service, _, repository = _make_service()
    doc1 = _make_document("doc-1")
    doc2 = _make_document("doc-2")
    service.index_document(doc1, [_make_chunk("a" * 64, "doc-1", 0, "doc1 chunk")])
    result2 = service.index_document(doc2, [_make_chunk("b" * 64, "doc-2", 0, "doc2 chunk")])

    service.delete_document("doc-1")

    assert repository.count(result2.collection_name) == 1


def test_switching_embedding_model_uses_a_different_collection() -> None:
    document = _make_document()
    chunk = _make_chunk("a" * 64, document.document_id, 0, "chunk text")

    repository = QdrantChunkRepository(QdrantClient(location=":memory:"))
    provider_a = FakeEmbeddingProvider(vector_dimension=8, model_id="model-a")
    provider_b = FakeEmbeddingProvider(vector_dimension=16, model_id="model-b")

    result_a = IndexingService(provider_a, repository).index_document(document, [chunk])
    result_b = IndexingService(provider_b, repository).index_document(document, [chunk])

    assert result_a.collection_name != result_b.collection_name
    assert result_a.collection_name == collection_name_for(provider_a.model_info)
    assert result_b.collection_name == collection_name_for(provider_b.model_info)
    assert repository.count(result_a.collection_name) == 1
    assert repository.count(result_b.collection_name) == 1


def test_context_generator_augments_embedded_text_but_not_stored_chunk_text() -> None:
    provider = FakeEmbeddingProvider(vector_dimension=8)
    repository = QdrantChunkRepository(QdrantClient(location=":memory:"))
    context_generator = FakeChunkContextGenerator(context="situating context")
    service = IndexingService(provider, repository, context_generator)
    document = _make_document()
    chunk = _make_chunk("a" * 64, document.document_id, 0, "original chunk text")

    service.index_document(document, [chunk], document_text="the full document")

    assert provider.last_embedded_texts == ["situating context\n\noriginal chunk text"]
    assert context_generator.calls == [("the full document", "original chunk text")]
    collection_name = collection_name_for(provider.model_info)
    stored = repository.search_similar(
        collection_name, provider.embed_query("original chunk text"), document.document_id, 1
    )
    assert stored[0].payload["text"] == "original chunk text"


def test_context_generator_is_skipped_without_document_text() -> None:
    provider = FakeEmbeddingProvider(vector_dimension=8)
    repository = QdrantChunkRepository(QdrantClient(location=":memory:"))
    context_generator = FakeChunkContextGenerator(context="situating context")
    service = IndexingService(provider, repository, context_generator)
    document = _make_document()
    chunk = _make_chunk("a" * 64, document.document_id, 0, "original chunk text")

    service.index_document(document, [chunk])

    assert provider.last_embedded_texts == ["original chunk text"]
    assert context_generator.calls == []


def test_no_context_generator_embeds_chunk_text_unmodified() -> None:
    service, provider, _ = _make_service()
    document = _make_document()
    chunk = _make_chunk("a" * 64, document.document_id, 0, "plain chunk text")

    service.index_document(document, [chunk], document_text="the full document")

    assert provider.last_embedded_texts == ["plain chunk text"]
