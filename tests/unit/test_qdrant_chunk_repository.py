import pytest
from app.domain.chunk import Chunk
from app.domain.embedding import EmbeddingModelInfo
from app.domain.indexing import IndexingError, IndexingErrorCode
from app.repositories.qdrant_chunk_repository import QdrantChunkRepository, collection_name_for
from qdrant_client import QdrantClient


def _make_chunk(
    chunk_id: str, document_id: str = "doc-1", chunk_index: int = 0, text: str = "chunk text"
) -> Chunk:
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


@pytest.fixture
def repository() -> QdrantChunkRepository:
    return QdrantChunkRepository(QdrantClient(location=":memory:"))


def test_collection_name_ties_model_and_schema_version() -> None:
    model_info = EmbeddingModelInfo(
        model_id="BAAI/bge-m3", revision="main", vector_dimension=1024, normalized=True
    )

    assert collection_name_for(model_info) == "BAAI__bge-m3__v1"


def test_ensure_collection_creates_it_with_cosine_distance(
    repository: QdrantChunkRepository,
) -> None:
    repository.ensure_collection("test_collection", vector_dimension=4)

    info = repository._client.get_collection("test_collection")
    assert info.config.params.vectors.size == 4
    assert info.config.params.vectors.distance.name == "COSINE"


def test_ensure_collection_is_idempotent(repository: QdrantChunkRepository) -> None:
    repository.ensure_collection("test_collection", vector_dimension=4)
    repository.ensure_collection("test_collection", vector_dimension=4)

    assert repository.count("test_collection") == 0


def test_ensure_collection_rejects_dimension_mismatch(repository: QdrantChunkRepository) -> None:
    repository.ensure_collection("test_collection", vector_dimension=4)

    with pytest.raises(IndexingError) as exc_info:
        repository.ensure_collection("test_collection", vector_dimension=8)

    assert exc_info.value.code == IndexingErrorCode.VECTOR_DIMENSION_MISMATCH


def test_upsert_and_count_chunks(repository: QdrantChunkRepository) -> None:
    repository.ensure_collection("test_collection", vector_dimension=4)
    chunks = [_make_chunk(chunk_id="a" * 64), _make_chunk(chunk_id="b" * 64, chunk_index=1)]
    vectors = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]

    repository.upsert_chunks("test_collection", chunks, vectors)

    assert repository.count("test_collection") == 2


def test_find_chunk_ids_by_document_filters_correctly(repository: QdrantChunkRepository) -> None:
    repository.ensure_collection("test_collection", vector_dimension=4)
    doc1_chunk = _make_chunk(chunk_id="a" * 64, document_id="doc-1")
    doc2_chunk = _make_chunk(chunk_id="b" * 64, document_id="doc-2")
    repository.upsert_chunks(
        "test_collection", [doc1_chunk, doc2_chunk], [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
    )

    chunk_ids = repository.find_chunk_ids_by_document("test_collection", "doc-1")

    assert chunk_ids == {"a" * 64}


def test_delete_by_document_id_removes_only_that_documents_chunks(
    repository: QdrantChunkRepository,
) -> None:
    repository.ensure_collection("test_collection", vector_dimension=4)
    doc1_chunk = _make_chunk(chunk_id="a" * 64, document_id="doc-1")
    doc2_chunk = _make_chunk(chunk_id="b" * 64, document_id="doc-2")
    repository.upsert_chunks(
        "test_collection", [doc1_chunk, doc2_chunk], [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
    )

    repository.delete_by_document_id("test_collection", "doc-1")

    assert repository.count("test_collection") == 1
    assert repository.find_chunk_ids_by_document("test_collection", "doc-1") == set()


def test_upserting_same_chunk_id_twice_does_not_duplicate(
    repository: QdrantChunkRepository,
) -> None:
    repository.ensure_collection("test_collection", vector_dimension=4)
    chunk = _make_chunk(chunk_id="a" * 64, text="original text")

    repository.upsert_chunks("test_collection", [chunk], [[0.1, 0.2, 0.3, 0.4]])
    updated_chunk = _make_chunk(chunk_id="a" * 64, text="updated text")
    repository.upsert_chunks("test_collection", [updated_chunk], [[0.9, 0.9, 0.9, 0.9]])

    assert repository.count("test_collection") == 1


def test_qdrant_unavailable_raises_indexing_error() -> None:
    unreachable_client = QdrantClient(url="http://127.0.0.1:1", timeout=1)
    repository = QdrantChunkRepository(unreachable_client)

    with pytest.raises(IndexingError) as exc_info:
        repository.ensure_collection("test_collection", vector_dimension=4)

    assert exc_info.value.code == IndexingErrorCode.QDRANT_UNAVAILABLE
