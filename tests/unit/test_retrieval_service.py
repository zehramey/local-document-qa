from app.domain.chunk import Chunk
from app.domain.document import Document, DocumentType
from app.domain.embedding import EmbeddingModelInfo
from app.domain.retrieval import RetrievalConfig
from app.repositories.qdrant_chunk_repository import QdrantChunkRepository
from app.services.indexing import IndexingService
from app.services.reranker import FakeReranker
from app.services.retrieval import RetrievalService
from qdrant_client import QdrantClient


class _StubEmbeddingProvider:
    """Test-only provider with fully controlled, hand-picked vectors —
    lets tests assert exact similarity ordering deterministically."""

    def __init__(self, vectors: dict[str, list[float]], dimension: int) -> None:
        self._vectors = vectors
        self._info = EmbeddingModelInfo(
            model_id="stub-provider", revision="test", vector_dimension=dimension, normalized=False
        )

    @property
    def model_info(self) -> EmbeddingModelInfo:
        return self._info

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectors[text]


def _make_chunk(chunk_id: str, document_id: str, chunk_index: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        filename="report.pdf",
        page_start=chunk_index + 1,
        page_end=chunk_index + 1,
        chunk_index=chunk_index,
        section_title=None,
        token_count=3,
        text=text,
        chunking_strategy="structural-heading-paragraph-sentence-token-v1",
        chunking_version="1",
    )


def _make_document(document_id: str) -> Document:
    return Document(
        document_id=document_id,
        filename="report.pdf",
        document_type=DocumentType.PDF,
        size_bytes=10,
    )


def test_retrieval_only_returns_chunks_from_the_selected_document() -> None:
    vectors = {
        "doc1 chunk about apples": [1.0, 0.0, 0.0, 0.0],
        "doc2 chunk about apples too": [1.0, 0.0, 0.0, 0.0],
        "apples query": [1.0, 0.0, 0.0, 0.0],
    }
    provider = _StubEmbeddingProvider(vectors, dimension=4)
    repository = QdrantChunkRepository(QdrantClient(location=":memory:"))
    indexing = IndexingService(provider, repository)
    indexing.index_document(
        _make_document("doc-1"),
        [_make_chunk("a" * 64, "doc-1", 0, "doc1 chunk about apples")],
    )
    indexing.index_document(
        _make_document("doc-2"),
        [_make_chunk("b" * 64, "doc-2", 0, "doc2 chunk about apples too")],
    )

    service = RetrievalService(provider, repository)
    result = service.retrieve("apples query", document_id="doc-1", config=RetrievalConfig())

    assert result.has_results
    assert {chunk.document_id for chunk in result.chunks} == {"doc-1"}


def test_retrieval_respects_top_k_limit() -> None:
    vectors = {f"chunk number {i}": [1.0, float(i) * 0.01, 0.0, 0.0] for i in range(5)}
    vectors["query"] = [1.0, 0.0, 0.0, 0.0]
    provider = _StubEmbeddingProvider(vectors, dimension=4)
    repository = QdrantChunkRepository(QdrantClient(location=":memory:"))
    indexing = IndexingService(provider, repository)
    chunks = [_make_chunk(f"{i}" * 64, "doc-1", i, f"chunk number {i}") for i in range(5)]
    indexing.index_document(_make_document("doc-1"), chunks)

    service = RetrievalService(provider, repository)
    config = RetrievalConfig(top_k=2, max_overlap_ratio=1.0)
    result = service.retrieve("query", document_id="doc-1", config=config)

    assert result.total_candidates == 2
    assert len(result.chunks) == 2


def test_retrieval_results_are_sorted_by_score_descending() -> None:
    vectors = {
        "very close to query": [1.0, 0.0, 0.0, 0.0],
        "somewhat close to query": [0.9, 0.1, 0.0, 0.0],
        "far from query": [0.0, 1.0, 0.0, 0.0],
        "query": [1.0, 0.0, 0.0, 0.0],
    }
    provider = _StubEmbeddingProvider(vectors, dimension=4)
    repository = QdrantChunkRepository(QdrantClient(location=":memory:"))
    indexing = IndexingService(provider, repository)
    indexing.index_document(
        _make_document("doc-1"),
        [
            _make_chunk("a" * 64, "doc-1", 0, "far from query"),
            _make_chunk("b" * 64, "doc-1", 1, "very close to query"),
            _make_chunk("c" * 64, "doc-1", 2, "somewhat close to query"),
        ],
    )

    service = RetrievalService(provider, repository)
    config = RetrievalConfig(max_overlap_ratio=1.0)
    result = service.retrieve("query", document_id="doc-1", config=config)

    texts_in_order = [chunk.text for chunk in result.chunks]
    assert texts_in_order == ["very close to query", "somewhat close to query", "far from query"]
    scores = [chunk.retrieval_score for chunk in result.chunks]
    assert scores == sorted(scores, reverse=True)
    assert [chunk.final_rank for chunk in result.chunks] == [1, 2, 3]


def test_near_duplicate_overlapping_chunks_are_reduced() -> None:
    vectors = {
        "the quick brown fox jumps over the lazy dog": [1.0, 0.0, 0.0, 0.0],
        "the quick brown fox jumps over a lazy dog": [0.99, 0.01, 0.0, 0.0],
        "completely unrelated content about oceans": [0.0, 1.0, 0.0, 0.0],
        "query": [1.0, 0.0, 0.0, 0.0],
    }
    provider = _StubEmbeddingProvider(vectors, dimension=4)
    repository = QdrantChunkRepository(QdrantClient(location=":memory:"))
    indexing = IndexingService(provider, repository)
    indexing.index_document(
        _make_document("doc-1"),
        [
            _make_chunk("a" * 64, "doc-1", 0, "the quick brown fox jumps over the lazy dog"),
            _make_chunk("b" * 64, "doc-1", 1, "the quick brown fox jumps over a lazy dog"),
            _make_chunk("c" * 64, "doc-1", 2, "completely unrelated content about oceans"),
        ],
    )

    service = RetrievalService(provider, repository)
    config = RetrievalConfig(max_overlap_ratio=0.7)
    result = service.retrieve("query", document_id="doc-1", config=config)

    assert len(result.chunks) == 2
    kept_texts = {chunk.text for chunk in result.chunks}
    assert "the quick brown fox jumps over the lazy dog" in kept_texts
    assert "completely unrelated content about oceans" in kept_texts


def test_results_below_threshold_are_reported_as_insufficient() -> None:
    vectors = {
        "unrelated chunk": [0.0, 1.0, 0.0, 0.0],
        "query": [1.0, 0.0, 0.0, 0.0],
    }
    provider = _StubEmbeddingProvider(vectors, dimension=4)
    repository = QdrantChunkRepository(QdrantClient(location=":memory:"))
    indexing = IndexingService(provider, repository)
    indexing.index_document(
        _make_document("doc-1"), [_make_chunk("a" * 64, "doc-1", 0, "unrelated chunk")]
    )

    service = RetrievalService(provider, repository)
    config = RetrievalConfig(score_threshold=0.9)
    result = service.retrieve("query", document_id="doc-1", config=config)

    assert result.has_results is False
    assert result.chunks == []


def test_reranker_reorders_results_while_preserving_dense_score() -> None:
    vectors = {
        "dense favorite but off topic": [1.0, 0.0, 0.0, 0.0],
        "less dense similarity but mentions bananas directly": [0.8, 0.2, 0.0, 0.0],
        "bananas": [1.0, 0.0, 0.0, 0.0],
    }
    provider = _StubEmbeddingProvider(vectors, dimension=4)
    repository = QdrantChunkRepository(QdrantClient(location=":memory:"))
    indexing = IndexingService(provider, repository)
    indexing.index_document(
        _make_document("doc-1"),
        [
            _make_chunk("a" * 64, "doc-1", 0, "dense favorite but off topic"),
            _make_chunk(
                "b" * 64, "doc-1", 1, "less dense similarity but mentions bananas directly"
            ),
        ],
    )

    service = RetrievalService(provider, repository, reranker=FakeReranker())
    result = service.retrieve(
        "bananas", document_id="doc-1", config=RetrievalConfig(max_overlap_ratio=1.0)
    )

    assert result.chunks[0].text == "less dense similarity but mentions bananas directly"
    assert all(chunk.retrieval_score is not None for chunk in result.chunks)
    assert all(chunk.reranker_score is not None for chunk in result.chunks)


def test_context_for_llm_truncates_to_llm_top_k() -> None:
    vectors = {f"chunk {i}": [1.0, float(i) * 0.001, 0.0, 0.0] for i in range(6)}
    vectors["query"] = [1.0, 0.0, 0.0, 0.0]
    provider = _StubEmbeddingProvider(vectors, dimension=4)
    repository = QdrantChunkRepository(QdrantClient(location=":memory:"))
    indexing = IndexingService(provider, repository)
    chunks = [_make_chunk(f"{i}" * 64, "doc-1", i, f"chunk {i}") for i in range(6)]
    indexing.index_document(_make_document("doc-1"), chunks)

    service = RetrievalService(provider, repository)
    config = RetrievalConfig(top_k=6, llm_top_k=3, max_overlap_ratio=1.0)
    result = service.retrieve("query", document_id="doc-1", config=config)
    context = service.context_for_llm(result, config)

    assert len(result.chunks) == 6
    assert len(context) == 3
