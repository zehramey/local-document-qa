import math

import pytest
from app.domain.embedding import EmbeddingError, EmbeddingErrorCode
from app.services.embedding_provider import FakeEmbeddingProvider


def test_embedding_dimension_matches_configured_value() -> None:
    provider = FakeEmbeddingProvider(vector_dimension=32)

    vectors = provider.embed_documents(["hello world"])

    assert len(vectors) == 1
    assert len(vectors[0]) == 32
    assert provider.model_info.vector_dimension == 32


def test_embeddings_are_normalized_when_configured() -> None:
    provider = FakeEmbeddingProvider(vector_dimension=16, normalized=True)

    vector = provider.embed_query("normalize me")

    norm = math.sqrt(sum(value * value for value in vector))
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_embeddings_are_not_normalized_when_disabled() -> None:
    provider = FakeEmbeddingProvider(vector_dimension=16, normalized=False)

    vector = provider.embed_query("raw vector")

    norm = math.sqrt(sum(value * value for value in vector))
    assert norm != pytest.approx(1.0, abs=1e-6)


def test_batch_embedding_returns_one_vector_per_text() -> None:
    provider = FakeEmbeddingProvider(vector_dimension=8)
    texts = ["first chunk", "second chunk", "third chunk"]

    vectors = provider.embed_documents(texts)

    assert len(vectors) == len(texts)
    assert vectors[0] != vectors[1]
    assert vectors[1] != vectors[2]


def test_document_and_query_embeddings_differ_for_same_text() -> None:
    provider = FakeEmbeddingProvider(vector_dimension=8)

    document_vector = provider.embed_documents(["same text"])[0]
    query_vector = provider.embed_query("same text")

    assert document_vector != query_vector


def test_empty_document_text_is_rejected() -> None:
    provider = FakeEmbeddingProvider()

    with pytest.raises(EmbeddingError) as exc_info:
        provider.embed_documents(["valid text", "   "])

    assert exc_info.value.code == EmbeddingErrorCode.EMPTY_TEXT


def test_empty_query_text_is_rejected() -> None:
    provider = FakeEmbeddingProvider()

    with pytest.raises(EmbeddingError) as exc_info:
        provider.embed_query("")

    assert exc_info.value.code == EmbeddingErrorCode.EMPTY_TEXT
