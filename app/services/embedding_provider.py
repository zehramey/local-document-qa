"""Model-independent embedding provider abstraction.

`EmbeddingProvider` separates document (passage) embedding from query
embedding because several retrieval models require different instruction
prefixes for each (see app/services/embedding_models.py). Concrete
providers are responsible for loading their model once and reusing it for
every call — never reloading per request.
"""

import hashlib
import math
from typing import Protocol

from app.domain.embedding import EmbeddingError, EmbeddingErrorCode, EmbeddingModelInfo


class EmbeddingProvider(Protocol):
    @property
    def model_info(self) -> EmbeddingModelInfo: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeds a batch of passages/chunks."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embeds a single search query."""
        ...


def ensure_non_empty_texts(texts: list[str]) -> None:
    for text in texts:
        if not text or not text.strip():
            raise EmbeddingError(
                EmbeddingErrorCode.EMPTY_TEXT, "Empty text cannot be sent to the embedding model."
            )


class FakeEmbeddingProvider:
    """Deterministic, dependency-free provider for tests.

    NOT a real embedding model: vectors are derived from a SHA-256 hash of
    the (instruction-prefixed) text. This exists so tests can exercise
    batching, normalization, and Qdrant integration without downloading or
    running a real model such as BAAI/bge-m3.
    """

    def __init__(
        self,
        vector_dimension: int = 16,
        model_id: str = "fake-embedding-provider",
        normalized: bool = True,
    ) -> None:
        self._info = EmbeddingModelInfo(
            model_id=model_id,
            revision="test",
            vector_dimension=vector_dimension,
            normalized=normalized,
        )
        # Recorded for tests that need to assert on what text was actually
        # sent to embed_documents (e.g. contextual-chunking augmentation) —
        # not used by the embedding logic itself.
        self.last_embedded_texts: list[str] = []

    @property
    def model_info(self) -> EmbeddingModelInfo:
        return self._info

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ensure_non_empty_texts(texts)
        self.last_embedded_texts = list(texts)
        return [self._embed_one(f"passage: {text}") for text in texts]

    def embed_query(self, text: str) -> list[float]:
        ensure_non_empty_texts([text])
        return self._embed_one(f"query: {text}")

    def _embed_one(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        dimension = self._info.vector_dimension
        raw = [(digest[i % len(digest)] / 255.0) * 2 - 1 for i in range(dimension)]
        if not self._info.normalized:
            return raw
        norm = math.sqrt(sum(value * value for value in raw)) or 1.0
        return [value / norm for value in raw]
