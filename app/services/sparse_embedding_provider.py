"""Model-independent sparse (lexical) embedding provider abstraction.

Mirrors EmbeddingProvider (app/services/embedding_provider.py) on purpose:
same embed_documents/embed_query split, same reason (query vs. passage can
need different handling). Kept as a *separate* protocol rather than folded
into EmbeddingProvider because not every embedding model produces a sparse
representation — bge-m3 does (see app/services/bgem3_sparse_provider.py),
most sentence-transformers models don't. A retrieval/indexing path that
never sets a SparseEmbeddingProvider behaves exactly as it did before
hybrid search existed.
"""

import zlib
from typing import Protocol

from app.domain.embedding import SparseVector


class SparseEmbeddingProvider(Protocol):
    def embed_documents_sparse(self, texts: list[str]) -> list[SparseVector]:
        """Embeds a batch of passages/chunks."""
        ...

    def embed_query_sparse(self, text: str) -> SparseVector:
        """Embeds a single search query."""
        ...


class FakeSparseEmbeddingProvider:
    """Deterministic, dependency-free provider for tests.

    NOT a real sparse model: each vector is plain term-frequency counts,
    hashed into a small fixed vocabulary. This exists so hybrid-search
    logic (Qdrant sparse-vector storage, RRF fusion) can be tested without
    downloading FlagEmbedding's BGE-M3 model.
    """

    def __init__(self, vocab_size: int = 2000) -> None:
        self._vocab_size = vocab_size
        # Recorded for tests that need to assert on what text was actually
        # sent to embed_documents_sparse — not used by the embedding itself.
        self.last_embedded_texts: list[str] = []

    def embed_documents_sparse(self, texts: list[str]) -> list[SparseVector]:
        self.last_embedded_texts = list(texts)
        return [self._embed_one(text) for text in texts]

    def embed_query_sparse(self, text: str) -> SparseVector:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> SparseVector:
        # zlib.crc32, not the builtin hash(): the latter is salted per
        # process (PYTHONHASHSEED), which would make the same word map to a
        # different bucket on every test run.
        counts: dict[int, float] = {}
        for word in text.lower().split():
            index = zlib.crc32(word.encode("utf-8")) % self._vocab_size
            counts[index] = counts.get(index, 0.0) + 1.0
        indices = sorted(counts)
        return SparseVector(indices=indices, values=[counts[i] for i in indices])
