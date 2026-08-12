"""Model-independent reranker abstraction.

A reranker re-scores and reorders the dense-retrieval candidates for a
query. It never overwrites `retrieval_score` — the dense baseline is always
preserved in `reranker_score` being a *separate* field, so switching
rerankers on/off never loses the underlying dense ranking.
"""

from dataclasses import replace
from typing import Protocol

from app.domain.reranking import RerankerModelInfo
from app.domain.retrieval import RetrievedChunk


class Reranker(Protocol):
    @property
    def model_info(self) -> RerankerModelInfo: ...

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Returns chunks with `reranker_score` set, sorted best-first."""
        ...


class FakeReranker:
    """Deterministic, dependency-free reranker for tests.

    NOT a real cross-encoder: the "score" is the Jaccard word-overlap
    between the query and each chunk's text. This exists so retrieval
    ordering/reranking logic can be tested without downloading a real
    multilingual reranker model.
    """

    def __init__(self, model_id: str = "fake-reranker") -> None:
        self._info = RerankerModelInfo(model_id=model_id, revision="test")

    @property
    def model_info(self) -> RerankerModelInfo:
        return self._info

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        query_words = set(query.lower().split())
        rescored = [
            replace(chunk, reranker_score=self._score(query_words, chunk)) for chunk in chunks
        ]
        return sorted(rescored, key=lambda chunk: chunk.reranker_score or 0.0, reverse=True)

    @staticmethod
    def _score(query_words: set[str], chunk: RetrievedChunk) -> float:
        chunk_words = set(chunk.text.lower().split())
        union = query_words | chunk_words
        if not union:
            return 0.0
        return len(query_words & chunk_words) / len(union)
