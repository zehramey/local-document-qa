"""Dense retrieval, independent of any LLM.

Pipeline: embed the query -> cosine search in Qdrant, scoped to exactly one
document_id (a user can never search outside the document they selected) ->
drop near-duplicate/heavily overlapping chunks -> apply the configured score
threshold (a raw cosine cutoff, not a calibrated accuracy value) -> rerank if
a Reranker is configured (the dense retrieval_score is always preserved
alongside reranker_score, never overwritten) -> assign final_rank.

`RetrievalResult.has_results` is the explicit "not enough results" signal —
callers must check it rather than assuming a non-empty answer is possible.
"""

from dataclasses import replace

from qdrant_client.http import models as qm

from app.domain.retrieval import RetrievalConfig, RetrievalResult, RetrievedChunk
from app.repositories.qdrant_chunk_repository import QdrantChunkRepository, collection_name_for
from app.services.embedding_provider import EmbeddingProvider
from app.services.reranker import Reranker


def _word_overlap_ratio(text_a: str, text_b: str) -> float:
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    union = words_a | words_b
    if not union:
        return 0.0
    return len(words_a & words_b) / len(union)


class RetrievalService:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        repository: QdrantChunkRepository,
        reranker: Reranker | None = None,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._repository = repository
        self._reranker = reranker

    def retrieve(self, query: str, document_id: str, config: RetrievalConfig) -> RetrievalResult:
        collection_name = collection_name_for(self._embedding_provider.model_info)
        query_vector = self._embedding_provider.embed_query(query)

        raw_results = self._repository.search_similar(
            collection_name, query_vector, document_id, config.top_k
        )

        chunks = [self._to_retrieved_chunk(point) for point in raw_results]
        chunks = self._apply_threshold(chunks, config.score_threshold)
        chunks = self._deduplicate_overlapping(chunks, config.max_overlap_ratio)

        if self._reranker is not None and chunks:
            chunks = self._reranker.rerank(query, chunks)
        else:
            chunks = sorted(chunks, key=lambda chunk: chunk.retrieval_score, reverse=True)

        chunks = self._assign_final_rank(chunks)
        return RetrievalResult(
            query=query, document_id=document_id, chunks=chunks, total_candidates=len(raw_results)
        )

    def context_for_llm(
        self, result: RetrievalResult, config: RetrievalConfig
    ) -> list[RetrievedChunk]:
        return result.chunks[: config.llm_top_k]

    @staticmethod
    def _to_retrieved_chunk(point: qm.ScoredPoint) -> RetrievedChunk:
        payload = point.payload or {}
        return RetrievedChunk(
            chunk_id=payload["chunk_id"],
            document_id=payload["document_id"],
            filename=payload["filename"],
            page_start=payload["page_start"],
            page_end=payload["page_end"],
            text=payload["text"],
            retrieval_score=point.score,
            reranker_score=None,
            final_rank=0,
        )

    @staticmethod
    def _apply_threshold(
        chunks: list[RetrievedChunk], score_threshold: float | None
    ) -> list[RetrievedChunk]:
        if score_threshold is None:
            return chunks
        return [chunk for chunk in chunks if chunk.retrieval_score >= score_threshold]

    @staticmethod
    def _deduplicate_overlapping(
        chunks: list[RetrievedChunk], max_overlap_ratio: float
    ) -> list[RetrievedChunk]:
        kept: list[RetrievedChunk] = []
        for chunk in sorted(chunks, key=lambda c: c.retrieval_score, reverse=True):
            if any(
                _word_overlap_ratio(chunk.text, kept_chunk.text) >= max_overlap_ratio
                for kept_chunk in kept
            ):
                continue
            kept.append(chunk)
        return kept

    @staticmethod
    def _assign_final_rank(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return [replace(chunk, final_rank=rank) for rank, chunk in enumerate(chunks, start=1)]
