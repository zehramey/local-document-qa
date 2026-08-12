"""Verifies the real multilingual reranker (BAAI/bge-reranker-v2-m3).

Excluded from the default test run (see pyproject `addopts`) because it
downloads a real model (first run only, then cached) and requires
`pip install -e ".[embeddings]"`.

Run explicitly with: pytest -m integration
"""

import pytest

from app.domain.retrieval import RetrievedChunk
from app.services.cross_encoder_reranker import CrossEncoderReranker
from app.services.reranker_models import BGE_RERANKER_V2_M3

pytestmark = pytest.mark.integration


def _make_chunk(chunk_id: str, text: str, retrieval_score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        filename="report.pdf",
        page_start=1,
        page_end=1,
        text=text,
        retrieval_score=retrieval_score,
        reranker_score=None,
        final_rank=0,
    )


def test_real_reranker_scores_relevant_chunk_higher() -> None:
    reranker = CrossEncoderReranker(model_id=BGE_RERANKER_V2_M3.model_id, device="cpu")
    assert reranker.model_info.model_id == BGE_RERANKER_V2_M3.model_id

    on_topic = _make_chunk(
        "a" * 64,
        "Kedilerin ortalama yaşam süresi 12 ile 18 yıl arasındadır.",
        retrieval_score=0.5,
    )
    off_topic = _make_chunk(
        "b" * 64,
        "Borsa endeksleri bu hafta hafif bir düşüş gösterdi.",
        retrieval_score=0.6,
    )

    reranked = reranker.rerank("Kediler ne kadar yaşar?", [off_topic, on_topic])

    assert reranked[0].chunk_id == on_topic.chunk_id
    assert reranked[0].reranker_score is not None
    assert reranked[1].reranker_score is not None
    assert reranked[0].reranker_score > reranked[1].reranker_score
    # Dense baseline scores must survive reranking untouched.
    assert {chunk.chunk_id: chunk.retrieval_score for chunk in reranked} == {
        on_topic.chunk_id: 0.5,
        off_topic.chunk_id: 0.6,
    }
