import pytest
from app.domain.citation import CitationError, CitationErrorCode
from app.domain.retrieval import RetrievedChunk
from app.services.citation import CitationValidator


def _make_retrieved_chunk(
    chunk_id: str,
    document_id: str,
    filename: str = "report.pdf",
    page_start: int = 3,
    page_end: int = 3,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        filename=filename,
        page_start=page_start,
        page_end=page_end,
        text="some retrieved text",
        retrieval_score=0.9,
        reranker_score=None,
        final_rank=1,
    )


def test_valid_citation_is_built_from_retrieved_chunk_metadata() -> None:
    chunk = _make_retrieved_chunk("a" * 64, "doc-1", page_start=7, page_end=8)
    validator = CitationValidator([chunk], expected_document_id="doc-1")

    citation = validator.validate("a" * 64)

    assert citation.chunk_id == "a" * 64
    assert citation.document_id == "doc-1"
    assert citation.page_start == 7
    assert citation.page_end == 8


def test_fabricated_chunk_id_is_rejected() -> None:
    chunk = _make_retrieved_chunk("a" * 64, "doc-1")
    validator = CitationValidator([chunk], expected_document_id="doc-1")

    with pytest.raises(CitationError) as exc_info:
        validator.validate("f" * 64)

    assert exc_info.value.code == CitationErrorCode.UNKNOWN_CHUNK_ID


def test_citation_page_ignores_any_page_the_caller_claims() -> None:
    # The chunk's real page (per Qdrant metadata) is 7-8. Simulate an LLM
    # that (incorrectly) claims the answer is on page 999: our API has no
    # parameter for that claim at all, so it cannot influence the result.
    chunk = _make_retrieved_chunk("a" * 64, "doc-1", page_start=7, page_end=8)
    validator = CitationValidator([chunk], expected_document_id="doc-1")
    llm_claimed_page = 999  # noqa: F841 (deliberately unused — must not affect the citation)

    citation = validator.validate("a" * 64)

    assert citation.page_start == 7
    assert citation.page_end == 8


def test_citation_belonging_to_another_document_is_rejected() -> None:
    chunk_from_other_doc = _make_retrieved_chunk("a" * 64, "doc-2")
    validator = CitationValidator([chunk_from_other_doc], expected_document_id="doc-1")

    with pytest.raises(CitationError) as exc_info:
        validator.validate("a" * 64)

    assert exc_info.value.code == CitationErrorCode.WRONG_DOCUMENT
