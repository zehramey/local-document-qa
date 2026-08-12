"""Validates LLM-cited chunk_ids and builds the citation shown to the user.

The LLM may only cite chunk_ids that were actually retrieved for the
current query — any other id (fabricated, or real but belonging to a
different document) is rejected. Page numbers are always read from the
retrieved chunk's own metadata (ultimately sourced from Qdrant); the
model's own claim about which page it's citing is never accepted as input
here, so there is no way for a wrong model-stated page number to reach the
user.
"""

from app.domain.citation import Citation, CitationError, CitationErrorCode
from app.domain.retrieval import RetrievedChunk


class CitationValidator:
    def __init__(self, retrieved_chunks: list[RetrievedChunk], expected_document_id: str) -> None:
        self._chunks_by_id = {chunk.chunk_id: chunk for chunk in retrieved_chunks}
        self._expected_document_id = expected_document_id

    def validate(self, claimed_chunk_id: str) -> Citation:
        chunk = self._chunks_by_id.get(claimed_chunk_id)
        if chunk is None:
            raise CitationError(
                CitationErrorCode.UNKNOWN_CHUNK_ID,
                f"'{claimed_chunk_id}' bu sorgu için retrieve edilen chunk'lar arasında "
                "değil; kaynak olarak kabul edilmedi.",
            )
        if chunk.document_id != self._expected_document_id:
            raise CitationError(
                CitationErrorCode.WRONG_DOCUMENT,
                f"'{claimed_chunk_id}' beklenen dokümana (document_id="
                f"'{self._expected_document_id}') ait değil.",
            )
        return Citation(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            filename=chunk.filename,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
        )
