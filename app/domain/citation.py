from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Citation:
    """Shown to the user; built entirely from backend/Qdrant metadata."""

    chunk_id: str
    document_id: str
    filename: str
    page_start: int
    page_end: int


class CitationErrorCode(str, Enum):
    UNKNOWN_CHUNK_ID = "unknown_chunk_id"
    WRONG_DOCUMENT = "wrong_document"


class CitationError(Exception):
    """Raised when an LLM cites a chunk_id that wasn't actually retrieved
    for this query, or that belongs to a different document."""

    def __init__(self, code: CitationErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
