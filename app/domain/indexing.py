from enum import Enum


class IndexingErrorCode(str, Enum):
    QDRANT_UNAVAILABLE = "qdrant_unavailable"
    VECTOR_DIMENSION_MISMATCH = "vector_dimension_mismatch"
    UPSERT_FAILED = "upsert_failed"
    DELETE_FAILED = "delete_failed"


class IndexingError(Exception):
    """Raised when Qdrant indexing (collection setup, upsert, delete) fails."""

    def __init__(self, code: IndexingErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
