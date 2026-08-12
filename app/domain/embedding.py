from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class EmbeddingModelInfo:
    model_id: str
    revision: str
    vector_dimension: int
    normalized: bool
    schema_version: str = "1"


class EmbeddingErrorCode(str, Enum):
    EMPTY_TEXT = "empty_text"
    MODEL_LOAD_FAILED = "model_load_failed"


class EmbeddingError(Exception):
    """Raised when text cannot be embedded (e.g. empty input, model load failure)."""

    def __init__(self, code: EmbeddingErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
