from dataclasses import dataclass
from enum import Enum

from app.domain.citation import Citation
from app.domain.llm import GenerationMetrics


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    answerable: bool
    citations: list[Citation]
    rejected_citation_ids: list[str]
    metrics: GenerationMetrics | None


class RagAnswerErrorCode(str, Enum):
    MALFORMED_JSON = "malformed_json"
    MISSING_REQUIRED_FIELD = "missing_required_field"


class RagAnswerError(Exception):
    """Raised when the LLM's output can't be parsed into a structured answer."""

    def __init__(self, code: RagAnswerErrorCode, message: str, raw_text: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.raw_text = raw_text
