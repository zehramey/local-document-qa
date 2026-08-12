from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class RerankerModelInfo:
    model_id: str
    revision: str


class RerankerErrorCode(str, Enum):
    DEVICE_UNAVAILABLE = "device_unavailable"
    MODEL_LOAD_FAILED = "model_load_failed"


class RerankerError(Exception):
    """Raised when a reranker model's device/hardware requirement can't be
    met, or when the model fails to load."""

    def __init__(self, code: RerankerErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
