from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 512
    context_token_limit: int = 3000
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.context_token_limit <= 0:
            raise ValueError("context_token_limit must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


@dataclass(frozen=True)
class GenerationMetrics:
    model_id: str
    quantization: str | None
    prompt_tokens: int
    output_tokens: int
    time_to_first_token_seconds: float | None
    total_duration_seconds: float
    tokens_per_second: float | None
    # Not measured: the LLM runs in a separate local server process (e.g.
    # LM Studio); reading its RSS/VRAM would require OS-level process
    # inspection tied to a specific running instance, out of scope here.
    ram_usage_mb: float | None = None


@dataclass(frozen=True)
class GenerationResult:
    text: str
    metrics: GenerationMetrics


@dataclass(frozen=True)
class AvailableModel:
    model_id: str
    model_type: str
    quantization: str | None
    state: str | None


class LlmErrorCode(str, Enum):
    SERVER_UNAVAILABLE = "server_unavailable"
    TIMEOUT = "timeout"
    MALFORMED_RESPONSE = "malformed_response"


class LlmError(Exception):
    """Raised when the local LLM backend can't be reached, times out, or
    returns something the provider itself can't make sense of."""

    def __init__(self, code: LlmErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
