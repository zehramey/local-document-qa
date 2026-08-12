"""Model-independent local LLM provider abstraction."""

from typing import Protocol

from app.domain.llm import GenerationConfig, GenerationMetrics, GenerationResult


class LlmProvider(Protocol):
    @property
    def model_id(self) -> str: ...

    def health_check(self) -> bool:
        """True if the backend is reachable and the model is loaded/ready."""
        ...

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult: ...


class FakeLlmProvider:
    """Deterministic, dependency-free provider for tests. NOT a real LLM."""

    def __init__(
        self,
        response_text: str = "",
        model_id: str = "fake-llm",
        healthy: bool = True,
        raises: Exception | None = None,
    ) -> None:
        self._response_text = response_text
        self._model_id = model_id
        self._healthy = healthy
        self._raises = raises
        self.generate_call_count = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    def health_check(self) -> bool:
        return self._healthy

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        self.generate_call_count += 1
        if self._raises is not None:
            raise self._raises

        output_tokens = len(self._response_text.split())
        return GenerationResult(
            text=self._response_text,
            metrics=GenerationMetrics(
                model_id=self._model_id,
                quantization="fake",
                prompt_tokens=len(prompt.split()),
                output_tokens=output_tokens,
                time_to_first_token_seconds=0.01,
                total_duration_seconds=0.02,
                tokens_per_second=(output_tokens / 0.02) if output_tokens else None,
            ),
        )
