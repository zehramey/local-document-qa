"""Real LLM provider backed by LM Studio's local OpenAI-compatible server.

Uses streaming so time-to-first-token can actually be measured from the
wall clock, rather than left unmeasured. `ram_usage_mb` stays None (see
app.domain.llm.GenerationMetrics) — LM Studio runs the model in its own
process, and this provider only talks to it over HTTP.
"""

import json
import time

import httpx

from app.domain.llm import (
    AvailableModel,
    GenerationConfig,
    GenerationMetrics,
    GenerationResult,
    LlmError,
    LlmErrorCode,
)

_DEFAULT_EXTRA_BODY: dict[str, object] = {"reasoning_effort": "none"}
# Found by testing this exact model against this exact server: without
# disabling "thinking", it can spend its entire max_new_tokens budget on
# reasoning_content and never emit an actual answer. Not a general claim
# about every model LM Studio can serve.


def list_available_models(
    base_url: str = "http://localhost:1234", timeout: float = 5.0
) -> list[AvailableModel]:
    """Lists models LM Studio currently knows about (loaded or not), for GET /models."""
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/api/v0/models", timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise LlmError(
            LlmErrorCode.SERVER_UNAVAILABLE, f"LM Studio model listesi alınamadı: {exc}"
        ) from exc

    return [
        AvailableModel(
            model_id=str(item["id"]),
            model_type=str(item.get("type", "unknown")),
            quantization=item.get("quantization"),
            state=item.get("state"),
        )
        for item in response.json().get("data", [])
    ]


class LMStudioProvider:
    def __init__(
        self,
        model_id: str,
        base_url: str = "http://localhost:1234",
        extra_body: dict[str, object] | None = None,
    ) -> None:
        self._model_id = model_id
        self._base_url = base_url.rstrip("/")
        self._extra_body = _DEFAULT_EXTRA_BODY if extra_body is None else extra_body
        self._quantization: str | None = None
        self._quantization_fetched = False

    @property
    def model_id(self) -> str:
        return self._model_id

    def health_check(self) -> bool:
        try:
            response = httpx.get(f"{self._base_url}/v1/models", timeout=5.0)
            response.raise_for_status()
        except httpx.HTTPError:
            return False
        model_ids = {item["id"] for item in response.json().get("data", [])}
        return self._model_id in model_ids

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        if not self._quantization_fetched:
            self._quantization = self._fetch_quantization()
            self._quantization_fetched = True

        payload: dict[str, object] = {
            "model": self._model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_new_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            **self._extra_body,
        }

        start = time.monotonic()
        first_token_at: float | None = None
        content_parts: list[str] = []
        usage: dict[str, int] = {}

        try:
            with httpx.stream(
                "POST",
                f"{self._base_url}/v1/chat/completions",
                json=payload,
                timeout=config.timeout_seconds,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[len("data: ") :]
                    if data.strip() == "[DONE]":
                        break
                    chunk = json.loads(data)
                    choices = chunk.get("choices") or []
                    if choices:
                        piece = (choices[0].get("delta") or {}).get("content")
                        if piece:
                            if first_token_at is None:
                                first_token_at = time.monotonic()
                            content_parts.append(piece)
                    if chunk.get("usage"):
                        usage = chunk["usage"]
        except httpx.TimeoutException as exc:
            raise LlmError(LlmErrorCode.TIMEOUT, f"LLM isteği zaman aşımına uğradı: {exc}") from exc
        except httpx.HTTPError as exc:
            raise LlmError(
                LlmErrorCode.SERVER_UNAVAILABLE, f"LLM sunucusuna erişilemedi: {exc}"
            ) from exc

        total_duration = time.monotonic() - start
        text = "".join(content_parts)
        prompt_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        time_to_first_token = first_token_at - start if first_token_at is not None else None
        tokens_per_second = (
            output_tokens / total_duration if output_tokens and total_duration > 0 else None
        )

        return GenerationResult(
            text=text,
            metrics=GenerationMetrics(
                model_id=self._model_id,
                quantization=self._quantization,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                time_to_first_token_seconds=time_to_first_token,
                total_duration_seconds=total_duration,
                tokens_per_second=tokens_per_second,
            ),
        )

    def _fetch_quantization(self) -> str | None:
        try:
            response = httpx.get(f"{self._base_url}/api/v0/models", timeout=5.0)
            response.raise_for_status()
        except httpx.HTTPError:
            return None
        for item in response.json().get("data", []):
            if item.get("id") == self._model_id:
                quantization = item.get("quantization")
                return str(quantization) if quantization is not None else None
        return None
