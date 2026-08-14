"""LLM provider backed by the Google Gemini API.

Non-streaming (unlike LMStudioProvider): the RAG flow only needs the final
JSON text, so there's no product need for token-by-token output here, and
skipping SSE parsing keeps this small. `time_to_first_token_seconds` is
therefore always None (see app.domain.llm.GenerationMetrics).
"""

import random
import time

import httpx

from app.domain.llm import (
    GenerationConfig,
    GenerationMetrics,
    GenerationResult,
    LlmError,
    LlmErrorCode,
)

_DEFAULT_GENERATION_EXTRA: dict[str, object] = {}
# 429 (free-tier rate limit) and 5xx (transient upstream outage) are the
# error classes actually observed against gemini-flash-latest during
# benchmarking (see benchmark/results/results_20260814_081138.json — most
# rows for this model failed with exactly these codes). Both are worth a
# retry; other 4xx codes (400 bad request, 404 unknown model, ...) are not,
# since retrying an invalid request just repeats the same failure.
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 5
_BACKOFF_BASE_SECONDS = 1.0
_BACKOFF_MAX_SECONDS = 20.0
# Same failure mode we hit with qwen3.5 in LM Studio: current Gemini models
# (tested against gemini-flash-latest -> gemini-3.6-flash) think by default
# and that counts against maxOutputTokens, so a small budget can be spent
# entirely on thoughtsTokenCount with no answer left. Unlike qwen3.5,
# thinking can't be turned off here — {"thinkingConfig": {"thinkingBudget":
# 0}} gets rejected with 400 INVALID_ARGUMENT. The fix is a generous
# max_new_tokens (see LLM_MAX_NEW_TOKENS in .env) rather than disabling
# thinking.


class GeminiProvider:
    def __init__(
        self,
        model_id: str,
        api_key: str,
        base_url: str = "https://generativelanguage.googleapis.com",
        generation_extra: dict[str, object] | None = None,
    ) -> None:
        self._model_id = model_id
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._generation_extra = (
            _DEFAULT_GENERATION_EXTRA if generation_extra is None else generation_extra
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    def health_check(self) -> bool:
        try:
            response = httpx.get(
                f"{self._base_url}/v1beta/models/{self._model_id}",
                params={"key": self._api_key},
                timeout=5.0,
            )
            response.raise_for_status()
        except httpx.HTTPError:
            return False
        return True

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        payload: dict[str, object] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": config.temperature,
                "topP": config.top_p,
                "maxOutputTokens": config.max_new_tokens,
                **self._generation_extra,
            },
        }

        start = time.monotonic()
        response = self._post_with_retry(payload, config.timeout_seconds)
        total_duration = time.monotonic() - start

        body = response.json()
        try:
            candidate = body["candidates"][0]
            text = "".join(part.get("text", "") for part in candidate["content"]["parts"])
        except (KeyError, IndexError) as exc:
            raise LlmError(
                LlmErrorCode.MALFORMED_RESPONSE,
                f"Gemini response is in an unexpected format: {body}",
            ) from exc

        usage = body.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", 0)
        output_tokens = usage.get("candidatesTokenCount", 0)
        tokens_per_second = (
            output_tokens / total_duration if output_tokens and total_duration > 0 else None
        )

        return GenerationResult(
            text=text,
            metrics=GenerationMetrics(
                model_id=self._model_id,
                quantization=None,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                time_to_first_token_seconds=None,
                total_duration_seconds=total_duration,
                tokens_per_second=tokens_per_second,
            ),
        )

    def _post_with_retry(
        self, payload: dict[str, object], timeout_seconds: float
    ) -> httpx.Response:
        last_exc: httpx.HTTPError | None = None
        for attempt in range(_MAX_ATTEMPTS):
            if attempt > 0:
                time.sleep(self._backoff_seconds(attempt))
            try:
                response = httpx.post(
                    f"{self._base_url}/v1beta/models/{self._model_id}:generateContent",
                    params={"key": self._api_key},
                    json=payload,
                    timeout=timeout_seconds,
                )
                response.raise_for_status()
                return response
            except httpx.TimeoutException as exc:
                raise LlmError(LlmErrorCode.TIMEOUT, f"LLM request timed out: {exc}") from exc
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if exc.response.status_code not in _RETRYABLE_STATUS_CODES:
                    raise LlmError(
                        LlmErrorCode.SERVER_UNAVAILABLE, f"Could not reach the LLM server: {exc}"
                    ) from exc
            except httpx.HTTPError as exc:
                last_exc = exc

        raise LlmError(
            LlmErrorCode.SERVER_UNAVAILABLE,
            f"Could not reach the LLM server after {_MAX_ATTEMPTS} attempts: {last_exc}",
        ) from last_exc

    @staticmethod
    def _backoff_seconds(attempt: int) -> float:
        exponential = min(_BACKOFF_BASE_SECONDS * (2.0 ** (attempt - 1)), _BACKOFF_MAX_SECONDS)
        return exponential + random.uniform(0, 1.0)
