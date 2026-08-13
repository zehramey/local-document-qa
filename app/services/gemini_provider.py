"""LLM provider backed by the Google Gemini API.

Non-streaming (unlike LMStudioProvider): the RAG flow only needs the final
JSON text, so there's no product need for token-by-token output here, and
skipping SSE parsing keeps this small. `time_to_first_token_seconds` is
therefore always None (see app.domain.llm.GenerationMetrics).
"""

import time

import httpx

from app.domain.llm import GenerationConfig, GenerationMetrics, GenerationResult, LlmError, LlmErrorCode

_DEFAULT_GENERATION_EXTRA: dict[str, object] = {}
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
        try:
            response = httpx.post(
                f"{self._base_url}/v1beta/models/{self._model_id}:generateContent",
                params={"key": self._api_key},
                json=payload,
                timeout=config.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LlmError(LlmErrorCode.TIMEOUT, f"LLM request timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise LlmError(
                LlmErrorCode.SERVER_UNAVAILABLE, f"Could not reach the LLM server: {exc}"
            ) from exc
        total_duration = time.monotonic() - start

        body = response.json()
        try:
            candidate = body["candidates"][0]
            text = "".join(part.get("text", "") for part in candidate["content"]["parts"])
        except (KeyError, IndexError) as exc:
            raise LlmError(
                LlmErrorCode.MALFORMED_RESPONSE, f"Gemini response is in an unexpected format: {body}"
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
