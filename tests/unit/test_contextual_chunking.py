from app.domain.llm import (
    GenerationConfig,
    GenerationMetrics,
    GenerationResult,
    LlmError,
    LlmErrorCode,
)
from app.services.contextual_chunking import LlmChunkContextGenerator
from app.services.tokenization import ApproximateTokenizer


class _RecordingLlmProvider:
    """Local stub (not the shared FakeLlmProvider) so the exact prompt sent
    can be inspected — needed to verify document-context truncation."""

    def __init__(self, response_text: str = "situating context") -> None:
        self._response_text = response_text
        self.last_prompt: str | None = None
        self.last_config: GenerationConfig | None = None

    @property
    def model_id(self) -> str:
        return "recording-llm"

    def health_check(self) -> bool:
        return True

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        self.last_prompt = prompt
        self.last_config = config
        return GenerationResult(
            text=self._response_text,
            metrics=GenerationMetrics(
                model_id=self.model_id,
                quantization=None,
                prompt_tokens=len(prompt.split()),
                output_tokens=len(self._response_text.split()),
                time_to_first_token_seconds=None,
                total_duration_seconds=0.01,
                tokens_per_second=None,
            ),
        )


def test_generate_returns_the_llm_response_stripped() -> None:
    llm = _RecordingLlmProvider(response_text="  This chunk covers pricing.  ")
    generator = LlmChunkContextGenerator(llm)

    context = generator.generate("some document text", "some chunk text")

    assert context == "This chunk covers pricing."


def test_generate_includes_chunk_and_document_text_in_the_prompt() -> None:
    llm = _RecordingLlmProvider()
    generator = LlmChunkContextGenerator(llm)

    generator.generate("THE_DOCUMENT_TEXT", "THE_CHUNK_TEXT")

    assert llm.last_prompt is not None
    assert "THE_DOCUMENT_TEXT" in llm.last_prompt
    assert "THE_CHUNK_TEXT" in llm.last_prompt


def test_generate_truncates_document_text_to_the_configured_token_budget() -> None:
    llm = _RecordingLlmProvider()
    tokenizer = ApproximateTokenizer()
    generator = LlmChunkContextGenerator(llm, doc_context_tokens=5, tokenizer=tokenizer)
    long_document = " ".join(f"word{i}" for i in range(50))

    generator.generate(long_document, "chunk text")

    assert llm.last_prompt is not None
    assert "word0" in llm.last_prompt
    assert "word49" not in llm.last_prompt


def test_generate_uses_a_small_bounded_max_new_tokens() -> None:
    llm = _RecordingLlmProvider()
    generator = LlmChunkContextGenerator(llm, max_new_tokens=42)

    generator.generate("document", "chunk")

    assert llm.last_config is not None
    assert llm.last_config.max_new_tokens == 42


def test_generate_returns_empty_string_on_llm_error_instead_of_raising() -> None:
    class _RaisingLlmProvider(_RecordingLlmProvider):
        def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
            raise LlmError(LlmErrorCode.SERVER_UNAVAILABLE, "down")

    generator = LlmChunkContextGenerator(_RaisingLlmProvider())

    assert generator.generate("document", "chunk") == ""
