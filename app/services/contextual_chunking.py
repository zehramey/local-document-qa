"""Generates a short situating context for each chunk before embedding.

Anthropic's "contextual retrieval" technique
(https://www.anthropic.com/engineering/contextual-retrieval): prepending a
short LLM-written context to a chunk before it is embedded measurably
improves retrieval, because the chunk's vector then encodes where it sits
in the document instead of just its own isolated sentence(s).

This is strictly an embedding-time augmentation. The generated context is
never stored as Chunk.text and never shown to the user or the answering
LLM (see IndexingService._embedding_texts) — only the vector changes.

Anthropic's own reported numbers assume prompt caching, so the whole
document isn't re-sent (and re-billed/re-computed) for every chunk. This
project has no prompt-caching path for a local LLM server, so
`doc_context_tokens` bounds how much of the document is actually sent per
chunk (a leading excerpt, not the full text) — otherwise a large document
with many chunks would mean one full-document-sized LLM call per chunk,
with unbounded latency/cost. This is a deliberate, bounded approximation
of the original technique, not a full reproduction of it.
"""

from typing import Protocol

from app.domain.llm import GenerationConfig, LlmError
from app.services.llm_provider import LlmProvider
from app.services.tokenization import ApproximateTokenizer, Tokenizer, cut_text_at_token_limit

_PROMPT_TEMPLATE = """<document>
{doc_context}
</document>

Here is a chunk from that document that we want to situate within the whole document:
<chunk>
{chunk_text}
</chunk>

Give a short, succinct context (1-2 sentences) to situate this chunk within \
the overall document, for the purpose of improving search retrieval of the \
chunk. Answer with only the context itself, in the same language as the \
document, and nothing else."""


class ChunkContextGenerator(Protocol):
    def generate(self, document_text: str, chunk_text: str) -> str:
        """Returns a short situating context for chunk_text, or "" if none
        could be generated. Callers must treat "" as "skip augmentation for
        this chunk", never as an error — a context-generation failure must
        not block indexing."""
        ...


class LlmChunkContextGenerator:
    def __init__(
        self,
        llm: LlmProvider,
        doc_context_tokens: int = 2000,
        max_new_tokens: int = 100,
        timeout_seconds: float = 30.0,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self._llm = llm
        self._doc_context_tokens = doc_context_tokens
        self._max_new_tokens = max_new_tokens
        self._timeout_seconds = timeout_seconds
        self._tokenizer = tokenizer or ApproximateTokenizer()

    def generate(self, document_text: str, chunk_text: str) -> str:
        doc_context, _ = cut_text_at_token_limit(
            document_text, self._doc_context_tokens, self._tokenizer
        )
        prompt = _PROMPT_TEMPLATE.format(doc_context=doc_context, chunk_text=chunk_text)
        config = GenerationConfig(
            temperature=0.0,
            max_new_tokens=self._max_new_tokens,
            timeout_seconds=self._timeout_seconds,
        )
        try:
            result = self._llm.generate(prompt, config)
        except LlmError:
            return ""
        return result.text.strip()


class FakeChunkContextGenerator:
    """Deterministic, dependency-free generator for tests. NOT a real LLM call."""

    def __init__(self, context: str = "fake context") -> None:
        self._context = context
        self.calls: list[tuple[str, str]] = []

    def generate(self, document_text: str, chunk_text: str) -> str:
        self.calls.append((document_text, chunk_text))
        return self._context
