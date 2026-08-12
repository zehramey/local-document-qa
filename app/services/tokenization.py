"""Model-independent tokenizer abstraction used by the chunking service.

Tokenizer choice and approach (documented per project requirements):

`ApproximateTokenizer` is a dependency-free, offline, deterministic tokenizer
based on Unicode word/punctuation splitting (regex `\\w+|[^\\w\\s]`). Each
word and each punctuation mark counts as one "token". It does **not**
reproduce the exact subword vocabulary of any specific embedding or LLM
model (e.g. BGE-M3, multilingual-E5, or a tiktoken/BPE tokenizer) — it exists
so chunking can operate on a stable, local, dependency-free notion of "token
count" before an embedding model has been selected (see Phase 4).

Token counts and chunk boundaries produced by `ApproximateTokenizer` are
directionally correct (more words -> more tokens) but are an approximation,
not an exact count against any real model's tokenizer. This is a design
choice, not a benchmarked claim of accuracy.

The `Tokenizer` protocol lets a real tokenizer (e.g. a HuggingFace
`AutoTokenizer` wrapping the chosen embedding model) be substituted later
without changing `ChunkingService` or any chunking logic.
"""

import re
from typing import Protocol

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ0-9\"'(])")


class Tokenizer(Protocol):
    def count_tokens(self, text: str) -> int: ...

    def token_boundaries(self, text: str) -> list[tuple[int, int]]:
        """Character (start, end) span of each token, in order."""
        ...


class ApproximateTokenizer:
    """Whitespace/punctuation-based approximate tokenizer. See module docstring."""

    def count_tokens(self, text: str) -> int:
        return len(_TOKEN_PATTERN.findall(text))

    def token_boundaries(self, text: str) -> list[tuple[int, int]]:
        return [match.span() for match in _TOKEN_PATTERN.finditer(text)]


def split_sentences(text: str) -> list[str]:
    """Splits on '.', '!' or '?' followed by whitespace and a capital/digit/quote.

    A simple, dependency-free heuristic — not a full sentence boundary
    disambiguation model, so it can mis-split on abbreviations.
    """
    return [sentence.strip() for sentence in _SENTENCE_BOUNDARY.split(text) if sentence.strip()]


def cut_text_at_token_limit(text: str, max_tokens: int, tokenizer: Tokenizer) -> tuple[str, str]:
    """Splits text so the first part has at most max_tokens tokens.

    Returns (head, remainder). If text already fits, remainder is "".
    """
    spans = tokenizer.token_boundaries(text)
    if len(spans) <= max_tokens or max_tokens <= 0:
        return text, ""
    cutoff = spans[max_tokens - 1][1]
    return text[:cutoff].rstrip(), text[cutoff:].lstrip()


def tail_by_token_count(text: str, token_count: int, tokenizer: Tokenizer) -> str:
    """Returns the trailing substring of text containing at most token_count tokens."""
    if token_count <= 0:
        return ""
    spans = tokenizer.token_boundaries(text)
    if len(spans) <= token_count:
        return text
    start = spans[-token_count][0]
    return text[start:]
