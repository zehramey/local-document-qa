from dataclasses import dataclass

CHUNKING_STRATEGY_NAME = "structural-heading-paragraph-sentence-token-v1"
CHUNKING_VERSION = "1"


@dataclass(frozen=True)
class ChunkingConfig:
    max_tokens: int
    overlap_tokens: int
    min_tokens: int = 40
    strategy_name: str = CHUNKING_STRATEGY_NAME
    version: str = CHUNKING_VERSION

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens must not be negative")
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        if self.min_tokens < 0:
            raise ValueError("min_tokens must not be negative")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    filename: str
    page_start: int
    page_end: int
    chunk_index: int
    section_title: str | None
    token_count: int
    text: str
    chunking_strategy: str
    chunking_version: str
