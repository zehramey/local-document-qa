from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 10
    llm_top_k: int = 5
    # Raw cosine-similarity cutoff chosen by the caller/ops — NOT a calibrated
    # accuracy or confidence percentage. None disables threshold filtering.
    score_threshold: float | None = None
    # Chunks whose word-overlap (Jaccard) with an already-kept, higher-scored
    # chunk meets or exceeds this ratio are dropped as near-duplicates.
    max_overlap_ratio: float = 0.8

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.llm_top_k <= 0:
            raise ValueError("llm_top_k must be positive")
        if not 0.0 < self.max_overlap_ratio <= 1.0:
            raise ValueError("max_overlap_ratio must be in (0, 1]")


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    filename: str
    page_start: int
    page_end: int
    text: str
    retrieval_score: float
    reranker_score: float | None
    final_rank: int


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    document_id: str
    chunks: list[RetrievedChunk]
    total_candidates: int

    @property
    def has_results(self) -> bool:
        return len(self.chunks) > 0
