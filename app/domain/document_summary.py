from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentSummary:
    document_id: str
    filename: str
    chunk_count: int
    page_count: int


@dataclass(frozen=True)
class IngestionResult:
    document_id: str
    filename: str
    chunk_count: int
    collection_name: str
    stale_chunks_removed: int
