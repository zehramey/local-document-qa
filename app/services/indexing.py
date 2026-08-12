"""Orchestrates embedding + Qdrant upsert for a document's chunks.

Re-indexing order favors never losing data over avoiding brief duplication:
new chunks are upserted first, and only after that succeeds are stale
chunks — chunk_ids that belonged to this document before but aren't part
of the new set (e.g. the chunking config changed) — deleted. If the upsert
step fails partway through, the previous version's chunks are left intact
instead of being deleted first and lost.
"""

from dataclasses import dataclass

from app.domain.chunk import Chunk
from app.domain.document import Document
from app.repositories.qdrant_chunk_repository import QdrantChunkRepository, collection_name_for
from app.services.embedding_provider import EmbeddingProvider


@dataclass(frozen=True)
class IndexingResult:
    collection_name: str
    chunks_indexed: int
    stale_chunks_removed: int
    total_points_for_document: int


class IndexingService:
    def __init__(self, provider: EmbeddingProvider, repository: QdrantChunkRepository) -> None:
        self._provider = provider
        self._repository = repository

    def index_document(self, document: Document, chunks: list[Chunk]) -> IndexingResult:
        collection_name = collection_name_for(self._provider.model_info)
        vector_dimension = self._provider.model_info.vector_dimension
        self._repository.ensure_collection(collection_name, vector_dimension)

        if not chunks:
            self._repository.delete_by_document_id(collection_name, document.document_id)
            return IndexingResult(collection_name, 0, 0, 0)

        vectors = self._provider.embed_documents([chunk.text for chunk in chunks])
        self._repository.upsert_chunks(collection_name, chunks, vectors)

        new_chunk_ids = {chunk.chunk_id for chunk in chunks}
        existing_chunk_ids = self._repository.find_chunk_ids_by_document(
            collection_name, document.document_id
        )
        stale_chunk_ids = existing_chunk_ids - new_chunk_ids
        if stale_chunk_ids:
            self._repository.delete_chunk_ids(collection_name, stale_chunk_ids)

        total_points = len(
            self._repository.find_chunk_ids_by_document(collection_name, document.document_id)
        )
        return IndexingResult(
            collection_name=collection_name,
            chunks_indexed=len(chunks),
            stale_chunks_removed=len(stale_chunk_ids),
            total_points_for_document=total_points,
        )

    def delete_document(self, document_id: str) -> None:
        collection_name = collection_name_for(self._provider.model_info)
        self._repository.delete_by_document_id(collection_name, document_id)
