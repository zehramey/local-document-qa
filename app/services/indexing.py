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
from app.services.contextual_chunking import ChunkContextGenerator
from app.services.embedding_provider import EmbeddingProvider
from app.services.sparse_embedding_provider import SparseEmbeddingProvider


@dataclass(frozen=True)
class IndexingResult:
    collection_name: str
    chunks_indexed: int
    stale_chunks_removed: int
    total_points_for_document: int


class IndexingService:
    def __init__(
        self,
        provider: EmbeddingProvider,
        repository: QdrantChunkRepository,
        context_generator: ChunkContextGenerator | None = None,
        sparse_provider: SparseEmbeddingProvider | None = None,
    ) -> None:
        self._provider = provider
        self._repository = repository
        self._context_generator = context_generator
        self._sparse_provider = sparse_provider

    def index_document(
        self, document: Document, chunks: list[Chunk], document_text: str | None = None
    ) -> IndexingResult:
        collection_name = self._collection_name()

        if self._sparse_provider is not None:
            self._repository.ensure_hybrid_collection(
                collection_name, self._provider.model_info.vector_dimension
            )
        else:
            self._repository.ensure_collection(
                collection_name, self._provider.model_info.vector_dimension
            )

        if not chunks:
            self._repository.delete_by_document_id(collection_name, document.document_id)
            return IndexingResult(collection_name, 0, 0, 0)

        embedding_texts = self._embedding_texts(chunks, document_text)
        if self._sparse_provider is not None:
            dense_vectors = self._provider.embed_documents(embedding_texts)
            sparse_vectors = self._sparse_provider.embed_documents_sparse(embedding_texts)
            self._repository.upsert_chunks_hybrid(
                collection_name, chunks, dense_vectors, sparse_vectors
            )
        else:
            vectors = self._provider.embed_documents(embedding_texts)
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
        self._repository.delete_by_document_id(self._collection_name(), document_id)

    def _collection_name(self) -> str:
        is_hybrid = self._sparse_provider is not None
        return collection_name_for(self._provider.model_info, hybrid=is_hybrid)

    def _embedding_texts(self, chunks: list[Chunk], document_text: str | None) -> list[str]:
        """What actually gets embedded — chunk.text itself stays untouched
        (see module docstring in contextual_chunking.py: the generated
        context is an embedding-time-only augmentation, never persisted or
        shown to a user)."""
        if self._context_generator is None or not document_text:
            return [chunk.text for chunk in chunks]
        texts = []
        for chunk in chunks:
            context = self._context_generator.generate(document_text, chunk.text)
            texts.append(f"{context}\n\n{chunk.text}" if context else chunk.text)
        return texts
