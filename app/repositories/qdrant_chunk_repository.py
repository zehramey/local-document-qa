"""Qdrant-backed storage for chunk vectors and their metadata.

Collection naming ties each collection to a specific embedding model and
schema version (`{model_id}__v{schema_version}`), so different embedding
models/dimensions are never mixed into the same collection. `ensure_collection`
additionally guards this at runtime: if a collection with that name already
exists but was created with a different vector size, it raises rather than
silently corrupting the collection.

Point IDs must be valid Qdrant IDs (UUID or unsigned int); a chunk's
`chunk_id` (a SHA-256 hex digest) is deterministically mapped to a UUID
derived from its first 32 hex characters, so the same chunk always maps to
the same point and re-indexing safely overwrites rather than duplicates.
"""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app.domain.chunk import Chunk
from app.domain.embedding import EmbeddingModelInfo
from app.domain.indexing import IndexingError, IndexingErrorCode

_QDRANT_CONNECTION_ERRORS = (ResponseHandlingException, UnexpectedResponse)


def collection_name_for(model_info: EmbeddingModelInfo) -> str:
    sanitized_model_id = model_info.model_id.replace("/", "__")
    return f"{sanitized_model_id}__v{model_info.schema_version}"


def _chunk_point_id(chunk_id: str) -> str:
    return str(uuid.UUID(chunk_id[:32]))


class QdrantChunkRepository:
    def __init__(self, client: QdrantClient) -> None:
        self._client = client

    def ensure_collection(self, collection_name: str, vector_dimension: int) -> None:
        try:
            exists = self._client.collection_exists(collection_name)
        except _QDRANT_CONNECTION_ERRORS as exc:
            raise IndexingError(
                IndexingErrorCode.QDRANT_UNAVAILABLE, f"Qdrant'a erişilemedi: {exc}"
            ) from exc

        if not exists:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=qm.VectorParams(size=vector_dimension, distance=qm.Distance.COSINE),
            )
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name="document_id",
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )
            return

        info = self._client.get_collection(collection_name)
        vectors_config = info.config.params.vectors
        if not isinstance(vectors_config, qm.VectorParams):
            raise IndexingError(
                IndexingErrorCode.VECTOR_DIMENSION_MISMATCH,
                f"Collection '{collection_name}' bilinmeyen/named-vector bir yapılandırmaya "
                "sahip; bu repository yalnızca tek, isimsiz vektör alanını destekler.",
            )
        existing_dimension = vectors_config.size
        if existing_dimension != vector_dimension:
            raise IndexingError(
                IndexingErrorCode.VECTOR_DIMENSION_MISMATCH,
                f"Collection '{collection_name}' {existing_dimension} boyutlu vektörler "
                f"içeriyor, ancak {vector_dimension} boyutlu vektör verildi. Farklı "
                "embedding modelleri/boyutları aynı collection'da karıştırılamaz.",
            )

    def upsert_chunks(
        self, collection_name: str, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks ve vectors sayısı eşleşmiyor")
        if not chunks:
            return

        points = [
            qm.PointStruct(
                id=_chunk_point_id(chunk.chunk_id),
                vector=vector,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "filename": chunk.filename,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "chunk_index": chunk.chunk_index,
                    "section_title": chunk.section_title,
                    "token_count": chunk.token_count,
                    "text": chunk.text,
                    "chunking_strategy": chunk.chunking_strategy,
                    "chunking_version": chunk.chunking_version,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        try:
            self._client.upsert(collection_name=collection_name, points=points)
        except _QDRANT_CONNECTION_ERRORS as exc:
            raise IndexingError(
                IndexingErrorCode.QDRANT_UNAVAILABLE, f"Qdrant'a erişilemedi: {exc}"
            ) from exc
        except Exception as exc:
            raise IndexingError(
                IndexingErrorCode.UPSERT_FAILED, f"Chunk upsert başarısız: {exc}"
            ) from exc

    def find_chunk_ids_by_document(self, collection_name: str, document_id: str) -> set[str]:
        try:
            records, _ = self._client.scroll(
                collection_name=collection_name,
                scroll_filter=qm.Filter(
                    must=[
                        qm.FieldCondition(key="document_id", match=qm.MatchValue(value=document_id))
                    ]
                ),
                with_payload=True,
                limit=10_000,
            )
        except _QDRANT_CONNECTION_ERRORS as exc:
            raise IndexingError(
                IndexingErrorCode.QDRANT_UNAVAILABLE, f"Qdrant'a erişilemedi: {exc}"
            ) from exc

        return {record.payload["chunk_id"] for record in records if record.payload}

    def delete_by_document_id(self, collection_name: str, document_id: str) -> None:
        try:
            self._client.delete(
                collection_name=collection_name,
                points_selector=qm.FilterSelector(
                    filter=qm.Filter(
                        must=[
                            qm.FieldCondition(
                                key="document_id", match=qm.MatchValue(value=document_id)
                            )
                        ]
                    )
                ),
            )
        except _QDRANT_CONNECTION_ERRORS as exc:
            raise IndexingError(
                IndexingErrorCode.QDRANT_UNAVAILABLE, f"Qdrant'a erişilemedi: {exc}"
            ) from exc
        except Exception as exc:
            raise IndexingError(IndexingErrorCode.DELETE_FAILED, f"Silme başarısız: {exc}") from exc

    def delete_chunk_ids(self, collection_name: str, chunk_ids: set[str]) -> None:
        if not chunk_ids:
            return
        point_ids: list[int | str] = [_chunk_point_id(chunk_id) for chunk_id in chunk_ids]
        try:
            self._client.delete(
                collection_name=collection_name,
                points_selector=qm.PointIdsList(points=point_ids),
            )
        except _QDRANT_CONNECTION_ERRORS as exc:
            raise IndexingError(
                IndexingErrorCode.QDRANT_UNAVAILABLE, f"Qdrant'a erişilemedi: {exc}"
            ) from exc
        except Exception as exc:
            raise IndexingError(IndexingErrorCode.DELETE_FAILED, f"Silme başarısız: {exc}") from exc

    def count(self, collection_name: str) -> int:
        try:
            return self._client.count(collection_name).count
        except _QDRANT_CONNECTION_ERRORS as exc:
            raise IndexingError(
                IndexingErrorCode.QDRANT_UNAVAILABLE, f"Qdrant'a erişilemedi: {exc}"
            ) from exc
