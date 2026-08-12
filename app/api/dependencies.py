"""FastAPI dependency factories.

Everything that loads a real model or opens a Qdrant client is cached
(`lru_cache`) so it is constructed once per process, never reloaded per
request. Tests override these with Fake providers via
`app.dependency_overrides` instead of exercising the real, heavy models.
Heavy, optional dependencies (sentence-transformers) are imported lazily
inside the factory functions that need them, so importing this module
never requires torch to be installed.
"""

from functools import lru_cache

from qdrant_client import QdrantClient

from app.core.config import Settings, get_settings
from app.domain.chunk import ChunkingConfig
from app.domain.llm import GenerationConfig
from app.domain.retrieval import RetrievalConfig
from app.repositories.qdrant_chunk_repository import QdrantChunkRepository
from app.services.chunking import ChunkingService
from app.services.document_pipeline import DocumentIngestionService
from app.services.embedding_provider import EmbeddingProvider
from app.services.file_validation import FileValidationConfig, FileValidator
from app.services.indexing import IndexingService
from app.services.llm_provider import LlmProvider
from app.services.lm_studio_provider import LMStudioProvider
from app.services.rag_answer import RagAnswerService
from app.services.reranker import Reranker
from app.services.retrieval import RetrievalService


def get_app_settings() -> Settings:
    return get_settings()


@lru_cache
def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    if settings.qdrant_storage_path:
        return QdrantClient(path=settings.qdrant_storage_path)
    # Without an explicit timeout, an unreachable Qdrant can hang a request
    # indefinitely instead of surfacing IndexingErrorCode.QDRANT_UNAVAILABLE.
    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        timeout=int(settings.qdrant_timeout_seconds),
    )


@lru_cache
def get_chunk_repository() -> QdrantChunkRepository:
    return QdrantChunkRepository(get_qdrant_client())


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    from app.services.sentence_transformer_provider import SentenceTransformerEmbeddingProvider

    settings = get_settings()
    return SentenceTransformerEmbeddingProvider(
        model_id=settings.embedding_model_id,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )


def get_file_validator() -> FileValidator:
    settings = get_settings()
    return FileValidator(FileValidationConfig(max_size_bytes=settings.max_upload_size_bytes))


def get_indexing_service() -> IndexingService:
    return IndexingService(get_embedding_provider(), get_chunk_repository())


def get_document_ingestion_service() -> DocumentIngestionService:
    settings = get_settings()
    return DocumentIngestionService(
        validator=get_file_validator(),
        embedding_provider=get_embedding_provider(),
        indexing_service=get_indexing_service(),
        chunking_config=ChunkingConfig(
            max_tokens=settings.chunking_max_tokens, overlap_tokens=settings.chunking_overlap_tokens
        ),
        chunking_service=ChunkingService(),
    )


@lru_cache
def get_reranker() -> Reranker | None:
    settings = get_settings()
    if not settings.enable_reranker:
        return None
    from app.services.cross_encoder_reranker import CrossEncoderReranker

    return CrossEncoderReranker(
        model_id=settings.reranker_model_id, device=settings.reranker_device
    )


def get_retrieval_service() -> RetrievalService:
    return RetrievalService(get_embedding_provider(), get_chunk_repository(), get_reranker())


def get_retrieval_config() -> RetrievalConfig:
    settings = get_settings()
    return RetrievalConfig(
        top_k=settings.retrieval_top_k,
        llm_top_k=settings.llm_context_top_k,
        score_threshold=settings.retrieval_score_threshold,
        max_overlap_ratio=settings.retrieval_max_overlap_ratio,
    )


@lru_cache
def _get_llm_provider_cached(model_id: str) -> LlmProvider:
    settings = get_settings()
    return LMStudioProvider(model_id=model_id, base_url=settings.llm_base_url)


def get_llm_provider(model_id: str | None = None) -> LlmProvider:
    settings = get_settings()
    return _get_llm_provider_cached(model_id or settings.llm_model_id)


def get_rag_answer_service(model_id: str | None = None) -> RagAnswerService:
    return RagAnswerService(get_llm_provider(model_id))


def get_default_rag_answer_service() -> RagAnswerService:
    """No-argument wrapper for use with FastAPI's Depends() — a dependency
    with a plain (non-FastAPI-marked) parameter would otherwise be treated
    as a query parameter on the endpoint, which is not what we want here
    since model_id comes from the request body instead."""
    return get_rag_answer_service()


def get_generation_config() -> GenerationConfig:
    settings = get_settings()
    return GenerationConfig(
        temperature=settings.llm_temperature,
        top_p=settings.llm_top_p,
        max_new_tokens=settings.llm_max_new_tokens,
        context_token_limit=settings.llm_context_token_limit,
        timeout_seconds=settings.llm_timeout_seconds,
    )
