from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "local-document-qa"
    environment: str = "development"

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    max_upload_size_bytes: int = 20 * 1024 * 1024

    embedding_model_id: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32

    retrieval_top_k: int = 10
    llm_context_top_k: int = 5
    retrieval_score_threshold: float | None = None
    retrieval_max_overlap_ratio: float = 0.8

    reranker_model_id: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cpu"


@lru_cache
def get_settings() -> Settings:
    return Settings()
