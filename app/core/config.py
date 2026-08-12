from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "local-document-qa"
    environment: str = "development"

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_timeout_seconds: float = 5.0
    # If set, connect to Qdrant in embedded/on-disk mode (no server
    # process) instead of host:port — useful for local development
    # without Docker. Ignored by docker-compose deployments, which don't
    # set this and use the real qdrant service instead.
    qdrant_storage_path: str | None = None

    max_upload_size_bytes: int = 20 * 1024 * 1024

    chunking_max_tokens: int = 500
    chunking_overlap_tokens: int = 75

    embedding_model_id: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32

    retrieval_top_k: int = 10
    llm_context_top_k: int = 5
    retrieval_score_threshold: float | None = None
    retrieval_max_overlap_ratio: float = 0.8

    reranker_model_id: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cpu"
    # Off by default: loading the reranker alongside the embedding model
    # and the LLM was tight on this development machine's RAM (see Faz 6
    # hardware notes, ~3.6GB free). Safe to enable where RAM allows.
    enable_reranker: bool = False

    llm_base_url: str = "http://localhost:1234"
    llm_model_id: str = "qwen/qwen3.5-4b"
    llm_temperature: float = 0.0
    llm_top_p: float = 1.0
    llm_max_new_tokens: int = 512
    llm_context_token_limit: int = 3000
    llm_timeout_seconds: float = 60.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
