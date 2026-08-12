from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "local-document-qa"
    environment: str = "development"

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333


@lru_cache
def get_settings() -> Settings:
    return Settings()
