"""Application configuration via environment variables."""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Qdrant Configuration
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY")
    collection_name: str = os.getenv("COLLECTION_NAME", "documents")

    # Voyage AI Configuration
    voyage_api_key: str = os.getenv("VOYAGE_API_KEY", "")
    voyage_embedding_model: str = "voyage-3"
    voyage_rerank_model: str = "rerank-2"

    # Google Gemini Configuration
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    gemini_model: str = "gemini-2.5-pro-preview-05-06"

    # RAG Configuration
    chunk_size: int = 1000  # tokens
    chunk_overlap: int = 100  # tokens
    search_top_k: int = 50  # candidates for reranking
    rerank_top_k: int = 12  # final chunks for LLM
    embedding_dimension: int = 1024  # voyage-3 dimension

    # Application Settings
    max_file_size_mb: int = 50
    allowed_extensions: list[str] = [".pdf", ".docx"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
