"""Voyage AI embedding service."""

import logging
from typing import Optional

import voyageai

from app.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Handles text embedding using Voyage AI."""

    def __init__(self):
        self.settings = get_settings()
        self._client: Optional[voyageai.Client] = None

    @property
    def client(self) -> voyageai.Client:
        """Lazy initialization of Voyage AI client."""
        if self._client is None:
            if not self.settings.voyage_api_key:
                raise ValueError("VOYAGE_API_KEY is not configured")
            self._client = voyageai.Client(api_key=self.settings.voyage_api_key)
        return self._client

    async def embed_texts(
        self,
        texts: list[str],
        input_type: str = "document"
    ) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            input_type: Either "document" for indexing or "query" for search

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        try:
            result = self.client.embed(
                texts=texts,
                model=self.settings.voyage_embedding_model,
                input_type=input_type
            )

            embeddings = result.embeddings
            logger.info(
                f"Generated {len(embeddings)} embeddings using {self.settings.voyage_embedding_model}"
            )
            return embeddings

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    async def embed_query(self, query: str) -> list[float]:
        """
        Generate embedding for a search query.

        Args:
            query: The search query text

        Returns:
            Single embedding vector
        """
        embeddings = await self.embed_texts([query], input_type="query")
        return embeddings[0]

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """
        Generate embeddings for document chunks.

        Args:
            documents: List of document text chunks

        Returns:
            List of embedding vectors
        """
        # Voyage AI has a batch limit, process in batches of 128
        batch_size = 128
        all_embeddings = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_embeddings = await self.embed_texts(batch, input_type="document")
            all_embeddings.extend(batch_embeddings)
            logger.debug(f"Processed embedding batch {i // batch_size + 1}")

        return all_embeddings

    def is_configured(self) -> bool:
        """Check if the embedding service is properly configured."""
        return bool(self.settings.voyage_api_key)


# Singleton instance
embedding_service = EmbeddingService()
