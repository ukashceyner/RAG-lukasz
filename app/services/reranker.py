"""Voyage AI reranking service."""

import logging
from dataclasses import dataclass
from typing import Optional

import voyageai

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Result from reranking operation."""
    index: int
    relevance_score: float
    document: str


class RerankerService:
    """Handles document reranking using Voyage AI."""

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

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: Optional[int] = None
    ) -> list[RerankResult]:
        """
        Rerank documents based on relevance to the query.

        Args:
            query: The search query
            documents: List of document texts to rerank
            top_k: Number of top results to return (default from settings)

        Returns:
            List of RerankResult objects sorted by relevance
        """
        if not documents:
            return []

        top_k = top_k or self.settings.rerank_top_k

        try:
            result = self.client.rerank(
                query=query,
                documents=documents,
                model=self.settings.voyage_rerank_model,
                top_k=min(top_k, len(documents))
            )

            rerank_results = [
                RerankResult(
                    index=r.index,
                    relevance_score=r.relevance_score,
                    document=documents[r.index]
                )
                for r in result.results
            ]

            logger.info(
                f"Reranked {len(documents)} documents to top {len(rerank_results)} "
                f"using {self.settings.voyage_rerank_model}"
            )
            return rerank_results

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            raise

    def is_configured(self) -> bool:
        """Check if the reranker service is properly configured."""
        return bool(self.settings.voyage_api_key)


# Singleton instance
reranker_service = RerankerService()
