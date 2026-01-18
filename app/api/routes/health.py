"""Health check API route."""

import logging

from fastapi import APIRouter

from app.models.schemas import HealthResponse
from app.services.embeddings import embedding_service
from app.services.vectorstore import vectorstore_service
from app.services.llm import llm_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse
)
async def health_check():
    """
    Check the health status of the API and its dependencies.

    Returns connectivity status for:
    - Qdrant vector database
    - Voyage AI (embedding service configuration)
    - Google Gemini (LLM service configuration)
    """
    # Check Qdrant connectivity
    qdrant_connected = await vectorstore_service.is_connected()

    # Check service configurations
    voyage_configured = embedding_service.is_configured()
    gemini_configured = llm_service.is_configured()

    # Determine overall status
    all_healthy = qdrant_connected and voyage_configured and gemini_configured
    status = "healthy" if all_healthy else "degraded"

    if not qdrant_connected:
        logger.warning("Health check: Qdrant not connected")
    if not voyage_configured:
        logger.warning("Health check: Voyage AI not configured")
    if not gemini_configured:
        logger.warning("Health check: Gemini not configured")

    return HealthResponse(
        status=status,
        qdrant_connected=qdrant_connected,
        voyage_configured=voyage_configured,
        gemini_configured=gemini_configured
    )
