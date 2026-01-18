"""FastAPI application initialization."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import documents, query, health
from app.services.vectorstore import vectorstore_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events."""
    # Startup
    logger.info("Starting RAG API...")
    try:
        await vectorstore_service.ensure_collection()
        logger.info("Qdrant collection initialized")
    except Exception as e:
        logger.warning(f"Could not initialize Qdrant collection: {e}")

    yield

    # Shutdown
    logger.info("Shutting down RAG API...")


# Create FastAPI application
app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation API for document Q&A",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(documents.router)
app.include_router(query.router)
app.include_router(health.router)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }
