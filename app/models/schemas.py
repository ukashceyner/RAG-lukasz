"""Pydantic models for request/response schemas."""

from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional


# ============== Document Models ==============

class DocumentMetadata(BaseModel):
    """Metadata stored with each document chunk."""
    document_id: str
    filename: str
    chunk_index: int
    total_chunks: int
    upload_date: datetime
    file_type: str


class DocumentUploadResponse(BaseModel):
    """Response after successful document upload."""
    document_id: str
    filename: str
    total_chunks: int
    message: str


class DocumentInfo(BaseModel):
    """Information about a stored document."""
    document_id: str
    filename: str
    total_chunks: int
    upload_date: datetime
    file_type: str


class DocumentListResponse(BaseModel):
    """Response for listing all documents."""
    documents: list[DocumentInfo]
    total_count: int


class DocumentDeleteResponse(BaseModel):
    """Response after document deletion."""
    document_id: str
    message: str
    chunks_deleted: int


# ============== Query Models ==============

class QueryRequest(BaseModel):
    """Request body for querying the RAG system."""
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class SourceChunk(BaseModel):
    """A source chunk returned with the answer."""
    document_id: str
    filename: str
    chunk_index: int
    content: str
    relevance_score: float


class QueryResponse(BaseModel):
    """Response containing the generated answer and sources."""
    answer: str
    sources: list[SourceChunk]
    query: str
    processing_time_ms: float


# ============== Health Models ==============

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    qdrant_connected: bool
    voyage_configured: bool
    gemini_configured: bool
    version: str = "1.0.0"


# ============== Error Models ==============

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
    status_code: int
