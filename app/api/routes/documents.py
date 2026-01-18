"""Document management API routes."""

import logging
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException, status

from app.config import get_settings
from app.models.schemas import (
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentDeleteResponse,
    DocumentInfo,
    ErrorResponse
)
from app.services.parser import parser
from app.services.chunker import chunker
from app.services.embeddings import embedding_service
from app.services.vectorstore import vectorstore_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file type or empty file"},
        413: {"model": ErrorResponse, "description": "File too large"},
        500: {"model": ErrorResponse, "description": "Processing error"}
    }
)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document (PDF or DOCX).

    The document will be:
    1. Parsed to extract text
    2. Chunked into ~1000 token segments with overlap
    3. Embedded using Voyage AI
    4. Stored in Qdrant vector database
    """
    settings = get_settings()

    # Validate file extension
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required"
        )

    extension = Path(file.filename).suffix.lower()
    if extension not in settings.allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {extension} not supported. Allowed: {settings.allowed_extensions}"
        )

    # Read file content
    content = await file.read()

    # Validate file size
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty"
        )

    max_size = settings.max_file_size_mb * 1024 * 1024
    if len(content) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {settings.max_file_size_mb}MB"
        )

    try:
        # Step 1: Parse document
        logger.info(f"Parsing document: {file.filename}")
        text = await parser.parse(content, file.filename)

        if not text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No text could be extracted from the document"
            )

        # Step 2: Chunk text
        logger.info("Chunking text")
        chunks = chunker.chunk_text(text)

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document produced no valid chunks"
            )

        # Step 3: Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        chunk_texts = [c.content for c in chunks]
        embeddings = await embedding_service.embed_documents(chunk_texts)

        # Step 4: Ensure collection exists
        await vectorstore_service.ensure_collection()

        # Step 5: Store in Qdrant
        logger.info("Storing chunks in vector database")
        document_id = await vectorstore_service.store_chunks(
            chunks=chunk_texts,
            embeddings=embeddings,
            filename=file.filename,
            file_type=extension
        )

        logger.info(f"Successfully processed document {document_id}")
        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            total_chunks=len(chunks),
            message="Document uploaded and processed successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}"
        )


@router.get(
    "",
    response_model=DocumentListResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Database error"}
    }
)
async def list_documents():
    """List all uploaded documents."""
    try:
        await vectorstore_service.ensure_collection()
        documents = await vectorstore_service.list_documents()

        return DocumentListResponse(
            documents=documents,
            total_count=len(documents)
        )

    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Document not found"},
        500: {"model": ErrorResponse, "description": "Database error"}
    }
)
async def delete_document(document_id: str):
    """Delete a document and all its chunks."""
    try:
        chunks_deleted = await vectorstore_service.delete_document(document_id)

        if chunks_deleted == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )

        return DocumentDeleteResponse(
            document_id=document_id,
            message="Document deleted successfully",
            chunks_deleted=chunks_deleted
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )
