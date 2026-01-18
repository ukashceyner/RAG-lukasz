"""Query/search API routes."""

import logging
import time

from fastapi import APIRouter, HTTPException, status

from app.config import get_settings
from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    SourceChunk,
    ErrorResponse
)
from app.services.embeddings import embedding_service
from app.services.reranker import reranker_service
from app.services.vectorstore import vectorstore_service
from app.services.llm import llm_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Query"])


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid query"},
        500: {"model": ErrorResponse, "description": "Processing error"}
    }
)
async def query(request: QueryRequest):
    """
    Ask a question and get an answer with source citations.

    The query pipeline:
    1. Embed the question using Voyage AI
    2. Search Qdrant for top 50 candidates (cosine similarity)
    3. Rerank results using Voyage rerank-2.5
    4. Send top 12 chunks to Gemini for answer generation
    5. Return answer with source citations
    """
    settings = get_settings()
    start_time = time.time()

    try:
        # Step 1: Embed the question
        logger.info(f"Processing query: {request.question[:100]}...")
        query_embedding = await embedding_service.embed_query(request.question)

        # Step 2: Search for candidates
        logger.info("Searching for candidates")
        await vectorstore_service.ensure_collection()
        search_results = await vectorstore_service.search(
            query_embedding=query_embedding,
            top_k=settings.search_top_k
        )

        if not search_results:
            return QueryResponse(
                answer="I couldn't find any relevant documents to answer your question. Please upload some documents first.",
                sources=[],
                query=request.question,
                processing_time_ms=(time.time() - start_time) * 1000
            )

        # Step 3: Rerank results
        logger.info(f"Reranking {len(search_results)} candidates")
        documents = [r["content"] for r in search_results]
        top_k = request.top_k or settings.rerank_top_k
        reranked = await reranker_service.rerank(
            query=request.question,
            documents=documents,
            top_k=top_k
        )

        # Build source chunks from reranked results
        sources = []
        for result in reranked:
            original = search_results[result.index]
            sources.append(SourceChunk(
                document_id=original["document_id"],
                filename=original["filename"],
                chunk_index=original["chunk_index"],
                content=result.document,
                relevance_score=result.relevance_score
            ))

        # Step 4: Generate answer with LLM
        logger.info(f"Generating answer from {len(sources)} sources")
        answer = await llm_service.generate_answer(
            question=request.question,
            sources=sources
        )

        processing_time = (time.time() - start_time) * 1000
        logger.info(f"Query processed in {processing_time:.2f}ms")

        return QueryResponse(
            answer=answer,
            sources=sources,
            query=request.question,
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {str(e)}"
        )
