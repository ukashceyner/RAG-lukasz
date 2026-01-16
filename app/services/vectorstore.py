"""Qdrant vector store service."""

import logging
from datetime import datetime
from typing import Optional
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import get_settings
from app.models.schemas import DocumentMetadata, DocumentInfo

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Handles all Qdrant vector database operations."""

    def __init__(self):
        self.settings = get_settings()
        self._client: Optional[QdrantClient] = None

    @property
    def client(self) -> QdrantClient:
        """Lazy initialization of Qdrant client."""
        if self._client is None:
            self._client = QdrantClient(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key,
            )
        return self._client

    async def ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        try:
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.settings.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.settings.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.settings.embedding_dimension,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.settings.collection_name}")
            else:
                logger.debug(f"Collection exists: {self.settings.collection_name}")

        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise

    async def store_chunks(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        filename: str,
        file_type: str
    ) -> str:
        """
        Store document chunks with their embeddings.

        Args:
            chunks: List of text chunks
            embeddings: List of embedding vectors
            filename: Original filename
            file_type: File extension (.pdf, .docx)

        Returns:
            Generated document_id
        """
        document_id = str(uuid4())
        upload_date = datetime.utcnow()
        total_chunks = len(chunks)

        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            metadata = DocumentMetadata(
                document_id=document_id,
                filename=filename,
                chunk_index=i,
                total_chunks=total_chunks,
                upload_date=upload_date,
                file_type=file_type
            )

            point = models.PointStruct(
                id=str(uuid4()),
                vector=embedding,
                payload={
                    "content": chunk,
                    **metadata.model_dump(mode="json")
                }
            )
            points.append(point)

        # Batch upsert points
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.settings.collection_name,
                points=batch
            )
            logger.debug(f"Upserted batch {i // batch_size + 1}")

        logger.info(f"Stored {total_chunks} chunks for document {document_id}")
        return document_id

    async def search(
        self,
        query_embedding: list[float],
        top_k: Optional[int] = None
    ) -> list[dict]:
        """
        Search for similar chunks.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return

        Returns:
            List of matching chunks with metadata
        """
        top_k = top_k or self.settings.search_top_k

        try:
            results = self.client.search(
                collection_name=self.settings.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True
            )

            return [
                {
                    "content": r.payload.get("content", ""),
                    "document_id": r.payload.get("document_id", ""),
                    "filename": r.payload.get("filename", ""),
                    "chunk_index": r.payload.get("chunk_index", 0),
                    "score": r.score
                }
                for r in results
            ]

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    async def list_documents(self) -> list[DocumentInfo]:
        """List all unique documents in the collection."""
        try:
            # Scroll through all points and aggregate by document_id
            documents = {}
            offset = None

            while True:
                results, offset = self.client.scroll(
                    collection_name=self.settings.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True
                )

                for point in results:
                    doc_id = point.payload.get("document_id")
                    if doc_id and doc_id not in documents:
                        documents[doc_id] = DocumentInfo(
                            document_id=doc_id,
                            filename=point.payload.get("filename", "unknown"),
                            total_chunks=point.payload.get("total_chunks", 0),
                            upload_date=point.payload.get("upload_date", datetime.utcnow()),
                            file_type=point.payload.get("file_type", "unknown")
                        )

                if offset is None:
                    break

            return list(documents.values())

        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            raise

    async def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks for a document.

        Args:
            document_id: The document ID to delete

        Returns:
            Number of chunks deleted
        """
        try:
            # Count chunks before deletion
            count_result = self.client.count(
                collection_name=self.settings.collection_name,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id)
                        )
                    ]
                )
            )
            chunks_to_delete = count_result.count

            # Delete by filter
            self.client.delete(
                collection_name=self.settings.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=document_id)
                            )
                        ]
                    )
                )
            )

            logger.info(f"Deleted {chunks_to_delete} chunks for document {document_id}")
            return chunks_to_delete

        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            raise

    async def is_connected(self) -> bool:
        """Check if Qdrant is reachable."""
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False


# Singleton instance
vectorstore_service = VectorStoreService()
