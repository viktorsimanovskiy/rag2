# ============================================================
# File: app/services/ingestion/document_ingestion_service.py
# Purpose:
#   Load documents into the knowledge base.
#
# Responsibilities:
#   - split documents into chunks
#   - generate embeddings
#   - persist chunks and embeddings
#
# Non-responsibilities:
#   - does NOT parse PDF/DOCX
#   - does NOT manage vector search
# ============================================================

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.ingestion.document_chunker import (
    DocumentChunker,
    DocumentChunk,
)

from app.services.retrieval.query_embedding_service import (
    QueryEmbeddingService,
)

logger = logging.getLogger(__name__)


# ============================================================
# DTO
# ============================================================

@dataclass(slots=True)
class IngestionResult:
    document_id: UUID
    chunks_created: int


# ============================================================
# Service
# ============================================================

class DocumentIngestionService:
    """
    High-level ingestion service for loading documents
    into the knowledge base.
    """

    def __init__(
        self,
        session: AsyncSession,
        *,
        chunker: DocumentChunker,
        embedding_service: QueryEmbeddingService,
    ) -> None:

        self.session = session
        self.chunker = chunker
        self.embedding_service = embedding_service

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    async def ingest_document(
        self,
        *,
        document_id: UUID,
        text: str,
    ) -> IngestionResult:

        chunks = self.chunker.chunk_document(
            document_id=document_id,
            text=text,
        )

        embeddings = await self.embedding_service.embed_batch(
            [c.text for c in chunks]
        )

        await self._persist_chunks(
            chunks=chunks,
            embeddings=embeddings,
        )

        logger.info(
            "Document ingested",
            extra={
                "document_id": str(document_id),
                "chunks": len(chunks),
            },
        )

        return IngestionResult(
            document_id=document_id,
            chunks_created=len(chunks),
        )

    # --------------------------------------------------------
    # Internal
    # --------------------------------------------------------

    async def _persist_chunks(
        self,
        *,
        chunks: List[DocumentChunk],
        embeddings,
    ) -> None:

        for chunk, emb in zip(chunks, embeddings):

            block_id = uuid4()

            await self.session.execute(
                """
                INSERT INTO document_block (
                    block_id,
                    document_id,
                    chunk_index,
                    text
                )
                VALUES (
                    :block_id,
                    :document_id,
                    :chunk_index,
                    :text
                )
                """,
                {
                    "block_id": block_id,
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                },
            )

            await self.session.execute(
                """
                INSERT INTO document_block_embedding (
                    block_id,
                    document_id,
                    embedding,
                    embedding_model
                )
                VALUES (
                    :block_id,
                    :document_id,
                    :embedding,
                    :model
                )
                """,
                {
                    "block_id": block_id,
                    "document_id": chunk.document_id,
                    "embedding": emb.embedding,
                    "model": emb.model_name,
                },
            )

        await self.session.commit()