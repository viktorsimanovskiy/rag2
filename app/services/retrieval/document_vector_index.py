# ============================================================
# File: app/services/retrieval/document_vector_index.py
# Purpose:
#   Vector index access layer for document block embeddings.
#
# Responsibilities:
#   - perform semantic similarity search over document blocks
#   - return top-k candidate blocks with similarity scores
#
# Non-responsibilities:
#   - does NOT orchestrate retrieval
#   - does NOT format evidence
#   - does NOT generate answers
# ============================================================

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# ============================================================
# DTOs
# ============================================================

@dataclass(slots=True)
class VectorSearchResult:
    """
    Result of semantic search over document blocks.
    """
    block_id: UUID
    document_id: UUID
    similarity_score: float


# ============================================================
# Service
# ============================================================

class DocumentVectorIndex:
    """
    Access layer for vector search over document_block embeddings.

    Assumes schema:

        document_block_embedding
            block_id UUID
            document_id UUID
            embedding vector
            embedding_model TEXT

    With pgvector index:

        CREATE INDEX idx_doc_block_embedding
        ON document_block_embedding
        USING hnsw (embedding vector_cosine_ops);
    """

    def __init__(
        self,
        session: AsyncSession,
        *,
        embedding_model: str,
        top_k: int = 20,
    ) -> None:
        self.session = session
        self.embedding_model = embedding_model
        self.top_k = top_k

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    async def semantic_search(
        self,
        query_embedding: List[float],
        *,
        top_k: int | None = None,
    ) -> List[VectorSearchResult]:
        """
        Perform semantic similarity search using pgvector.

        Returns top-k document blocks ordered by similarity.
        """

        limit = top_k or self.top_k

        stmt = text(
            """
            SELECT
                block_id,
                document_id,
                1 - (embedding <=> :query_embedding) AS similarity
            FROM document_block_embedding
            WHERE embedding_model = :embedding_model
            ORDER BY embedding <=> :query_embedding
            LIMIT :limit
            """
        )

        result = await self.session.execute(
            stmt,
            {
                "query_embedding": query_embedding,
                "embedding_model": self.embedding_model,
                "limit": limit,
            },
        )

        rows = result.fetchall()

        candidates: List[VectorSearchResult] = []

        for row in rows:
            candidates.append(
                VectorSearchResult(
                    block_id=row.block_id,
                    document_id=row.document_id,
                    similarity_score=float(row.similarity),
                )
            )

        logger.debug(
            "Vector search executed",
            extra={
                "embedding_model": self.embedding_model,
                "result_count": len(candidates),
                "limit": limit,
            },
        )

        return candidates

    # --------------------------------------------------------
    # Batch search
    # --------------------------------------------------------

    async def batch_search(
        self,
        embeddings: List[List[float]],
        *,
        top_k: int | None = None,
    ) -> List[List[VectorSearchResult]]:
        """
        Perform semantic search for multiple query embeddings.
        """

        results: List[List[VectorSearchResult]] = []

        for embedding in embeddings:
            results.append(
                await self.semantic_search(
                    embedding,
                    top_k=top_k,
                )
            )

        return results