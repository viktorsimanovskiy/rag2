# ============================================================
# File: app/services/retrieval/query_embedding_service.py
# Purpose:
#   Generate embeddings for user queries.
#
# Responsibilities:
#   - convert question text to vector embedding
#   - handle retries / batching
#   - normalize query input
#
# Non-responsibilities:
#   - does NOT perform vector search
#   - does NOT orchestrate retrieval
# ============================================================

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import List, Protocol

logger = logging.getLogger(__name__)


# ============================================================
# Exceptions
# ============================================================

class EmbeddingServiceError(Exception):
    """Raised when query embedding generation fails."""


# ============================================================
# DTO
# ============================================================

@dataclass(slots=True)
class QueryEmbeddingResult:
    embedding: List[float]
    model_name: str


# ============================================================
# Protocol
# ============================================================

class EmbeddingProviderProtocol(Protocol):
    async def embed(self, texts: List[str]) -> List[List[float]]:
        ...


# ============================================================
# Service
# ============================================================

class QueryEmbeddingService:
    """
    Generate embeddings for user questions.
    """

    def __init__(
        self,
        provider: EmbeddingProviderProtocol,
        *,
        model_name: str,
        max_text_length: int = 2000,
        retry_attempts: int = 3,
        retry_delay: float = 0.5,
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.max_text_length = max_text_length
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    async def embed_query(
        self,
        question_text: str,
    ) -> QueryEmbeddingResult:
        normalized = self._normalize_text(question_text)
        embeddings = await self._embed_with_retry([normalized])

        return QueryEmbeddingResult(
            embedding=embeddings[0],
            model_name=self.model_name,
        )

    async def embed_batch(
        self,
        texts: List[str],
    ) -> List[QueryEmbeddingResult]:
        normalized = [self._normalize_text(t) for t in texts]
        vectors = await self._embed_with_retry(normalized)

        return [
            QueryEmbeddingResult(
                embedding=vector,
                model_name=self.model_name,
            )
            for vector in vectors
        ]

    # --------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------

    def _normalize_text(self, text: str) -> str:
        if not isinstance(text, str):
            raise EmbeddingServiceError("Query text must be a string.")

        normalized = " ".join(text.strip().split())

        if not normalized:
            raise EmbeddingServiceError("Query text must not be empty.")

        if len(normalized) > self.max_text_length:
            normalized = normalized[: self.max_text_length]

        return normalized

    async def _embed_with_retry(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        attempt = 0

        while True:
            try:
                vectors = await self.provider.embed(texts)

                if not vectors:
                    raise EmbeddingServiceError(
                        "Embedding provider returned empty result."
                    )

                if len(vectors) != len(texts):
                    raise EmbeddingServiceError(
                        "Embedding provider returned unexpected vector count."
                    )

                return vectors

            except Exception as exc:
                attempt += 1

                if attempt >= self.retry_attempts:
                    logger.exception("Embedding failed after retries")
                    raise EmbeddingServiceError(str(exc)) from exc

                logger.warning(
                    "Embedding retry",
                    extra={
                        "attempt": attempt,
                        "max_attempts": self.retry_attempts,
                    },
                )

                await asyncio.sleep(self.retry_delay)