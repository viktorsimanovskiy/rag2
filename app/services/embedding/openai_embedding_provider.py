# ============================================================
# File: app/services/embedding/openai_embedding_provider.py
# Purpose:
#   OpenAI-based embedding provider for query/document embeddings.
#
# Responsibilities:
#   - call OpenAI embeddings API
#   - batch input texts
#   - normalize output vectors
#   - retry transient failures conservatively
#
# Non-responsibilities:
#   - does NOT decide retrieval strategy
#   - does NOT store vectors in DB
#   - does NOT chunk documents
# ============================================================

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from openai import AsyncOpenAI

from app.config.constants import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
)

logger = logging.getLogger(__name__)


# ============================================================
# Exceptions
# ============================================================

class OpenAIEmbeddingProviderError(Exception):
    """Base embedding provider error."""


class OpenAIEmbeddingValidationError(OpenAIEmbeddingProviderError):
    """Raised when provider input/output is invalid."""


# ============================================================
# Config
# ============================================================

@dataclass(slots=True, frozen=True)
class OpenAIEmbeddingProviderConfig:
    """
    Provider config for OpenAI embeddings API.

    batch_size:
        How many texts to send in one embeddings request.

    max_retries:
        Conservative retry count for transient failures.

    retry_delay_seconds:
        Base retry delay. Backoff is linear for predictability.

    dimensions:
        Optional output dimensions. Supported by text-embedding-3 models.
        Keep it aligned with app/config/constants.py.
    """

    model_name: str = EMBEDDING_MODEL_NAME
    dimensions: int = EMBEDDING_DIMENSION
    batch_size: int = 32
    max_retries: int = 3
    retry_delay_seconds: float = 0.75


# ============================================================
# Provider
# ============================================================

class OpenAIEmbeddingProvider:
    """
    OpenAI embeddings provider.

    Contract:
        async embed(texts: list[str]) -> list[list[float]]

    This provider is intentionally infrastructure-only and should be used
    by higher-level services such as QueryEmbeddingService.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        config: Optional[OpenAIEmbeddingProviderConfig] = None,
    ) -> None:
        self.client = client
        self.config = config or OpenAIEmbeddingProviderConfig()

        if self.config.batch_size < 1:
            raise OpenAIEmbeddingValidationError("batch_size must be >= 1.")

        if self.config.max_retries < 1:
            raise OpenAIEmbeddingValidationError("max_retries must be >= 1.")

        if self.config.dimensions < 1:
            raise OpenAIEmbeddingValidationError("dimensions must be >= 1.")

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    async def embed(
        self,
        texts: Sequence[str],
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Guarantees:
        - preserves input ordering
        - validates empty input
        - batches requests
        - validates output vector dimensions
        """
        prepared = self._prepare_texts(texts)

        if not prepared:
            raise OpenAIEmbeddingValidationError("texts must not be empty.")

        batches = self._chunk(prepared, self.config.batch_size)
        all_vectors: List[List[float]] = []

        for batch_index, batch in enumerate(batches, start=1):
            vectors = await self._embed_batch_with_retry(
                batch,
                batch_index=batch_index,
            )
            all_vectors.extend(vectors)

        if len(all_vectors) != len(prepared):
            raise OpenAIEmbeddingValidationError(
                "Embeddings response size does not match request size."
            )

        return all_vectors

    async def embed_one(self, text: str) -> List[float]:
        """
        Convenience helper for a single text.
        """
        vectors = await self.embed([text])
        return vectors[0]

    # --------------------------------------------------------
    # Internal API calls
    # --------------------------------------------------------

    async def _embed_batch_with_retry(
        self,
        batch: list[str],
        *,
        batch_index: int,
    ) -> List[List[float]]:
        last_error: Optional[Exception] = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = await self.client.embeddings.create(
                    model=self.config.model_name,
                    input=batch,
                    dimensions=self.config.dimensions,
                )

                vectors = [item.embedding for item in response.data]
                self._validate_vectors(vectors)

                logger.debug(
                    "Embedding batch created",
                    extra={
                        "model_name": self.config.model_name,
                        "batch_index": batch_index,
                        "batch_size": len(batch),
                        "dimensions": self.config.dimensions,
                    },
                )
                return vectors

            except Exception as exc:
                last_error = exc

                logger.warning(
                    "Embedding batch failed",
                    extra={
                        "model_name": self.config.model_name,
                        "batch_index": batch_index,
                        "attempt": attempt,
                        "max_retries": self.config.max_retries,
                    },
                )

                if attempt >= self.config.max_retries:
                    break

                await asyncio.sleep(self.config.retry_delay_seconds * attempt)

        logger.exception(
            "Embedding batch failed after retries",
            extra={
                "model_name": self.config.model_name,
                "batch_index": batch_index,
                "max_retries": self.config.max_retries,
            },
        )
        raise OpenAIEmbeddingProviderError(
            f"Failed to create embeddings after {self.config.max_retries} attempts."
        ) from last_error

    # --------------------------------------------------------
    # Validation / normalization
    # --------------------------------------------------------

    def _prepare_texts(
        self,
        texts: Sequence[str],
    ) -> list[str]:
        prepared: list[str] = []

        for index, value in enumerate(texts):
            if not isinstance(value, str):
                raise OpenAIEmbeddingValidationError(
                    f"text at index={index} must be a string."
                )

            normalized = " ".join(value.strip().split())
            if not normalized:
                raise OpenAIEmbeddingValidationError(
                    f"text at index={index} must not be empty."
                )

            prepared.append(normalized)

        return prepared

    def _validate_vectors(
        self,
        vectors: Iterable[Sequence[float]],
    ) -> None:
        for index, vector in enumerate(vectors):
            if not vector:
                raise OpenAIEmbeddingValidationError(
                    f"embedding at index={index} is empty."
                )

            if len(vector) != self.config.dimensions:
                raise OpenAIEmbeddingValidationError(
                    f"embedding at index={index} has dimension={len(vector)}, "
                    f"expected={self.config.dimensions}."
                )

    def _chunk(
        self,
        items: Sequence[str],
        size: int,
    ) -> list[list[str]]:
        return [
            list(items[i : i + size])
            for i in range(0, len(items), size)
        ]