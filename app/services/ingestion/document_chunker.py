# ============================================================
# File: app/services/ingestion/document_chunker.py
# Purpose:
#   Split normative documents into semantic blocks
#   suitable for vector embeddings.
#
# Responsibilities:
#   - normalize document text
#   - split text into semantic chunks
#   - apply overlap
#   - preserve paragraph boundaries
#
# Non-responsibilities:
#   - does NOT create embeddings
#   - does NOT write to DB
# ============================================================

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List
from uuid import UUID

from app.config.constants import (
    DOCUMENT_BLOCK_MAX_CHARS,
    DOCUMENT_BLOCK_OVERLAP,
)

logger = logging.getLogger(__name__)


# ============================================================
# DTO
# ============================================================

@dataclass(slots=True)
class DocumentChunk:
    """
    One semantic chunk of a document.
    """

    document_id: UUID
    chunk_index: int
    text: str


# ============================================================
# Service
# ============================================================

class DocumentChunker:
    """
    Split document text into semantic chunks.

    Strategy:
        - paragraph aware splitting
        - max size control
        - overlap between chunks
    """

    def __init__(
        self,
        *,
        max_chars: int = DOCUMENT_BLOCK_MAX_CHARS,
        overlap: int = DOCUMENT_BLOCK_OVERLAP,
    ) -> None:

        if overlap >= max_chars:
            raise ValueError("overlap must be smaller than max_chars")

        self.max_chars = max_chars
        self.overlap = overlap

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def chunk_document(
        self,
        *,
        document_id: UUID,
        text: str,
    ) -> List[DocumentChunk]:

        normalized = self._normalize_text(text)

        paragraphs = self._split_paragraphs(normalized)

        chunks = self._build_chunks(
            document_id=document_id,
            paragraphs=paragraphs,
        )

        logger.debug(
            "Document chunked",
            extra={
                "document_id": str(document_id),
                "chunks": len(chunks),
            },
        )

        return chunks

    # --------------------------------------------------------
    # Internal
    # --------------------------------------------------------

    def _normalize_text(
        self,
        text: str,
    ) -> str:

        if not text or not text.strip():
            raise ValueError("Document text must not be empty")

        normalized = text.replace("\r\n", "\n")

        # collapse excessive spaces
        normalized = " ".join(normalized.split())

        return normalized

    def _split_paragraphs(
        self,
        text: str,
    ) -> List[str]:

        paragraphs = text.split("\n")

        cleaned: List[str] = []

        for p in paragraphs:
            p = p.strip()

            if not p:
                continue

            cleaned.append(p)

        return cleaned

    def _build_chunks(
        self,
        *,
        document_id: UUID,
        paragraphs: List[str],
    ) -> List[DocumentChunk]:

        chunks: List[DocumentChunk] = []

        current: List[str] = []
        current_len = 0
        chunk_index = 0

        for paragraph in paragraphs:

            p_len = len(paragraph)

            if current_len + p_len > self.max_chars and current:

                chunk_text = "\n".join(current)

                chunks.append(
                    DocumentChunk(
                        document_id=document_id,
                        chunk_index=chunk_index,
                        text=chunk_text,
                    )
                )

                chunk_index += 1

                # overlap
                overlap_text = chunk_text[-self.overlap :]

                current = [overlap_text]
                current_len = len(overlap_text)

            current.append(paragraph)
            current_len += p_len

        if current:
            chunks.append(
                DocumentChunk(
                    document_id=document_id,
                    chunk_index=chunk_index,
                    text="\n".join(current),
                )
            )

        return chunks