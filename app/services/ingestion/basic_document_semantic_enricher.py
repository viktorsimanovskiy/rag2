from __future__ import annotations

import re
from typing import Any

from app.services.ingestion.document_ingestion_pipeline import (
    ExtractionResult,
    SemanticEnrichmentInput,
    SemanticEnrichmentResult,
)


class BasicDocumentSemanticEnricher:
    """
    Minimal deterministic enricher for the first real ingestion tests.

    Important:
    - this implementation is temporary
    - it must not become the final long-term enrichment strategy
    - keyword-driven enrichment does not scale to 111+ services
    """

    _AUTHORITY_PATTERNS: tuple[tuple[str, str], ...] = (
        (
            r"министерств[оа]\s+социальной\s+политики\s+красноярского\s+края",
            "ministry_social_policy_krsk",
        ),
        (
            r"правительств[оа]\s+красноярского\s+края",
            "government_krsk",
        ),
        (
            r"губернатор[а]?\s+красноярского\s+края",
            "governor_krsk",
        ),
    )

    async def enrich(
        self,
        payload: SemanticEnrichmentInput,
    ) -> SemanticEnrichmentResult:
        text = (payload.normalized_text or "").strip()
        extraction: ExtractionResult = payload.extraction_result
        title = (extraction.document_title or "").strip()

        haystack = f"{title}\n{text[:5000]}".lower()

        source_authority = self._detect_source_authority(haystack)
        document_type = self._detect_document_type(haystack)
        measure_codes = self._detect_measure_codes(haystack)
        aliases = self._build_aliases(measure_codes)

        enrichment_payload_json: dict[str, Any] = {
            "enricher": "basic_document_semantic_enricher",
            "is_temporary_test_stage": True,
            "source_authority": source_authority,
            "document_type": document_type,
            "measure_codes": measure_codes,
            "document_title": extraction.document_title,
            "doc_uid_base": extraction.doc_uid_base,
            "revision_date": (
                extraction.revision_date.isoformat()
                if extraction.revision_date is not None
                else None
            ),
            "warning": (
                "Temporary deterministic enrichment. "
                "Must be replaced later with a more scalable approach."
            ),
        }

        return SemanticEnrichmentResult(
            source_authority=source_authority,
            document_type=document_type,
            measure_codes=measure_codes,
            legal_facts=[],
            aliases=aliases,
            enrichment_payload_json=enrichment_payload_json,
        )

    def _detect_source_authority(self, haystack: str) -> str | None:
        for pattern, code in self._AUTHORITY_PATTERNS:
            if re.search(pattern, haystack, flags=re.IGNORECASE):
                return code

        if "красноярского края" in haystack:
            return "krasnoyarsk_krai_authority"

        return None

    def _detect_document_type(self, haystack: str) -> str:
        if "административный регламент" in haystack:
            return "administrative_regulation"
        if re.search(r"\bприказ\b", haystack, flags=re.IGNORECASE):
            return "order"
        if re.search(r"\bпостановлени[ея]\b", haystack, flags=re.IGNORECASE):
            return "resolution"
        if re.search(r"\bзакон\b", haystack, flags=re.IGNORECASE):
            return "law"
        return "normative_document"

    def _detect_measure_codes(self, haystack: str) -> list[str]:
        codes: list[str] = []

        if re.search(
            r"\bедв\b|ежемесячн\w*\s+денежн\w*\s+выплат",
            haystack,
            flags=re.IGNORECASE,
        ):
            codes.append("edv")

        return codes

    def _build_aliases(self, measure_codes: list[str]) -> list[dict[str, Any]]:
        aliases: list[dict[str, Any]] = []

        if "edv" in measure_codes:
            aliases.append(
                {
                    "alias": "ЕДВ",
                    "measure_code": "edv",
                    "canonical_name": "Ежемесячная денежная выплата",
                    "metadata_json": {
                        "source": "deterministic_enricher",
                        "temporary": True,
                    },
                }
            )

        return aliases