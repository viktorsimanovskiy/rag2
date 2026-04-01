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
        
        legal_facts = self._extract_deadline_facts(
            extraction=extraction,
            measure_codes=measure_codes,
        )

        enrichment_payload_json: dict[str, Any] = {
            "enricher": "basic_document_semantic_enricher",
            "is_temporary_test_stage": True,
            "source_authority": source_authority,
            "document_type": document_type,
            "measure_codes": measure_codes,
            "document_title": extraction.document_title,
            "doc_uid_base": extraction.doc_uid_base,
            "legal_facts_count": len(legal_facts),
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
            legal_facts=legal_facts,
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
        
    def _extract_deadline_facts(
        self,
        *,
        extraction: ExtractionResult,
        measure_codes: list[str],
    ) -> list[dict[str, Any]]:
        facts: list[dict[str, Any]] = []
        measure_code = measure_codes[0] if measure_codes else None

        for block in extraction.blocks:
            text = (block.get("content_clean") or "").strip()
            if not text:
                continue

            metadata = block.get("metadata_json") or {}
            semantic_hints = metadata.get("block_semantic_hints") or {}
            heading_text = (metadata.get("current_heading_text") or "").strip()

            if not semantic_hints.get("is_deadline_related") and not self._looks_like_deadline_block(text, heading_text):
                continue

            deadline_value = self._extract_deadline_value_from_text(text)
            if not deadline_value:
                continue

            fact_type = self._detect_deadline_fact_type(
                text=text,
                heading_text=heading_text,
                hint=semantic_hints.get("deadline_kind_hint"),
            )

            facts.append(
                {
                    "fact_type": fact_type,
                    "measure_code": measure_code,
                    "subject_category": None,
                    "condition_json": {
                        "heading_text": heading_text or None,
                        "block_order": block.get("block_order"),
                        "section_number": block.get("section_number"),
                        "clause_number": block.get("clause_number"),
                    },
                    "value_json": {
                        "deadline_value": deadline_value,
                        "source_text": text,
                    },
                    "validity_note": text,
                    "citation_json": block.get("citation_json") or {},
                    "metadata_json": {
                        "source": "block_deadline_enrichment",
                        "temporary": True,
                        "block_order": block.get("block_order"),
                        "heading_path": metadata.get("heading_path") or [],
                    },
                }
            )

        return facts

    def _looks_like_deadline_block(
        self,
        text: str,
        heading_text: str,
    ) -> bool:
        haystack = f"{heading_text} {text}".lower()

        return any(
            marker in haystack
            for marker in (
                "срок предоставления",
                "срок регистрации",
                "срок исправления",
                "в течение",
                "не позднее",
                "уведомляется",
                "уведомление направляется",
                "решение принимается",
                "выплачивается",
                "26-го числа",
                "26 числа",
            )
        )

    def _extract_deadline_value_from_text(
        self,
        text: str,
    ) -> str | None:
        patterns = (
            r"в течение\s+\d+\s+(?:рабоч(?:их|его)|календарн(?:ых|ого))\s+дн(?:я|ей)",
            r"не более\s+\d+\s+(?:рабоч(?:их|его)|календарн(?:ых|ого))\s+дн(?:я|ей)",
            r"не позднее\s+26(?:-го)?\s+числа(?:\s+месяца)?",
            r"не позднее\s+\d+(?:-го)?\s+рабочего\s+дня",
            r"в день регистрации",
            r"в день поступления",
        )

        lowered = text.lower()
        for pattern in patterns:
            match = re.search(pattern, lowered, flags=re.IGNORECASE)
            if match:
                return text[match.start():match.end()].strip()

        return None

    def _detect_deadline_fact_type(
        self,
        *,
        text: str,
        heading_text: str,
        hint: str | None,
    ) -> str:
        haystack = f"{heading_text} {text}".lower()

        if hint == "payment" or any(x in haystack for x in ("выплачивается", "выплата", "26-го числа", "26 числа")):
            return "payment_deadline"

        if hint == "notification" or any(x in haystack for x in ("уведомляется", "уведомление", "направляется заявителю", "извещает")):
            return "notification_deadline"

        if any(x in haystack for x in ("регистрац", "регистрирует", "регистрация")):
            return "registration_deadline"

        if any(x in haystack for x in ("исправлении опечаток", "исправления ошибок", "опечаток и ошибок")):
            return "correction_deadline"

        return "decision_deadline"