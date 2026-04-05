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
            heading_path = metadata.get("heading_path") or []

            if not self._looks_like_deadline_block(
                text=text,
                heading_text=heading_text,
                semantic_hints=semantic_hints,
            ):
                continue

            deadline_value = self._extract_deadline_value_from_text(text)
            if not deadline_value:
                continue

            normalized_deadline_value = self._normalize_deadline_value(deadline_value)
            deadline_scope_text = self._extract_deadline_scope_text(
                text=text,
                heading_text=heading_text,
            )

            if self._is_deadline_noise(
                text=text,
                heading_text=heading_text,
                deadline_scope_text=deadline_scope_text,
            ):
                continue

            fact_type = self._detect_deadline_fact_type(
                text=text,
                heading_text=heading_text,
                hint=semantic_hints.get("deadline_kind_hint"),
                deadline_scope_text=deadline_scope_text,
            )

            is_service_core_deadline = self._is_service_core_deadline(
                fact_type=fact_type,
                text=text,
                heading_text=heading_text,
                deadline_scope_text=deadline_scope_text,
            )

            if fact_type == "internal_procedure_deadline" and not is_service_core_deadline:
                # На этом этапе лучше срезать явный procedural noise,
                # чем тащить его дальше в retrieval/generation.
                continue

            facts.append(
                {
                    "fact_type": fact_type,
                    "measure_code": measure_code,
                    "subject_category": None,
                    "condition_json": {
                        "heading_text": heading_text or None,
                        "heading_path": heading_path,
                        "deadline_scope_text": deadline_scope_text or None,
                        "block_order": block.get("block_order"),
                        "section_number": block.get("section_number"),
                        "clause_number": block.get("clause_number"),
                    },
                    "value_json": {
                        "deadline_value": normalized_deadline_value,
                        "source_text": text,
                    },
                    "validity_note": text,
                    "citation_json": block.get("citation_json") or {},
                    "metadata_json": {
                        "source": "block_deadline_enrichment",
                        "temporary": True,
                        "block_order": block.get("block_order"),
                        "heading_text": heading_text or None,
                        "heading_path": heading_path,
                        "deadline_scope_text": deadline_scope_text or None,
                        "is_service_core_deadline": is_service_core_deadline,
                        "deadline_kind_hint": semantic_hints.get("deadline_kind_hint"),
                    },
                }
            )

        return facts

    def _looks_like_deadline_block(
        self,
        *,
        text: str,
        heading_text: str,
        semantic_hints: dict[str, Any] | None = None,
    ) -> bool:
        haystack = f"{heading_text} {text}".lower()

        if semantic_hints and semantic_hints.get("is_deadline_related"):
            return True

        strong_markers = (
            "срок предоставления государственной услуги",
            "срок предоставления государственной услуги составляет",
            "срок предоставления государственной услуги не должен превышать",
            "срок исправления ошибок",
            "срок регистрации запроса",
            "о принятом решении заявитель",
            "уведомляется в течение",
            "не позднее 26-го числа",
            "решение принимается в течение",
            "максимальный срок предоставления государственной услуги",
        )
        if any(marker in haystack for marker in strong_markers):
            return True

        weak_markers = (
            "в течение",
            "не позднее",
            "рабочих дней",
            "рабочего дня",
            "календарных дней",
            "календарного дня",
            "в день регистрации",
            "в день поступления",
        )
        if any(marker in haystack for marker in weak_markers):
            return True

        return False

    def _extract_deadline_value_from_text(
        self,
        text: str,
    ) -> str | None:
        patterns = (
            r"в течение\s+\d+\s+(?:рабоч(?:их|его)|календарн(?:ых|ого))\s+дн(?:я|ей)",
            r"не более\s+\d+\s+(?:рабоч(?:их|его)|календарн(?:ых|ого))\s+дн(?:я|ей)",
            r"не позднее\s+\d+(?:-го)?\s+рабочего\s+дня",
            r"не позднее\s+\d+(?:-го)?\s+числа(?:\s+месяца)?",
            r"в день регистрации",
            r"в день поступления",
            r"ежемесячно",
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
        deadline_scope_text: str,
    ) -> str:
        haystack = f"{heading_text} {deadline_scope_text} {text}".lower()

        # 1. Выплата — самый отдельный и понятный случай.
        if any(
            marker in haystack
            for marker in (
                "не позднее 26-го числа",
                "не позднее 26 числа",
                "выплачивается",
                "выплата",
                "перечисляется",
                "предоставление едв осуществляется",
                "возобновление едв осуществляется",
            )
        ):
            return "payment_deadline"

        # 2. Исправление ошибок/опечаток.
        if any(
            marker in haystack
            for marker in (
                "исправлении опечаток",
                "исправления ошибок",
                "опечаток и ошибок",
                "выданном документе",
                "нового документа",
                "уведомления об отсутствии ошибок",
            )
        ):
            return "correction_deadline"

        # 3. Регистрация запроса/заявления.
        if any(
            marker in haystack
            for marker in (
                "срок регистрации запроса",
                "регистрация запроса",
                "регистрация заявления",
                "регистрируется",
                "регистрирует заявление",
                "в день поступления",
                "в первый рабочий день, следующий за днем их поступления",
            )
        ):
            return "registration_deadline"

        # 4. Срок на действия заявителя: донести, доработать, представить лично и т.п.
        if self._is_applicant_action_deadline(haystack):
            return "applicant_action_deadline"

        # 5. Срок принятия решения должен идти РАНЬШЕ notification,
        # иначе фразы вроде "принимает решение ... и уведомляет" уезжают не туда.
        if any(
            marker in haystack
            for marker in (
                "принимает решение",
                "решение о предоставлении",
                "решение об отказе",
                "решение о назначении",
                "решение о предоставлении государственной услуги",
                "решение о предоставлении едв",
                "решение об отказе в предоставлении едв",
                "срок предоставления государственной услуги",
                "максимальный срок предоставления государственной услуги",
                "срок предоставления государственной услуги составляет",
                "срок предоставления государственной услуги не должен превышать",
                "принятие решения",
            )
        ):
            return "decision_deadline"

        # 6. Уведомление заявителя — отдельный тип.
        if any(
            marker in haystack
            for marker in (
                "уведомляется",
                "уведомление направляется",
                "направляется заявителю",
                "извещает заявителя",
                "о принятом решении заявитель",
                "о готовности нового документа",
            )
        ):
            return "notification_deadline"

        # 7. Внутренние procedural deadlines.
        if any(
            marker in haystack
            for marker in (
                "межведомствен",
                "направляет межведомственный запрос",
                "направляет ее в министерство",
                "передаются руководителю",
                "передается руководителю",
                "направляет представленный запрос",
                "направляет представленные документы",
                "в течение 3 рабочих дней со дня получения",
                "в течение 1 рабочего дня со дня регистрации документов",
            )
        ):
            return "internal_procedure_deadline"

        if hint == "payment":
            return "payment_deadline"
        if hint == "notification":
            return "notification_deadline"
        if hint == "decision":
            return "decision_deadline"

        return "internal_procedure_deadline"
        
    def _is_applicant_action_deadline(
        self,
        haystack: str,
    ) -> bool:
        markers = (
            "необходимо предоставить лично",
            "необходимо представить лично",
            "необходимо представить",
            "доработки запроса",
            "необходимости доработки запроса",
            "доработанного запроса",
            "представления недостающих документов",
            "предоставления полного комплекта документов",
            "в срок, установленный пунктом",
            "со дня получения заявителем указанной информации",
            "со дня получения заявителем указанного уведомления",
            "заявитель вправе повторно обратиться",
            "заявителю необходимо",
            "представить лично",
            "доработать заявление",
            "доработки заявления",
        )
        return any(marker in haystack for marker in markers)
        
    def _extract_deadline_scope_text(
        self,
        *,
        text: str,
        heading_text: str,
    ) -> str:
        candidates = []

        clean_heading = (heading_text or "").strip()
        if clean_heading:
            candidates.append(clean_heading)

        sentences = re.split(r"(?<=[\.\:\;])\s+", text)
        for sentence in sentences:
            clean_sentence = sentence.strip()
            if not clean_sentence:
                continue
            if self._extract_deadline_value_from_text(clean_sentence):
                candidates.append(clean_sentence)
                break

        if candidates:
            return " | ".join(candidates)

        return text[:220].strip()

    def _normalize_deadline_value(
        self,
        value: str,
    ) -> str:
        normalized = " ".join((value or "").split())
        normalized = normalized.replace("26 числа", "26-го числа")
        return normalized

    def _is_service_core_deadline(
        self,
        *,
        fact_type: str,
        text: str,
        heading_text: str,
        deadline_scope_text: str,
    ) -> bool:
        haystack = f"{heading_text} {deadline_scope_text} {text}".lower()

        # Core-сроки, которые обычно нужны пользователю по вопросу о сроке услуги.
        if fact_type in {
            "decision_deadline",
            "notification_deadline",
            "payment_deadline",
        }:
            return True

        # Регистрация запроса по самой госуслуге — пограничный случай, но полезный.
        if fact_type == "registration_deadline":
            if any(
                marker in haystack
                for marker in (
                    "регистрация запроса о предоставлении государственной услуги",
                    "регистрация заявления о предоставлении государственной услуги",
                    "регистрирует заявление",
                    "регистрирует запрос",
                    "со дня их поступления",
                )
            ):
                return True
            return False

        # Всё это не core.
        if fact_type in {
            "correction_deadline",
            "applicant_action_deadline",
            "internal_procedure_deadline",
        }:
            return False

        # Дополнительная страховка.
        if any(
            marker in haystack
            for marker in (
                "межведомствен",
                "опросн",
                "обратн",
                "реинжиниринг",
                "направляет ее в министерство",
                "представить лично",
                "доработки запроса",
                "доработки заявления",
                "опечаток и ошибок",
            )
        ):
            return False

        return False

    def _is_deadline_noise(
        self,
        *,
        text: str,
        heading_text: str,
        deadline_scope_text: str,
    ) -> bool:
        haystack = f"{heading_text} {deadline_scope_text} {text}".lower()

        noise_markers = (
            "опросн",
            "реинжиниринг",
            "обратн",
            "рассматривает опросные формы",
            "направляет ее в министерство",
            "социологическ",
            "удовлетворенност",
            "контент-анализ",
            "жалоб",
            "обращений",
            "профилактика нарушений",
        )

        return any(marker in haystack for marker in noise_markers)