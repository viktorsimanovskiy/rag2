# ============================================================
# File: app/services/retrieval/service_discovery.py
# Purpose:
#   Safe deterministic discovery of potentially relevant services
#   for broad questions like "что мне положено".
#
# Important:
#   This service does NOT decide that a person is eligible.
#   It only finds services whose applicant-category rows may match
#   the signs stated in the user's question.
# ============================================================

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.documents import DocumentRegistry, DocumentTable, DocumentTableRow
from app.services.retrieval.applicant_category_taxonomy import (
    build_applicant_category_groups,
    normalize_text as normalize_taxonomy_text,
)


SERVICE_DISCOVERY_VERSION = "service_discovery_v3_row_filtering"


# ============================================================
# DTOs
# ============================================================

@dataclass(slots=True)
class ApplicantSignalDefinition:
    code: str
    label: str
    question_patterns: tuple[str, ...]
    evidence_terms: tuple[str, ...]
    weight: float


@dataclass(slots=True)
class ApplicantSignal:
    code: str
    label: str
    matched_question_patterns: list[str]
    evidence_terms: list[str]
    weight: float


@dataclass(slots=True)
class ServiceDiscoveryInput:
    question_text_raw: str
    question_text_normalized: str
    max_services: int = 7
    max_rows_per_service: int = 3
    min_score: float = 2.0


@dataclass(slots=True)
class ServiceDiscoveryMatchedRow:
    document_id: UUID
    table_id: UUID
    row_id: UUID
    row_order: int

    service_key: str
    service_name_short: Optional[str]
    service_name_full: Optional[str]
    service_frgu_1: Optional[str]
    service_frgu_3: Optional[str]

    document_name: Optional[str]
    original_filename: Optional[str]
    table_title: Optional[str]
    table_number: Optional[str]

    applicant_category_id: Optional[str]
    applicant_category_name: Optional[str]
    row_summary: Optional[str]

    score: float
    matched_signal_codes: list[str] = field(default_factory=list)
    matched_signal_labels: list[str] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)
    citation_json: dict[str, Any] = field(default_factory=dict)
    metadata_json: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ServiceDiscoveryCandidate:
    service_key: str
    service_name_short: Optional[str]
    service_name_full: Optional[str]
    service_frgu_1: Optional[str]
    service_frgu_3: Optional[str]
    document_id: UUID
    document_name: Optional[str]
    original_filename: Optional[str]

    score: float
    matched_signal_codes: list[str]
    matched_signal_labels: list[str]
    matched_terms: list[str]
    matched_rows: list[ServiceDiscoveryMatchedRow] = field(default_factory=list)


@dataclass(slots=True)
class ServiceDiscoveryResult:
    can_answer: bool
    answer_text: str
    answer_text_short: str
    candidates: list[ServiceDiscoveryCandidate] = field(default_factory=list)
    signals: list[ApplicantSignal] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    debug_payload_json: dict[str, Any] = field(default_factory=dict)
    citations_json: list[dict[str, Any]] = field(default_factory=list)


# ============================================================
# Service
# ============================================================

class ServiceDiscovery:
    """
    Deterministic service finder for broad entitlement questions.

    It searches only applicant-category rows from identifiers tables.
    The result is deliberately cautious: it returns possible directions,
    not a legal conclusion that the applicant has the right to receive them.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.signal_definitions = _build_signal_definitions()

    async def discover(self, payload: ServiceDiscoveryInput) -> ServiceDiscoveryResult:
        question_text = payload.question_text_normalized or payload.question_text_raw
        normalized_question = normalize_text(question_text)
        signals = self._extract_signals(normalized_question)

        if not signals:
            return self._build_no_signal_result(normalized_question)

        rows = await self._load_identifier_rows()
        matched_rows = self._score_rows(
            rows=rows,
            normalized_question=normalized_question,
            signals=signals,
            min_score=payload.min_score,
        )

        candidates = self._group_rows_by_service(
            matched_rows,
            max_services=payload.max_services,
            max_rows_per_service=payload.max_rows_per_service,
        )

        if not candidates:
            return self._build_no_candidate_result(
                normalized_question=normalized_question,
                signals=signals,
                scanned_rows_count=len(rows),
            )

        citations = self._build_citations(candidates)
        answer_text = self._render_answer(candidates=candidates, signals=signals)
        answer_text_short = _shorten(answer_text, limit=420)

        return ServiceDiscoveryResult(
            can_answer=True,
            answer_text=answer_text,
            answer_text_short=answer_text_short,
            candidates=candidates,
            signals=signals,
            warnings=[
                "Подбор мер не подтверждает право заявителя на получение услуги.",
                "Для точного вывода нужны дополнительные условия из конкретного НПА.",
            ],
            debug_payload_json={
                "version": SERVICE_DISCOVERY_VERSION,
                "normalized_question": normalized_question,
                "signals": [self._signal_to_json(signal) for signal in signals],
                "scanned_identifier_rows_count": len(rows),
                "matched_identifier_rows_count": len(matched_rows),
                "selected_services_count": len(candidates),
            },
            citations_json=citations,
        )

    # --------------------------------------------------------
    # Loading
    # --------------------------------------------------------

    async def _load_identifier_rows(self) -> list[dict[str, Any]]:
        stmt = (
            select(
                DocumentRegistry.document_id,
                DocumentRegistry.service_key,
                DocumentRegistry.service_name_short,
                DocumentRegistry.service_name_full,
                DocumentRegistry.service_frgu_1,
                DocumentRegistry.service_frgu_3,
                DocumentRegistry.document_name,
                DocumentRegistry.original_filename,
                DocumentTable.table_id,
                DocumentTable.table_title,
                DocumentTable.table_number,
                DocumentTableRow.row_id,
                DocumentTableRow.row_order,
                DocumentTableRow.row_summary,
                DocumentTableRow.normalized_row_json,
                DocumentTableRow.metadata_json,
                DocumentTableRow.citation_json,
            )
            .join(DocumentTable, DocumentTable.document_id == DocumentRegistry.document_id)
            .join(DocumentTableRow, DocumentTableRow.table_id == DocumentTable.table_id)
            .where(DocumentRegistry.status == "active")
            .where(DocumentTable.table_type == "identifiers")
            .order_by(DocumentRegistry.service_key, DocumentTableRow.row_order)
        )
        result = await self.db.execute(stmt)
        return [dict(row) for row in result.mappings().all()]

    # --------------------------------------------------------
    # Signals and scoring
    # --------------------------------------------------------

    def _extract_signals(self, normalized_question: str) -> list[ApplicantSignal]:
        signals: list[ApplicantSignal] = []

        for definition in self.signal_definitions:
            matched_patterns = [
                pattern
                for pattern in definition.question_patterns
                if re.search(pattern, normalized_question, flags=re.IGNORECASE)
            ]
            if not matched_patterns:
                continue

            signals.append(
                ApplicantSignal(
                    code=definition.code,
                    label=definition.label,
                    matched_question_patterns=matched_patterns,
                    evidence_terms=list(definition.evidence_terms),
                    weight=definition.weight,
                )
            )

        return self._deduplicate_signals(signals)

    def _score_rows(
        self,
        *,
        rows: list[dict[str, Any]],
        normalized_question: str,
        signals: list[ApplicantSignal],
        min_score: float,
    ) -> list[ServiceDiscoveryMatchedRow]:
        question_terms = extract_meaningful_terms(normalized_question)
        scored_rows: list[ServiceDiscoveryMatchedRow] = []

        for row in rows:
            service_key = str(row.get("service_key") or "").strip()
            if not service_key:
                continue

            normalized_row_json = _dict_or_empty(row.get("normalized_row_json"))
            metadata_json = _dict_or_empty(row.get("metadata_json"))
            citation_json = _dict_or_empty(row.get("citation_json"))
            row_summary = _str_or_none(row.get("row_summary"))
            applicant_category_id = self._extract_applicant_category_id(
                normalized_row_json,
                metadata_json,
                row_summary=row_summary,
            )
            applicant_category_name = self._extract_applicant_category_name(
                normalized_row_json,
                metadata_json,
                row_summary=row_summary,
            )

            if self._is_technical_identifier_row(
                applicant_category_id=applicant_category_id,
                applicant_category_name=applicant_category_name,
                row_summary=row_summary,
            ):
                continue

            row_text = self._build_search_text(
                row,
                applicant_category_id=applicant_category_id,
                applicant_category_name=applicant_category_name,
                metadata_json=metadata_json,
            )
            if not row_text:
                continue

            matched_signal_codes: list[str] = []
            matched_signal_labels: list[str] = []
            matched_terms: list[str] = []
            score = 0.0

            for signal in signals:
                signal_matches = [term for term in signal.evidence_terms if term and term in row_text]
                if not signal_matches:
                    continue

                matched_signal_codes.append(signal.code)
                matched_signal_labels.append(signal.label)
                matched_terms.extend(signal_matches)
                score += signal.weight

                # Для неполной семьи засчитываем только явные формулировки.
                # Общее "одиноко проживающие" не равно "одинокий родитель".
                if signal.code == "single_parent" and _has_single_parent_evidence(row_text):
                    score += 1.5

                if signal.code == "large_family" and any(
                    term in row_text for term in ("многодет", "троих", "трех", "трёх", "3", "трое")
                ):
                    score += 1.5

                if signal.code in {"honorary_donor", "vov_participant", "vov_disabled", "combat_veteran"}:
                    score += 0.7

            lexical_matches = [term for term in question_terms if term in row_text]
            if lexical_matches:
                score += min(2.5, len(set(lexical_matches)) * 0.35)
                matched_terms.extend(lexical_matches)

            territorial_context = normalize_text(
                " ".join(
                    [
                        row_text,
                        str(row.get("service_name_short") or ""),
                        str(row.get("service_name_full") or ""),
                        str(row.get("document_name") or ""),
                        str(row.get("original_filename") or ""),
                    ]
                )
            )
            score = self._apply_territorial_penalty(
                score=score,
                row_text=territorial_context,
                normalized_question=normalized_question,
            )

            if score < min_score:
                continue

            scored_rows.append(
                ServiceDiscoveryMatchedRow(
                    document_id=row["document_id"],
                    table_id=row["table_id"],
                    row_id=row["row_id"],
                    row_order=int(row.get("row_order") or 0),
                    service_key=service_key,
                    service_name_short=_str_or_none(row.get("service_name_short")),
                    service_name_full=_str_or_none(row.get("service_name_full")),
                    service_frgu_1=_str_or_none(row.get("service_frgu_1")),
                    service_frgu_3=_str_or_none(row.get("service_frgu_3")),
                    document_name=_str_or_none(row.get("document_name")),
                    original_filename=_str_or_none(row.get("original_filename")),
                    table_title=_str_or_none(row.get("table_title")),
                    table_number=_str_or_none(row.get("table_number")),
                    applicant_category_id=applicant_category_id,
                    applicant_category_name=applicant_category_name,
                    row_summary=row_summary,
                    score=round(score, 4),
                    matched_signal_codes=_stable_unique(matched_signal_codes),
                    matched_signal_labels=_stable_unique(matched_signal_labels),
                    matched_terms=_stable_unique(matched_terms),
                    citation_json=citation_json,
                    metadata_json=metadata_json,
                )
            )

        scored_rows.sort(key=lambda item: (-item.score, item.service_name_short or "", item.row_order))
        return scored_rows

    def _group_rows_by_service(
        self,
        rows: list[ServiceDiscoveryMatchedRow],
        *,
        max_services: int,
        max_rows_per_service: int,
    ) -> list[ServiceDiscoveryCandidate]:
        grouped: dict[str, list[ServiceDiscoveryMatchedRow]] = {}
        for row in rows:
            grouped.setdefault(row.service_key, []).append(row)

        candidates: list[ServiceDiscoveryCandidate] = []
        for service_key, service_rows in grouped.items():
            service_rows.sort(key=lambda item: (-item.score, item.row_order))
            selected_rows = service_rows[:max_rows_per_service]
            first = selected_rows[0]
            signal_codes = _stable_unique(
                code for row in selected_rows for code in row.matched_signal_codes
            )
            signal_labels = _stable_unique(
                label for row in selected_rows for label in row.matched_signal_labels
            )
            matched_terms = _stable_unique(
                term for row in selected_rows for term in row.matched_terms
            )

            # Итоговый счёт: лучший ряд + вклад дополнительных рядов + разнообразие признаков.
            score = first.score
            if len(selected_rows) > 1:
                score += sum(row.score for row in selected_rows[1:]) * 0.35
            score += len(signal_codes) * 0.6

            candidates.append(
                ServiceDiscoveryCandidate(
                    service_key=service_key,
                    service_name_short=first.service_name_short,
                    service_name_full=first.service_name_full,
                    service_frgu_1=first.service_frgu_1,
                    service_frgu_3=first.service_frgu_3,
                    document_id=first.document_id,
                    document_name=first.document_name,
                    original_filename=first.original_filename,
                    score=round(score, 4),
                    matched_signal_codes=signal_codes,
                    matched_signal_labels=signal_labels,
                    matched_terms=matched_terms,
                    matched_rows=selected_rows,
                )
            )

        candidates.sort(key=lambda item: (-item.score, item.service_name_short or ""))
        return candidates[:max_services]

    def _build_search_text(
        self,
        row: dict[str, Any],
        *,
        applicant_category_id: Optional[str],
        applicant_category_name: Optional[str],
        metadata_json: dict[str, Any],
    ) -> str:
        """
        Текст для совпадений строим прежде всего по самой категории заявителя.

        Важно не подмешивать название услуги, название таблицы и служебные
        заголовки: иначе широкие слова вроде "дети", "родители", "семья"
        начинают давать ложные совпадения по контексту, а не по категории.
        """
        parts: list[str] = []

        if applicant_category_id:
            parts.append(applicant_category_id)
        if applicant_category_name:
            parts.append(applicant_category_name)

        semantic_cells = _cells_by_semantic_key(metadata_json)
        for key in (
            "applicant_category_name",
            "applicant_category",
            "category_name",
            "applicant_sign",
            "applicant_feature",
        ):
            value = semantic_cells.get(key)
            if value:
                parts.append(str(value))

        if not parts:
            # Резерв для старых/нестандартно разобранных строк.
            parts.append(str(row.get("row_summary") or ""))

        return normalize_text(" ".join(parts))

    @staticmethod
    def _is_technical_identifier_row(
        *,
        applicant_category_id: Optional[str],
        applicant_category_name: Optional[str],
        row_summary: Optional[str],
    ) -> bool:
        text = normalize_text(" ".join([applicant_category_id or "", applicant_category_name or "", row_summary or ""]))
        if not text:
            return True

        technical_exact = {
            "наименование признака заявителя",
            "идентификаторы категорий",
            "идентификаторы категорий признаков заявителей",
            "n п п",
            "n п/п",
        }
        category_text = normalize_text(applicant_category_name or "")
        id_text = normalize_text(applicant_category_id or "")
        if category_text in technical_exact or id_text in technical_exact:
            return True

        if "наименование признака заявителя наименование признака заявителя" in text:
            return True
        if text.startswith("идентификаторы категорий ") or text == "идентификаторы категорий":
            return True

        return False

    @staticmethod
    def _apply_territorial_penalty(
        *,
        score: float,
        row_text: str,
        normalized_question: str,
    ) -> float:
        territorial_terms = ("эвенк", "таймыр", "долгано", "ненец")
        question_mentions_territory = any(term in normalized_question for term in territorial_terms)
        row_mentions_territory = any(term in row_text for term in territorial_terms)
        if row_mentions_territory and not question_mentions_territory:
            return max(0.0, score - 2.5)
        return score

    @staticmethod
    def _extract_applicant_category_id(
        normalized_row_json: dict[str, Any],
        metadata_json: dict[str, Any],
        *,
        row_summary: Optional[str],
    ) -> Optional[str]:
        direct = _str_or_none(normalized_row_json.get("applicant_category_id"))
        if direct:
            return direct

        cells = _cells_by_semantic_key(metadata_json)
        cell_value = _str_or_none(cells.get("applicant_category_id"))
        if cell_value:
            return cell_value

        return _extract_value_from_identifier_row_summary(
            row_summary,
            labels=("идентификатор категории", "идентификаторы категорий"),
        )

    @staticmethod
    def _extract_applicant_category_name(
        normalized_row_json: dict[str, Any],
        metadata_json: dict[str, Any],
        *,
        row_summary: Optional[str],
    ) -> Optional[str]:
        direct = _str_or_none(normalized_row_json.get("applicant_category_name"))
        if direct and not _looks_like_identifier_table_header(direct):
            return direct

        cells = _cells_by_semantic_key(metadata_json)
        for key in (
            "applicant_category_name",
            "applicant_category",
            "category_name",
            "applicant_sign",
            "applicant_feature",
            "наименование признака заявителя",
        ):
            cell_value = _str_or_none(cells.get(key))
            if cell_value and not _looks_like_identifier_table_header(cell_value):
                return cell_value

        return _extract_value_from_identifier_row_summary(
            row_summary,
            labels=("наименование признака заявителя",),
        )

    @staticmethod
    def _deduplicate_signals(signals: list[ApplicantSignal]) -> list[ApplicantSignal]:
        result: list[ApplicantSignal] = []
        seen: set[str] = set()
        for signal in signals:
            if signal.code in seen:
                continue
            seen.add(signal.code)
            result.append(signal)
        return result

    @staticmethod
    def _signal_to_json(signal: ApplicantSignal) -> dict[str, Any]:
        return {
            "code": signal.code,
            "label": signal.label,
            "matched_question_patterns": signal.matched_question_patterns,
            "evidence_terms": signal.evidence_terms,
            "weight": signal.weight,
        }

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    def _render_answer(
        self,
        *,
        candidates: list[ServiceDiscoveryCandidate],
        signals: list[ApplicantSignal],
    ) -> str:
        signal_labels = ", ".join(signal.label for signal in signals)
        lines: list[str] = [
            "По указанным признакам я нашёл несколько мер, которые могут быть релевантны. "
            "Это не означает, что право на них уже подтверждено: для точного вывода нужно проверить условия конкретной услуги по НПА.",
        ]

        if signal_labels:
            lines.append(f"Учтённые признаки из вопроса: {signal_labels}.")

        lines.append("Возможные направления для проверки:")

        for index, candidate in enumerate(candidates, start=1):
            service_name = candidate.service_name_short or candidate.service_name_full or candidate.service_key
            matched_labels = ", ".join(candidate.matched_signal_labels[:5])
            row_fragments = self._render_candidate_row_fragments(candidate)

            line = f"{index}. {service_name}"
            if matched_labels:
                line += f" — совпавшие признаки: {matched_labels}"
            if row_fragments:
                line += f". В таблице категорий заявителей найдено: {row_fragments}"
            line += "."
            lines.append(line)

        lines.append(
            "Что нужно уточнить для точного ответа: состав семьи, возраст детей, место жительства, доход, "
            "наличие подтверждённых статусов и цель обращения. После уточнения можно проверять каждую меру отдельно."
        )
        return "\n".join(lines)

    @staticmethod
    def _render_candidate_row_fragments(candidate: ServiceDiscoveryCandidate) -> str:
        fragments: list[str] = []
        for row in candidate.matched_rows[:2]:
            category = row.applicant_category_name
            if not category and row.row_summary and not _looks_like_identifier_table_header(row.row_summary):
                category = row.row_summary
            category = _shorten(category, limit=160) if category else None
            if not category:
                continue
            if row.applicant_category_id:
                fragments.append(f"{row.applicant_category_id} — {category}")
            else:
                fragments.append(category)
        return "; ".join(fragments[:2])

    def _build_citations(self, candidates: list[ServiceDiscoveryCandidate]) -> list[dict[str, Any]]:
        citations: list[dict[str, Any]] = []
        seen_row_ids: set[UUID] = set()

        for candidate in candidates:
            for row in candidate.matched_rows[:1]:
                if row.row_id in seen_row_ids:
                    continue
                seen_row_ids.add(row.row_id)
                citations.append(
                    {
                        "source_type": "table_row",
                        "document_id": str(row.document_id),
                        "source_id": str(row.row_id),
                        "display_label": self._build_display_label(row),
                        "citation_text": self._build_citation_text(row),
                        "document_name": row.document_name,
                        "download_url": None,
                        "metadata_json": {
                            "service_key": row.service_key,
                            "service_name_short": row.service_name_short,
                            "table_title": row.table_title,
                            "table_number": row.table_number,
                            "row_order": row.row_order,
                            "applicant_category_id": row.applicant_category_id,
                            "matched_signal_codes": row.matched_signal_codes,
                            "matched_terms": row.matched_terms,
                        },
                    }
                )
                if len(citations) >= 5:
                    return citations
        return citations

    @staticmethod
    def _build_display_label(row: ServiceDiscoveryMatchedRow) -> str:
        service_name = row.service_name_short or row.service_name_full or row.service_key
        table_title = row.table_title or "таблица категорий заявителей"
        return f"{service_name}: {table_title}, строка {row.row_order}"

    @staticmethod
    def _build_citation_text(row: ServiceDiscoveryMatchedRow) -> str:
        parts = []
        if row.applicant_category_id:
            parts.append(f"идентификатор категории: {row.applicant_category_id}")
        if row.applicant_category_name:
            parts.append(f"категория: {row.applicant_category_name}")
        if not parts and row.row_summary and not _looks_like_identifier_table_header(row.row_summary):
            parts.append(row.row_summary)
        return "; ".join(parts) or "строка таблицы категорий заявителей"

    def _build_no_signal_result(self, normalized_question: str) -> ServiceDiscoveryResult:
        answer_text = (
            "Я вижу, что вопрос похож на подбор мер поддержки, но в нём недостаточно признаков заявителя. "
            "Укажите, пожалуйста, кто обращается, состав семьи, наличие детей, инвалидности, статуса ветерана, "
            "пенсионного статуса, дохода или трудной жизненной ситуации. Тогда можно подобрать возможные меры для проверки."
        )
        return ServiceDiscoveryResult(
            can_answer=False,
            answer_text=answer_text,
            answer_text_short=answer_text,
            warnings=["Недостаточно признаков заявителя для подбора мер."],
            debug_payload_json={
                "version": SERVICE_DISCOVERY_VERSION,
                "normalized_question": normalized_question,
                "reason_code": "no_applicant_signals",
            },
        )

    def _build_no_candidate_result(
        self,
        *,
        normalized_question: str,
        signals: list[ApplicantSignal],
        scanned_rows_count: int,
    ) -> ServiceDiscoveryResult:
        signal_labels = ", ".join(signal.label for signal in signals) or "не определены"
        answer_text = (
            "Я выделил признаки заявителя, но не нашёл достаточно надёжных совпадений в таблицах категорий заявителей. "
            f"Учтённые признаки: {signal_labels}. Лучше уточнить вопрос или проверить конкретную услугу."
        )
        return ServiceDiscoveryResult(
            can_answer=False,
            answer_text=answer_text,
            answer_text_short=answer_text,
            signals=signals,
            warnings=["Не найдено достаточно надёжных совпадений по таблицам категорий заявителей."],
            debug_payload_json={
                "version": SERVICE_DISCOVERY_VERSION,
                "normalized_question": normalized_question,
                "reason_code": "no_matching_identifier_rows",
                "signals": [self._signal_to_json(signal) for signal in signals],
                "scanned_identifier_rows_count": scanned_rows_count,
            },
        )


# ============================================================
# Signal definitions
# ============================================================

def _build_signal_definitions() -> list[ApplicantSignalDefinition]:
    """
    Словарь признаков заявителя строится из единой карты категорий.

    Карта вынесена в applicant_category_taxonomy.py, чтобы один и тот же
    смысловой перечень использовали и подбор мер, и диагностический отчёт
    по таблицам категорий заявителей.
    """
    definitions: list[ApplicantSignalDefinition] = []
    for group in build_applicant_category_groups():
        definitions.append(
            ApplicantSignalDefinition(
                code=group.code,
                label=group.label,
                question_patterns=group.question_patterns,
                evidence_terms=tuple(
                    normalize_taxonomy_text(term)
                    for term in group.evidence_terms
                    if normalize_taxonomy_text(term)
                ),
                weight=group.weight,
            )
        )
    return definitions


# ============================================================
# Text helpers
# ============================================================

def normalize_text(value: str) -> str:
    text = str(value or "").strip().lower().replace("ё", "е")
    text = re.sub(r"[\u00a0\t\r\n]+", " ", text)
    text = re.sub(r"[^0-9a-zа-я_\-\s./]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_meaningful_terms(value: str) -> list[str]:
    stop_words = {
        "что", "мне", "нам", "для", "при", "как", "или", "это", "могу", "может",
        "какие", "какая", "какой", "положено", "положена", "положены", "получить",
        "имею", "право", "поддержки", "поддержку", "мера", "меры", "услуги",
    }
    terms: list[str] = []
    for token in re.split(r"\s+", normalize_text(value)):
        token = token.strip(" .,/\\-_")
        if len(token) < 4:
            continue
        if token in stop_words:
            continue
        terms.append(token)
    return _stable_unique(terms)


def _has_single_parent_evidence(value: str) -> bool:
    text = normalize_text(value)
    positive_patterns = (
        r"\bмать\s*-?\s*одиноч",
        r"\bотец\s*-?\s*одиноч",
        r"\bодинок\w*\s+(?:мать|отец|родител)",
        r"\bединственн\w*\s+родител",
        r"\bнеполн\w*\s+сем",
        r"\bодин\s+из\s+родител\w*.*\bнеполн\w*\s+сем",
    )
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in positive_patterns)


def _extract_value_from_identifier_row_summary(
    row_summary: Optional[str],
    *,
    labels: tuple[str, ...],
) -> Optional[str]:
    """
    Извлекает чистое значение из человекочитаемого row_summary.

    В части документов структурные поля категории заявителя заполнены не полностью,
    и в ответ раньше попадал весь служебный фрагмент вида:
    "Таблица: ... Колонки таблицы: ... Наименование признака заявителя: ...".
    Для подбора мер нам нужна только сама категория заявителя.
    """
    text = _str_or_none(row_summary)
    if not text:
        return None

    stop_labels = (
        "перечень результатов предоставления государственной услуги",
        "решение о представлении",
        "решение о предоставлении",
        "результат предоставления",
        "идентификатор категории",
        "идентификаторы категорий",
        "n п п",
        "n п/п",
    )

    for label in labels:
        pattern = re.compile(
            rf"{re.escape(label)}\s*:\s*(.+?)(?=(?:\.\s*)?(?:"
            + "|".join(re.escape(stop_label) for stop_label in stop_labels if stop_label != label)
            + r")\s*:|$)",
            flags=re.IGNORECASE | re.DOTALL,
        )
        match = pattern.search(text)
        if not match:
            continue

        value = _str_or_none(match.group(1))
        if not value:
            continue
        value = re.sub(r"^[:.\s]+", "", value).strip()
        value = re.sub(r"\s+", " ", value).strip(" .;:-")
        if value and not _looks_like_identifier_table_header(value):
            return value

    return None


def _looks_like_identifier_table_header(value: Optional[str]) -> bool:
    text = normalize_text(value or "")
    if not text:
        return False

    technical_phrases = (
        "таблица идентификаторы категорий",
        "идентификаторы категорий признаков заявителей",
        "идентификаторы категорий",
        "колонки таблицы",
        "наименование признака заявителя",
    )
    if any(phrase in text for phrase in technical_phrases):
        # Короткие/служебные фрагменты точно не являются категорией.
        # Длинный row_summary с этой фразой тоже нельзя показывать как категорию,
        # потому что чистое значение должно быть извлечено отдельно.
        return True

    return False


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _cells_by_semantic_key(metadata_json: dict[str, Any]) -> dict[str, Any]:
    cells = metadata_json.get("cells_by_semantic_key")
    return cells if isinstance(cells, dict) else {}


def _str_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = " ".join(str(value).strip().split())
    return text or None


def _stable_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = " ".join(str(value or "").strip().split())
        if not text:
            continue
        key = text.lower().replace("ё", "е")
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _shorten(value: Optional[str], *, limit: int) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"
