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


<<<<<<< HEAD
SERVICE_DISCOVERY_VERSION = "service_discovery_v11_practical_need_profiles_noise_cleanup"
# second_step_30_practical_need_profiles_noise_cleanup_v1
=======
SERVICE_DISCOVERY_VERSION = "service_discovery_v7_school_noise_filter"
>>>>>>> bba36515540dbe4eec46b473a736432fb4d55ceb


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
class ServiceDiscoveryProfile:
    """
    Narrowing profile for broad service discovery.

    The profile is not a legal concept. It only helps keep demo answers
    focused when the user's wording clearly describes a practical crisis
    such as food, fuel, school expenses or fire.
    """

    code: str
    label: str
    max_services: int = 7
    max_rows_per_service: int = 3


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
        profile = self._build_discovery_profile(normalized_question, signals)

        if not signals:
            return self._build_no_signal_result(normalized_question)

        rows = await self._load_identifier_rows()
        matched_rows = self._score_rows(
            rows=rows,
            normalized_question=normalized_question,
            signals=signals,
            min_score=payload.min_score,
            profile=profile,
        )

        candidates = self._group_rows_by_service(
            matched_rows,
            max_services=min(payload.max_services, profile.max_services),
            max_rows_per_service=min(payload.max_rows_per_service, profile.max_rows_per_service),
            profile=profile,
        )

        if not candidates:
            return self._build_no_candidate_result(
                normalized_question=normalized_question,
                signals=signals,
                scanned_rows_count=len(rows),
            )

        citations = self._build_citations(candidates)
        answer_text = self._render_answer(candidates=candidates, signals=signals, profile=profile)
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
                "profile": {
                    "code": profile.code,
                    "label": profile.label,
                    "max_services": profile.max_services,
                    "max_rows_per_service": profile.max_rows_per_service,
                },
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
        profile: ServiceDiscoveryProfile,
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

            service_context = normalize_text(
                " ".join(
                    [
                        str(row.get("service_name_short") or ""),
                        str(row.get("service_name_full") or ""),
                        str(row.get("document_name") or ""),
                        str(row.get("original_filename") or ""),
                    ]
                )
            )

            service_context_bonus = self._score_service_context_for_signals(
                service_context=service_context,
                normalized_question=normalized_question,
                signals=signals,
                matched_signal_codes=matched_signal_codes,
                matched_signal_labels=matched_signal_labels,
                matched_terms=matched_terms,
            )
            score += service_context_bonus

            penalty_context = normalize_text(" ".join([row_text, service_context]))
            score = self._apply_context_penalties(
                score=score,
                row_text=penalty_context,
                normalized_question=normalized_question,
            )

            score = self._apply_profile_score_adjustments(
                score=score,
                row_text=penalty_context,
                service_context=service_context,
                profile=profile,
                matched_terms=matched_terms,
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
        profile: ServiceDiscoveryProfile,
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

        candidates = self._apply_profile_candidate_priority(candidates, profile)
        candidates = self._filter_candidates_by_profile(candidates, profile)
        candidates = self._apply_profile_candidate_priority(candidates, profile)
        return candidates[:max_services]

    def _apply_profile_candidate_priority(
        self,
        candidates: list[ServiceDiscoveryCandidate],
        profile: ServiceDiscoveryProfile,
    ) -> list[ServiceDiscoveryCandidate]:
        if profile.code == "general" or not candidates:
            candidates.sort(key=lambda item: (-item.score, item.service_name_short or ""))
            return candidates

        prioritized: list[tuple[float, ServiceDiscoveryCandidate]] = []
        for candidate in candidates:
            bonus = self._profile_candidate_priority_bonus(candidate, profile)
            if bonus:
                candidate.score = round(candidate.score + bonus, 4)
            prioritized.append((bonus, candidate))

        prioritized.sort(key=lambda pair: (-pair[1].score, pair[1].service_name_short or ""))
        return [candidate for _, candidate in prioritized]

    @staticmethod
    def _profile_candidate_priority_bonus(
        candidate: ServiceDiscoveryCandidate,
        profile: ServiceDiscoveryProfile,
    ) -> float:
        context = normalize_text(
            " ".join(
                [
                    candidate.service_name_short or "",
                    candidate.service_name_full or "",
                    candidate.document_name or "",
                    candidate.original_filename or "",
                    " ".join(candidate.matched_terms),
                ]
            )
        )

        if profile.code == "school_need":
            # Прямой школьный вопрос должен в первую очередь показывать
            # профильную услугу, даже если таблица identifiers в этой услуге
            # описывает заявителей через многодетность/инвалидность родителей,
            # а не повторяет слово «школа».
            if ("ежегодн пособ" in context and "школьн" in context) or "пособие школьникам" in context:
                return 12.0
            if "ребенк школьн" in context or "ребенка школьн" in context or "школьн возраст" in context:
                return 9.0
            if "социальн контракт" in context:
                return 4.0
            if "материальн помощ" in context or "трудн жизненн" in context:
                return 3.4
            if "соцобслужив" in context:
                return 1.2

        if profile.code == "food_need":
            if "материальн помощ" in context or "трудн жизненн" in context:
                return 3.0
            if "социальн контракт" in context:
                return 2.2

<<<<<<< HEAD
        if profile.code in {"fuel_need", "solid_fuel_need"}:
=======
        if profile.code == "fuel_need":
>>>>>>> bba36515540dbe4eec46b473a736432fb4d55ceb
            if any(term in context for term in ("печн отоплен", "дров", "топлив")):
                return 4.5
            if "материальн помощ" in context or "трудн жизненн" in context:
                return 2.4

<<<<<<< HEAD
        if profile.code == "assistive_device_need":
            if any(term in context for term in ("тср", "техническ средств", "средств реабилитац", "кресло каталк", "кресло коляск", "коляск", "слухов аппарат")):
                return 10.0
            if any(term in context for term in ("инвалид", "ребенок инвалид", "детей инвалид")):
                return 2.2

        if profile.code == "dental_prosthesis_need":
            if any(term in context for term in ("зубопротез", "стоматологическ протез", "зубн протез", "протезирован")):
                return 10.0
            if any(term in context for term in ("ветеран труда", "вов", "таймыр")):
                return 2.0

        if profile.code == "public_transport_need":
            if any(term in context for term in ("соцкарт", "социальн карт", "проездн удостовер", "бесплатн проезд", "льготн проезд", "общественн транспорт", "автобус")):
                return 13.0
            if "проезд" in context and not any(term in context for term in ("к месту", "туда и обратно", "расход", "компенсац")):
                return 3.0

        if profile.code == "free_travel_need":
            if any(term in context for term in ("бесплатн проезд", "льготн проезд", "соцкарт", "социальн карт", "проездн удостовер", "общественн транспорт", "автобус")):
                return 10.0
            if "проезд" in context:
                return 4.0

=======
>>>>>>> bba36515540dbe4eec46b473a736432fb4d55ceb
        if profile.code == "emergency_fire":
            if any(term in context for term in ("утрат имущества", "вред здоров", "чрезвычайн", "чс")):
                return 3.8
            if "материальн помощ" in context or "трудн жизненн" in context:
                return 2.2

        return 0.0

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


    def _build_discovery_profile(
        self,
        normalized_question: str,
        signals: list[ApplicantSignal],
    ) -> ServiceDiscoveryProfile:
        """
        Select a narrowing profile for broad service discovery.

        This is intentionally conservative: profiles only narrow obviously
        practical crisis questions. General entitlement questions still use
        the regular broad behaviour.
        """
        signal_codes = {signal.code for signal in signals}
        question = normalize_text(normalized_question)

        if "emergency_victim" in signal_codes or "пожар" in question or "сгорел" in question:
            return ServiceDiscoveryProfile(
                code="emergency_fire",
                label="пожар, ЧС или утрата имущества",
                max_services=5,
                max_rows_per_service=2,
            )
        if "fuel_need" in signal_codes:
<<<<<<< HEAD
            if _is_solid_fuel_purchase_question(question):
                return ServiceDiscoveryProfile(
                    code="solid_fuel_need",
                    label="дрова, уголь или твёрдое топливо",
                    max_services=3,
                    max_rows_per_service=1,
                )
            return ServiceDiscoveryProfile(
                code="fuel_need",
                label="дрова, твёрдое топливо или отопление",
                max_services=4,
                max_rows_per_service=1,
            )
        if "assistive_device_need" in signal_codes:
            return ServiceDiscoveryProfile(
                code="assistive_device_need",
                label="технические средства реабилитации, коляска или слуховой аппарат",
                max_services=3,
                max_rows_per_service=1,
            )
        if "dental_prosthesis_need" in signal_codes:
            return ServiceDiscoveryProfile(
                code="dental_prosthesis_need",
                label="зубные или стоматологические протезы",
                max_services=3,
                max_rows_per_service=1,
            )
        if "free_travel_need" in signal_codes:
            if _is_public_transport_question(question):
                return ServiceDiscoveryProfile(
                    code="public_transport_need",
                    label="льготный или бесплатный проезд в общественном транспорте",
                    max_services=4,
                    max_rows_per_service=1,
                )
            return ServiceDiscoveryProfile(
                code="free_travel_need",
                label="льготный или бесплатный проезд",
=======
            return ServiceDiscoveryProfile(
                code="fuel_need",
                label="дрова, топливо или отопление",
>>>>>>> bba36515540dbe4eec46b473a736432fb4d55ceb
                max_services=4,
                max_rows_per_service=1,
            )
        if "food_need" in signal_codes:
            return ServiceDiscoveryProfile(
                code="food_need",
                label="еда, продукты или предметы первой необходимости",
                max_services=4,
                max_rows_per_service=1,
            )
        if "school_need" in signal_codes or ("школ" in question and "реб" in question):
            return ServiceDiscoveryProfile(
                code="school_need",
                label="подготовка ребёнка к школе",
                max_services=4,
                max_rows_per_service=1,
            )
        if "honorary_donor" in signal_codes:
            return ServiceDiscoveryProfile(
                code="honorary_donor",
                label="почётный донор",
                max_services=3,
                max_rows_per_service=2,
            )
        if signal_codes & {"vov_participant", "vov_disabled"}:
            return ServiceDiscoveryProfile(
                code="vov_status",
                label="статус участника или инвалида ВОВ",
                max_services=6,
                max_rows_per_service=2,
            )
        return ServiceDiscoveryProfile(code="general", label="общий подбор мер")

    @staticmethod
    def _apply_profile_score_adjustments(
        *,
        score: float,
        row_text: str,
        service_context: str,
        profile: ServiceDiscoveryProfile,
        matched_terms: list[str],
    ) -> float:
        if profile.code == "general":
            return score

        context = normalize_text(service_context)
        combined = normalize_text(" ".join([row_text, context]))

        preferred_terms: dict[str, tuple[str, ...]] = {
            "fuel_need": (
                "ремонт печн",
                "печн отоплен",
                "электропроводк",
                "дров",
<<<<<<< HEAD
                "уголь",
                "тверд топлив",
                "твердое топливо",
                "твёрдое топливо",
=======
>>>>>>> bba36515540dbe4eec46b473a736432fb4d55ceb
                "топлив",
                "отоплен",
                "материальн помощ",
                "трудн жизненн",
                "социальн контракт",
            ),
<<<<<<< HEAD
            "solid_fuel_need": (
                "дров",
                "уголь",
                "тверд топлив",
                "твердое топливо",
                "твёрдое топливо",
                "топлив",
                "материальн помощ",
                "трудн жизненн",
                "социальн контракт",
                "адресн материальн",
            ),
            "assistive_device_need": (
                "тср",
                "техническ средств",
                "средств реабилитац",
                "кресло каталк",
                "кресло коляск",
                "коляск",
                "слухов аппарат",
            ),
            "dental_prosthesis_need": (
                "зубопротез",
                "зубн протез",
                "стоматологическ протез",
                "стоматологическ",
                "протезирован",
            ),
            "public_transport_need": (
                "соцкарт",
                "социальн карт",
                "проездн удостовер",
                "бесплатн проезд",
                "льготн проезд",
                "общественн транспорт",
                "автобус",
            ),
            "free_travel_need": (
                "бесплатн проезд",
                "льготн проезд",
                "соцкарт",
                "социальн карт",
                "проездн удостовер",
                "общественн транспорт",
                "автобус",
                "проезд",
            ),
=======
>>>>>>> bba36515540dbe4eec46b473a736432fb4d55ceb
            "food_need": (
                "материальн помощ",
                "трудн жизненн",
                "адресн социальн помощ",
                "социальн контракт",
                "соцобслужив",
                "нуждающимся в соцобслужив",
            ),
            "school_need": (
                "школ",
                "школьн",
                "ежегодн пособ",
                "ребенк школьн",
                "социальн контракт",
                "материальн помощ",
                "трудн жизненн",
            ),
            "emergency_fire": (
                "чрезвычайн",
                "чс",
                "пострадавш",
                "утрат имущества",
                "вред здоров",
                "пожар",
                "материальн помощ",
                "трудн жизненн",
            ),
            "honorary_donor": (
                "донор",
                "почетн донор",
                "ежегодн денежн выплат",
                "льготн проезд",
            ),
            "vov_status": (
                "вов",
                "велик отечественн",
                "зубопротез",
                "жку",
                "проезд",
                "побед",
                "жилье ветеранам",
                "ремонт ветеранам",
                "тревожн кнопк",
            ),
        }
        negative_terms: dict[str, tuple[str, ...]] = {
            "fuel_need": (
                "жилье отдельным",
                "жилищно коммунальн",
                "жку",
                "погибш",
                "санаторн",
                "лагер",
                "зубопротез",
                "донор",
            ),
<<<<<<< HEAD
            "solid_fuel_need": (
                "соцобслужив",
                "материнск",
                "семейн капитал",
                "распоряжение",
                "ремонт печн",
                "электропроводк",
                "жилищно коммунальн",
                "жку",
                "жилье отдельным",
                "жилье отдельн",
                "жилищн обеспеч",
                "жилое помещ",
                "санаторн",
                "лагер",
                "зубопротез",
                "донор",
            ),
=======
>>>>>>> bba36515540dbe4eec46b473a736432fb4d55ceb
            "food_need": (
                "санаторн",
                "лагер",
                "путевк",
                "чрезвычайн",
                "погибш",
                "вред здоров",
                "утрат имущества",
                "жилищно коммунальн",
                "жку",
                "жилье отдельным",
            ),
            "school_need": (
                "санаторн",
                "лагер",
                "путевк",
                "жилищно коммунальн",
                "жку",
                "жилье отдельным",
                "чрезвычайн",
                "погибш",
                "зубопротез",
            ),
<<<<<<< HEAD
            "assistive_device_need": (
                "жилищно коммунальн",
                "жку",
                "санаторн",
                "лагер",
                "путевк",
                "зубопротез",
                "погреб",
                "почетн донор",
                "материнск",
                "семейн капитал",
                "тревожн кнопк",
                "свидетельств о праве",
            ),
            "dental_prosthesis_need": (
                "жилищно коммунальн",
                "жку",
                "санаторн",
                "лагер",
                "путевк",
                "дров",
                "топлив",
                "соцобслужив",
                "свидетельств о праве",
                "материнск",
                "едв",
                "ежемесячн денежн",
            ),
            "public_transport_need": (
                "передача транспорта",
                "транспортн средств в собственность",
                "месту отдыха",
                "к месту",
                "туда и обратно",
                "обследован",
                "лечени",
                "санаторн",
                "лагер",
                "путевк",
                "зубопротез",
                "погреб",
                "материнск",
                "топлив",
                "дров",
                "беременн",
                "безработн",
            ),
            "free_travel_need": (
                "передача транспорта",
                "транспортн средств в собственность",
                "месту отдыха",
                "санаторн",
                "лагер",
                "путевк",
                "зубопротез",
                "погреб",
                "материнск",
                "топлив",
                "дров",
            ),
=======
>>>>>>> bba36515540dbe4eec46b473a736432fb4d55ceb
            "emergency_fire": (
                "политическ репресс",
                "свидетельств о праве",
                "почетн донор",
                "зубопротез",
                "санаторн",
                "лагер",
            ),
        }

        hits = [term for term in preferred_terms.get(profile.code, ()) if term in context or term in combined]
        if hits:
            matched_terms.extend(hits)
            score += min(4.5, 1.4 + len(set(hits)) * 0.7)

        bad_hits = [term for term in negative_terms.get(profile.code, ()) if term in context]
        if bad_hits:
            score = max(0.0, score - min(5.5, 2.6 + len(set(bad_hits)) * 0.9))

        # A low-income child category inside sanatorium/camp services is not
        # enough evidence for an urgent food or school-expense request.
        if profile.code in {"food_need", "school_need"} and any(term in context for term in ("санаторн", "лагер", "путевк")):
            score = max(0.0, score - 4.0)

        # For a fire/house loss, generic social service may be relevant but
        # should not outrank direct emergency payments.
        if profile.code == "emergency_fire" and "соцобслужив" in context:
            score = max(0.0, score - 1.8)

        return score

    def _filter_candidates_by_profile(
        self,
        candidates: list[ServiceDiscoveryCandidate],
        profile: ServiceDiscoveryProfile,
    ) -> list[ServiceDiscoveryCandidate]:
        if profile.code == "general" or not candidates:
            return candidates

        filtered = [candidate for candidate in candidates if not self._is_candidate_noise_for_profile(candidate, profile)]
        if not filtered:
            return candidates

        # For practical crisis questions it is better to return fewer focused
        # directions than a long list with obviously unrelated benefits.
<<<<<<< HEAD
        if profile.code in {"fuel_need", "solid_fuel_need", "food_need", "school_need", "emergency_fire", "assistive_device_need", "dental_prosthesis_need", "public_transport_need", "free_travel_need"}:
=======
        if profile.code in {"fuel_need", "food_need", "school_need", "emergency_fire"}:
>>>>>>> bba36515540dbe4eec46b473a736432fb4d55ceb
            return filtered

        # For status questions keep the original list if filtering would hide
        # too much potentially relevant support.
        if len(filtered) >= 2:
            return filtered
        return candidates

    @staticmethod
    def _is_candidate_noise_for_profile(
        candidate: ServiceDiscoveryCandidate,
        profile: ServiceDiscoveryProfile,
    ) -> bool:
        service_context = normalize_text(
            " ".join(
                [
                    candidate.service_name_short or "",
                    candidate.service_name_full or "",
                    candidate.document_name or "",
                    candidate.original_filename or "",
                ]
            )
        )
        terms_context = normalize_text(" ".join(candidate.matched_terms))
        context = normalize_text(" ".join([service_context, terms_context]))

<<<<<<< HEAD
        if profile.code in {"assistive_device_need", "dental_prosthesis_need", "public_transport_need"}:
            # For practical need profiles, a candidate without the matched
            # practical signal is almost always a tail status/payment service
            # pulled in by a broad applicant category. Do not show it in a
            # compact demo answer.
            if not candidate.matched_signal_codes:
                return True

        if profile.code == "solid_fuel_need":
            # A request for help buying/obtaining firewood, coal or other solid
            # fuel should not be diluted by adjacent but different directions:
            # recognition as needing social services, maternity-capital disposal
            # for stove repair, utility benefits, sanatorium/camp benefits, etc.
            # If the user asks about repairing a stove/heating system, the broader
            # fuel_need profile is used instead.
            if any(
                term in service_context
                for term in (
                    "соцобслужив",
                    "нуждающимся в соцобслужив",
                    "материнск",
                    "семейн капитал",
                    "распоряжение",
                    "ремонт печн",
                    "электропроводк",
                    "жилищно коммунальн",
                    "жку",
                    "жилье отдельным",
                    "жилье отдельн",
                    "жилищн обеспеч",
                    "жилое помещ",
                    "санаторн",
                    "лагер",
                    "зубопротез",
                    "донор",
                )
            ):
                return True
            return False

        if profile.code == "fuel_need":
            if any(term in service_context for term in ("жилищно коммунальн", "жку", "жилье отдельным", "погибш", "зубопротез", "донор")):
                return not any(term in context for term in ("печн", "отоплен", "дров", "уголь", "тверд", "топлив", "материальн помощ", "трудн жизненн"))
=======
        if profile.code == "fuel_need":
            if any(term in service_context for term in ("жилищно коммунальн", "жку", "жилье отдельным", "погибш", "зубопротез", "донор")):
                return not any(term in context for term in ("печн", "отоплен", "дров", "топлив", "материальн помощ", "трудн жизненн"))
>>>>>>> bba36515540dbe4eec46b473a736432fb4d55ceb
            return False

        if profile.code == "food_need":
            return any(
                term in service_context
                for term in (
                    "санаторн",
                    "лагер",
                    "путевк",
                    "чрезвычайн",
                    "погибш",
                    "вред здоров",
                    "утрат имущества",
                    "жилищно коммунальн",
                    "жку",
                    "жилье отдельным",
                )
            )

        if profile.code == "school_need":
            # Жёстко отсекаем дошкольные и СВО-специфичные услуги.
            # Важно смотреть именно на название/документ услуги, а не на
            # matched_terms: туда могут попасть школьные слова из вопроса,
            # из-за чего шумная услуга ошибочно считается школьной.
            has_kindergarten_context = (
                ("дет" in service_context and "сад" in service_context)
                or "дошколь" in service_context
                or "не предоставлено место" in service_context
            )
            has_svo_context = (
                "сво" in service_context
                or "специальн военн" in service_context
                or "участник" in service_context and "военн" in service_context
            )
            if has_kindergarten_context or has_svo_context:
                return True

            if "школ" in service_context or "школь" in service_context:
                return False
            if "социальн контракт" in service_context or "материальн помощ" in service_context or "трудн жизненн" in service_context:
                return False
            # Общие детские/семейные услуги без школьной, адресной или
            # контрактной привязки для запроса «собрать ребёнка в школу»
            # лучше не показывать в демо-ответе.
            if any(term in service_context for term in ("ребен", "семь", "родител")):
                return True
            return any(
                term in service_context
                for term in (
                    "санаторн",
                    "лагер",
                    "путевк",
                    "жилищно коммунальн",
                    "жку",
                    "жилье отдельным",
                    "чрезвычайн",
                    "погибш",
                    "зубопротез",
                )
            )

<<<<<<< HEAD
        if profile.code == "assistive_device_need":
            if any(
                term in service_context
                for term in (
                    "жилищно коммунальн",
                    "жку",
                    "санаторн",
                    "лагер",
                    "путевк",
                    "зубопротез",
                    "погреб",
                    "почетн донор",
                    "материнск",
                    "семейн капитал",
                    "тревожн кнопк",
                    "свидетельств о праве",
                )
            ):
                return True
            return not any(
                term in service_context
                for term in (
                    "тср",
                    "техническ средств",
                    "средств реабилитац",
                    "кресло каталк",
                    "кресло коляск",
                    "коляск",
                    "слухов аппарат",
                )
            )

        if profile.code == "dental_prosthesis_need":
            if any(
                term in service_context
                for term in (
                    "жилищно коммунальн",
                    "жку",
                    "санаторн",
                    "лагер",
                    "путевк",
                    "дров",
                    "топлив",
                    "соцобслужив",
                    "материнск",
                    "свидетельств о праве",
                    "ежемесячн денежн",
                    "едв",
                )
            ):
                return True
            return not any(
                term in service_context
                for term in (
                    "зубопротез",
                    "зубн протез",
                    "стоматологическ протез",
                    "стоматологическ",
                    "протезирован",
                )
            )

        if profile.code == "public_transport_need":
            if any(
                term in service_context
                for term in (
                    "передача транспорта",
                    "транспортн средств в собственность",
                    "месту отдыха",
                    "к месту",
                    "туда и обратно",
                    "обследован",
                    "лечени",
                    "санаторн",
                    "лагер",
                    "путевк",
                    "зубопротез",
                    "погреб",
                    "материнск",
                    "дров",
                    "топлив",
                    "беременн",
                    "безработн",
                )
            ):
                return True
            return not any(
                term in service_context
                for term in (
                    "соцкарт",
                    "социальн карт",
                    "проездн удостовер",
                    "бесплатн проезд",
                    "льготн проезд",
                    "общественн транспорт",
                    "автобус",
                )
            )

        if profile.code == "free_travel_need":
            if any(
                term in service_context
                for term in (
                    "передача транспорта",
                    "транспортн средств в собственность",
                    "месту отдыха",
                    "санаторн",
                    "лагер",
                    "путевк",
                    "зубопротез",
                    "погреб",
                    "материнск",
                    "дров",
                    "топлив",
                )
            ):
                return True
            return not any(
                term in service_context
                for term in (
                    "бесплатн проезд",
                    "льготн проезд",
                    "соцкарт",
                    "социальн карт",
                    "проездн удостовер",
                    "общественн транспорт",
                    "автобус",
                    "проезд",
                )
            )

=======
>>>>>>> bba36515540dbe4eec46b473a736432fb4d55ceb
        if profile.code == "emergency_fire":
            return any(
                term in service_context
                for term in (
                    "политическ репресс",
                    "свидетельств о праве",
                    "почетн донор",
                    "зубопротез",
                    "санаторн",
                    "лагер",
                )
            )

        return False

    def _score_service_context_for_signals(
        self,
        *,
        service_context: str,
        normalized_question: str,
        signals: list[ApplicantSignal],
        matched_signal_codes: list[str],
        matched_signal_labels: list[str],
        matched_terms: list[str],
    ) -> float:
        """
        Даёт ограниченный бонус по названию услуги только для узких жизненных
        ситуаций. Основной подбор по-прежнему идёт по таблицам identifiers.

        Это нужно для вопросов вроде "нечего есть", "дрова", "сгорел дом",
        где конкретная цель помощи часто отражена в названии услуги/регламента,
        а не в названии категории заявителя.
        """
        bonus = 0.0

        for signal in signals:
            matched_terms_for_signal = self._service_context_terms_for_signal(
                signal_code=signal.code,
                service_context=service_context,
                normalized_question=normalized_question,
            )
            if not matched_terms_for_signal:
                continue

            if signal.code not in matched_signal_codes:
                matched_signal_codes.append(signal.code)
            if signal.label not in matched_signal_labels:
                matched_signal_labels.append(signal.label)
            matched_terms.extend(matched_terms_for_signal)
            bonus += self._service_context_bonus_for_signal(signal.code)

        return bonus

    @staticmethod
    def _service_context_terms_for_signal(
        *,
        signal_code: str,
        service_context: str,
        normalized_question: str,
    ) -> list[str]:
        context = normalize_text(service_context)
        question = normalize_text(normalized_question)

        signal_terms: dict[str, tuple[str, ...]] = {
            "emergency_victim": (
                "чрезвычайн",
                "чс",
                "пострадавш",
                "стихийн",
                "пожар",
                "бедств",
            ),
            "hardship": (
                "трудн жизненн",
                "тжс",
                "адресн материальн",
                "материальн помощ",
                "единовременн адресн",
            ),
            "food_need": (
                "трудн жизненн",
                "тжс",
                "адресн материальн",
                "материальн помощ",
                "малоимущ",
            ),
            "fuel_need": (
                "дров",
                "угол",
<<<<<<< HEAD
                "уголь",
                "тверд топлив",
                "твердое топливо",
                "твёрдое топливо",
=======
>>>>>>> bba36515540dbe4eec46b473a736432fb4d55ceb
                "топлив",
                "печн отоплен",
                "отоплен",
                "трудн жизненн",
                "тжс",
                "адресн материальн",
            ),
<<<<<<< HEAD
            "assistive_device_need": (
                "тср",
                "техническ средств",
                "средств реабилитац",
                "кресло каталк",
                "кресло коляск",
                "коляск",
                "слухов аппарат",
            ),
            "dental_prosthesis_need": (
                "зубопротез",
                "зубн протез",
                "стоматологическ протез",
                "протезирован",
            ),
            "public_transport_need": (
                "соцкарт",
                "социальн карт",
                "проездн удостовер",
                "бесплатн проезд",
                "льготн проезд",
                "общественн транспорт",
                "автобус",
            ),
            "free_travel_need": (
                "бесплатн проезд",
                "льготн проезд",
                "соцкарт",
                "социальн карт",
                "проездн удостовер",
                "общественн транспорт",
                "автобус",
                "проезд",
            ),
=======
>>>>>>> bba36515540dbe4eec46b473a736432fb4d55ceb
            "school_need": (
                "школьн возраст",
                "ребенк школьн",
                "ребенка школьн",
                "пособие на ребенк школьн",
                "ежегодн пособ",
                "школьн выплат",
                "компенсац проезд",
            ),
            "low_income": (
                "адресн социальн помощ",
                "адресн материальн",
                "малоимущ",
                "социальн помощ",
            ),
        }

        terms = signal_terms.get(signal_code)
        if not terms:
            return []

        matched = [term for term in terms if term in context]

        # Пожар в пользовательском вопросе должен особенно уверенно вести к
        # помощи при ЧС/пожаре или ТЖС, даже если в категории стоит общий текст.
        if signal_code == "emergency_victim" and "пожар" in question:
            if "трудн жизненн" in context or "адресн материальн" in context:
                matched.append("пожар / адресная материальная помощь")

        return _stable_unique(matched)

    @staticmethod
    def _service_context_bonus_for_signal(signal_code: str) -> float:
        bonuses = {
            "emergency_victim": 3.4,
            "hardship": 2.9,
            "food_need": 2.7,
            "fuel_need": 3.0,
<<<<<<< HEAD
            "assistive_device_need": 4.8,
            "dental_prosthesis_need": 4.8,
            "public_transport_need": 5.4,
            "free_travel_need": 4.4,
=======
>>>>>>> bba36515540dbe4eec46b473a736432fb4d55ceb
            "school_need": 3.1,
            "low_income": 1.8,
        }
        return bonuses.get(signal_code, 0.0)

    @staticmethod
    def _apply_context_penalties(
        *,
        score: float,
        row_text: str,
        normalized_question: str,
    ) -> float:
        territorial_terms = ("эвенк", "таймыр", "долгано", "ненец")
        question_mentions_territory = any(term in normalized_question for term in territorial_terms)
        row_mentions_territory = any(term in row_text for term in territorial_terms)
        if row_mentions_territory and not question_mentions_territory:
            score = max(0.0, score - 2.5)

        memorial_terms = ("могил", "памятник", "надгроб", "погреб", "похорон")
        question_mentions_memorial = any(term in normalized_question for term in memorial_terms)
        row_mentions_memorial = any(term in row_text for term in memorial_terms)
        if row_mentions_memorial and not question_mentions_memorial:
            score = max(0.0, score - 4.0)

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
        profile: ServiceDiscoveryProfile,
    ) -> str:
        signal_labels = ", ".join(signal.label for signal in signals)
        signal_codes = {signal.code for signal in signals}

        profile_intro = self._build_profile_intro(profile)
        if profile_intro:
            lines: list[str] = [profile_intro]
        elif signal_codes & {"hardship", "food_need", "fuel_need", "school_need", "emergency_victim"}:
            lines = [
                "По описанной ситуации я нашёл несколько направлений помощи, которые стоит проверить. "
                "Это не означает, что право уже подтверждено: условия нужно сверить по конкретной услуге и документам.",
            ]
        else:
            lines = [
                "По указанным признакам я нашёл несколько мер, которые могут быть релевантны. "
                "Это не означает, что право на них уже подтверждено: для точного вывода нужно проверить условия конкретной услуги по НПА.",
            ]

        if signal_labels:
            lines.append(f"Учтённые признаки из вопроса: {signal_labels}.")

        lines.append("Возможные направления для проверки:")

        max_fragments = 1 if profile.code != "general" else 2
        for index, candidate in enumerate(candidates, start=1):
            raw_service_name = candidate.service_name_short or candidate.service_name_full or candidate.service_key
            service_name = _shorten(self._clean_service_display_name(raw_service_name), limit=170)
            matched_labels = ", ".join(candidate.matched_signal_labels[:4])
            row_fragments = self._render_candidate_row_fragments(candidate, max_fragments=max_fragments)

            line = f"{index}. {service_name}"
            if matched_labels:
                line += f" — совпавшие признаки: {matched_labels}"
            if row_fragments:
                if profile.code == "general":
                    line += f". В таблице категорий заявителей найдено: {row_fragments}"
                else:
                    line += f". Категория для проверки: {row_fragments}"
            line += "."
            lines.append(line)

        lines.append(self._build_clarification_hint(signal_codes))
        return "\n".join(lines)


    @staticmethod
    def _build_profile_intro(profile: ServiceDiscoveryProfile) -> Optional[str]:
<<<<<<< HEAD
        if profile.code == "solid_fuel_need":
            return (
                "По вопросу о дровах, угле или другом твёрдом топливе в первую очередь стоит проверить адресную материальную помощь, "
                "помощь в трудной жизненной ситуации и социальный контракт. "
                "Если речь не о покупке топлива, а о ремонте печного отопления, это нужно уточнить отдельно."
            )
        if profile.code == "fuel_need":
            return (
                "По вопросу о дровах, твёрдом топливе или отоплении в первую очередь стоит проверить адресную материальную помощь, "
                "помощь в трудной жизненной ситуации, социальный контракт и специальные меры, связанные с печным отоплением. "
                "Право на выплату нужно подтверждать условиями конкретной услуги и документами."
            )
        if profile.code == "assistive_device_need":
            return (
                "По вопросу о коляске, слуховом аппарате или другом техническом средстве реабилитации нужно сначала определить категорию заявителя "
                "и основание получения средства. Я могу показать возможные направления, но право зависит от инвалидности, ИПРА, возраста заявителя и уже понесённых расходов."
            )
        if profile.code == "dental_prosthesis_need":
            return (
                "По вопросу о зубных или стоматологических протезах найдено несколько возможных направлений. "
                "Они зависят от категории заявителя: ветеран труда края, участник или инвалид ВОВ, территория проживания и другие условия."
            )
        if profile.code == "public_transport_need":
            return (
                "По вопросу о льготном или бесплатном проезде в общественном транспорте нужно уточнить льготную категорию "
                "и вид подтверждающего документа: социальная карта, проездное удостоверение или иная мера. "
                "Ниже приведены возможные направления для проверки, но право подтверждается условиями конкретной услуги."
            )
        if profile.code == "free_travel_need":
            return (
                "По вопросу о льготном или бесплатном проезде нужно уточнить льготную категорию и вид проезда. "
                "Ниже приведены возможные направления для проверки, но право подтверждается условиями конкретной услуги."
            )
=======
        if profile.code == "fuel_need":
            return (
                "По вопросу о дровах, топливе или отоплении в первую очередь стоит проверить адресную материальную помощь, "
                "помощь в трудной жизненной ситуации и специальные меры, связанные с печным отоплением. "
                "Право на выплату нужно подтверждать условиями конкретной услуги и документами."
            )
>>>>>>> bba36515540dbe4eec46b473a736432fb4d55ceb
        if profile.code == "food_need":
            return (
                "По вопросу о еде или предметах первой необходимости в первую очередь стоит проверить адресную материальную помощь, "
                "помощь в трудной жизненной ситуации, соцобслуживание и социальный контракт. "
                "Сам факт обращения ещё не подтверждает право на меру — нужны условия и документы по конкретной услуге."
            )
        if profile.code == "school_need":
            return (
                "По вопросу подготовки ребёнка к школе стоит проверить меры для семей с детьми, адресную помощь, "
                "социальный контракт и специальные школьные выплаты, если они применимы к семье. "
                "Точное право зависит от состава семьи, возраста ребёнка, дохода и подтверждённых статусов."
            )
        if profile.code == "emergency_fire":
            return (
                "По ситуации с пожаром, ЧС или утратой имущества в первую очередь стоит проверить материальную помощь пострадавшим, "
                "выплаты при утрате имущества или вреде здоровью, а также помощь в трудной жизненной ситуации. "
                "Точное право зависит от подтверждения события, места проживания и вида утраченного имущества."
            )
        return None

    @staticmethod
    def _clean_service_display_name(value: Optional[str]) -> str:
        text = " ".join(str(value or "").strip().split())
        replacements = {
            "инваоидам": "инвалидам",
            "Инваоидам": "Инвалидам",
            "Проверка выплата": "Выплата",
            "проверка выплата": "выплата",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    @staticmethod
    def _build_clarification_hint(signal_codes: set[str]) -> str:
        if signal_codes & {"hardship", "food_need", "fuel_need", "emergency_victim"}:
            return (
                "Что нужно уточнить для точного ответа: где проживает заявитель, что именно произошло, "
                "какая помощь уже предоставлялась в текущем году, есть ли подтверждающие документы и какой вид помощи нужен. "
                "После уточнения можно проверять конкретную услугу отдельно."
            )

<<<<<<< HEAD
        if "assistive_device_need" in signal_codes:
            return (
                "Что нужно уточнить: заявитель взрослый инвалид или ребёнок-инвалид, есть ли ИПРА, какое средство нужно, "
                "речь о получении средства или компенсации уже понесённых расходов, а также территория проживания. "
                "После этого можно проверить конкретную услугу и документы."
            )

        if "dental_prosthesis_need" in signal_codes:
            return (
                "Что нужно уточнить: категория заявителя, есть ли статус ветерана труда края, участника или инвалида ВОВ, "
                "проживает ли заявитель на Таймыре или в иной территории, и речь о компенсации расходов или оплате изготовления протезов."
            )

        if "free_travel_need" in signal_codes:
            return (
                "Что нужно уточнить: льготная категория заявителя, территория проживания, нужен бесплатный проезд, социальная карта, "
                "проездное удостоверение или компенсация расходов на конкретную поездку."
            )

=======
>>>>>>> bba36515540dbe4eec46b473a736432fb4d55ceb
        if signal_codes & {"school_need", "family_with_children", "large_family", "single_parent"}:
            return (
                "Что нужно уточнить для точного ответа: состав семьи, возраст детей, место жительства, доход, "
                "подтверждённые статусы семьи и цель обращения. После уточнения можно проверять каждую меру отдельно."
            )

        if signal_codes & {
            "honorary_donor",
            "vov_participant",
            "vov_disabled",
            "labor_veteran",
            "regional_labor_veteran",
            "combat_veteran",
            "rehabilitated",
        }:
            return (
                "Что нужно уточнить для точного ответа: подтверждённый статус, место жительства, возраст, "
                "цель обращения и не получалась ли аналогичная мера ранее. После уточнения можно проверять каждую меру отдельно."
            )

        return (
            "Что нужно уточнить для точного ответа: состав семьи, возраст детей, место жительства, доход, "
            "наличие подтверждённых статусов и цель обращения. После уточнения можно проверять каждую меру отдельно."
        )

    @staticmethod
    def _render_candidate_row_fragments(candidate: ServiceDiscoveryCandidate, *, max_fragments: int = 2) -> str:
        fragments: list[str] = []
        for row in candidate.matched_rows[:max_fragments]:
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
        return "; ".join(fragments[:max_fragments])

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


<<<<<<< HEAD

def _is_solid_fuel_purchase_question(normalized_question: str) -> bool:
    """Return True when the user asks about obtaining/buying fuel itself.

    This separates requests like "нужна помощь на дрова" from adjacent
    heating-system requests such as "ремонт печного отопления". Both are
    related to heating, but they should not produce the same list of possible
    measures.
    """
    question = normalize_text(normalized_question)
    if not question:
        return False

    solid_fuel_terms = ("дров", "угол", "уголь", "тверд топлив", "твердое топливо", "твёрдое топливо", "топлив")
    if not any(term in question for term in solid_fuel_terms):
        return False

    repair_terms = ("ремонт", "печн", "печь", "электропровод", "проводк", "котел", "котёл")
    if any(term in question for term in repair_terms) and not any(term in question for term in ("куп", "приобр", "достав", "нужн", "помощь на", "денег на")):
        return False

    return True



def _is_public_transport_question(normalized_question: str) -> bool:
    """Return True for local/public-transport wording.

    This separates requests like "бесплатный проезд в автобусе" from
    compensation for a specific trip to treatment, sanatorium, burial, etc.
    """
    question = normalize_text(normalized_question)
    return any(
        term in question
        for term in (
            "автобус",
            "общественн транспорт",
            "социальн карт",
            "соцкарт",
            "проездн удостовер",
            "льготн проезд",
            "бесплатн проезд",
        )
    )

=======
>>>>>>> bba36515540dbe4eec46b473a736432fb4d55ceb
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
