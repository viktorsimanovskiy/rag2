# ============================================================
# File: app/services/generation/generation_pipeline.py
# Purpose:
#   Convert structured retrieval evidence into a grounded answer
#   suitable for persistence and delivery to the user.
#
# Responsibilities:
#   - load evidence objects selected by retrieval
#   - choose answer mode
#   - build answer plan
#   - compose grounded answer
#   - construct citations
#   - build evidence items for answer_event persistence
#   - run deterministic / semantic validation hooks
#
# Design principles:
#   - grounded-by-evidence only
#   - conservative answering
#   - safe_no_answer is first-class mode
#   - answer text and evidence package stay tightly linked
# ============================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.documents import (
    DocumentBlock,
    DocumentRegistry,
    DocumentTable,
    DocumentTableRow,
    LegalFact,
)
from app.db.models.enums import AnswerModeEnum, QuestionIntentEnum, ValidationStatusEnum
from app.services.feedback.feedback_service import EvidenceItemInput
from app.services.retrieval.retrieval_orchestrator import EvidencePackage, RetrievedCandidate
from app.services.generation.table_documents_answer_builder import TableDocumentsAnswerBuilder

logger = logging.getLogger(__name__)


# ============================================================
# Placeholder protocols
# ============================================================

class DeterministicAnswerValidatorProtocol:
    async def validate(self, payload: "DeterministicValidationInput") -> "DeterministicValidationResult":
        raise NotImplementedError


class SemanticAnswerValidatorProtocol:
    async def validate(self, payload: "SemanticValidationInput") -> "SemanticValidationResult":
        raise NotImplementedError


# ============================================================
# DTOs
# ============================================================

@dataclass(slots=True)
class GenerationRequest:
    """
    Input from answer orchestrator into generation pipeline.
    """
    session_id: UUID
    question_event_id: UUID
    channel_code: Any

    question_text_raw: str
    question_text_normalized: str
    language_code: str

    intent_type: QuestionIntentEnum
    measure_code: Optional[str]
    subject_category_code: Optional[str]

    routing_payload_json: dict[str, Any] = field(default_factory=dict)
    query_constraints_json: dict[str, Any] = field(default_factory=dict)
    request_metadata_json: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CitationItem:
    """
    Display-oriented citation object.
    """
    source_type: str
    document_id: UUID
    source_id: UUID

    display_label: str
    citation_text: str
    document_name: Optional[str] = None
    download_url: Optional[str] = None

    metadata_json: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AnswerPlan:
    """
    Intermediate reasoning artifact.

    Not chain-of-thought; only a safe structured plan
    for answer assembly.
    """
    answer_mode: AnswerModeEnum
    strategy_code: str

    primary_candidates: list[RetrievedCandidate] = field(default_factory=list)
    supporting_candidates: list[RetrievedCandidate] = field(default_factory=list)

    direct_answer_points: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    no_answer_reason_code: Optional[str] = None


@dataclass(slots=True)
class DeterministicValidationInput:
    question_text: str
    answer_text: str
    evidence_package: EvidencePackage
    citations_json: list[dict[str, Any]]


@dataclass(slots=True)
class DeterministicValidationResult:
    passed: bool
    issues: list[str] = field(default_factory=list)
    score: Optional[float] = None


@dataclass(slots=True)
class SemanticValidationInput:
    question_text: str
    answer_text: str
    evidence_package: EvidencePackage
    answer_mode: AnswerModeEnum


@dataclass(slots=True)
class SemanticValidationResult:
    passed: bool
    issues: list[str] = field(default_factory=list)
    score: Optional[float] = None


@dataclass(slots=True)
class GenerationResult:
    """
    Output expected by answer_orchestrator.
    """
    answer_mode: AnswerModeEnum
    answer_text: str
    answer_text_short: Optional[str]

    confidence_score: Optional[float]
    trust_score_at_generation: Optional[float]

    validation_status: ValidationStatusEnum
    deterministic_validation_passed: bool
    semantic_validation_passed: bool

    reuse_allowed: bool
    reuse_policy_version: Optional[str]

    citations_json: list[dict[str, Any]] = field(default_factory=list)
    answer_payload_json: dict[str, Any] = field(default_factory=dict)
    reuse_decision_payload_json: dict[str, Any] = field(default_factory=dict)

    evidence_items: list[EvidenceItemInput] = field(default_factory=list)

    generation_model_name: Optional[str] = None
    generation_prompt_version: Optional[str] = None
    pipeline_version: Optional[str] = None


# ============================================================
# Pipeline
# ============================================================

class GenerationPipeline:
    """
    Grounded answer generation pipeline driven by structured evidence.
    """

    def __init__(
        self,
        db: AsyncSession,
        *,
        deterministic_validator: Optional[DeterministicAnswerValidatorProtocol] = None,
        semantic_validator: Optional[SemanticAnswerValidatorProtocol] = None,
    ) -> None:
        self.db = db
        self.deterministic_validator = deterministic_validator
        self.semantic_validator = semantic_validator
        self.table_documents_answer_builder = TableDocumentsAnswerBuilder()

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    async def generate_answer(
        self,
        payload: GenerationRequest,
        evidence_package: Optional[EvidencePackage] = None,
    ) -> GenerationResult:
        """
        Main generation flow.

        Expected usage:
        - retrieval happens before this layer
        - evidence_package is passed in from retrieval layer

        If evidence_package is empty or weak, safe_no_answer is returned.
        """
        self._validate_input(payload)

        if evidence_package is None or not evidence_package.selected_candidates:
            return self._build_safe_no_answer_result(
                payload=payload,
                reason_code="no_evidence_package",
                evidence_package=evidence_package,
            )

        evidence_quality = str(
            (evidence_package.debug_payload_json or {}).get("evidence_quality") or ""
        ).strip().lower()
        guard_reason = str(
            (evidence_package.debug_payload_json or {}).get("guard_reason") or ""
        ).strip()

        if evidence_quality == "insufficient":
            return self._build_safe_no_answer_result(
                payload=payload,
                reason_code="retrieval_insufficient",
                evidence_package=evidence_package,
            )

        hydrated_objects = await self._hydrate_evidence_objects(evidence_package)
        plan = self._build_answer_plan(
            payload=payload,
            evidence_package=evidence_package,
            hydrated_objects=hydrated_objects,
        )

        if (
            plan.answer_mode == AnswerModeEnum.SAFE_NO_ANSWER
            and evidence_quality == "weak"
            and plan.no_answer_reason_code == "insufficient_grounded_points"
        ):
            logger.info(
                "Weak retrieval evidence forced SAFE_NO_ANSWER",
                extra={
                    "question_event_id": str(payload.question_event_id),
                    "guard_reason": guard_reason,
                },
            )

        documents_answer_payload = self._prepare_documents_table_answer(
            payload=payload,
            evidence_package=evidence_package,
        )

        answer_text = self._compose_answer_text(
            payload=payload,
            plan=plan,
            evidence_package=evidence_package,
            hydrated_objects=hydrated_objects,
            documents_answer_payload=documents_answer_payload,
        )

        answer_text_short = self._build_short_answer(
            answer_text=answer_text,
            answer_mode=plan.answer_mode,
        )

        citations = self._build_citations(
            plan=plan,
            hydrated_objects=hydrated_objects,
        )

        evidence_items = self._build_evidence_items(
            candidates=plan.primary_candidates + plan.supporting_candidates
        )

        det_validation = await self._run_deterministic_validation(
            question_text=payload.question_text_raw,
            answer_text=answer_text,
            evidence_package=evidence_package,
            citations_json=citations,
        )

        sem_validation = await self._run_semantic_validation(
            question_text=payload.question_text_raw,
            answer_text=answer_text,
            evidence_package=evidence_package,
            answer_mode=plan.answer_mode,
        )

        validation_status = self._derive_validation_status(
            deterministic_passed=det_validation.passed,
            semantic_passed=sem_validation.passed,
        )

        confidence_score = self._calculate_confidence_score(
            plan=plan,
            evidence_package=evidence_package,
            det_validation=det_validation,
            sem_validation=sem_validation,
        )

        trust_score = self._calculate_trust_score(
            answer_mode=plan.answer_mode,
            confidence_score=confidence_score,
            det_validation=det_validation,
            sem_validation=sem_validation,
            evidence_package=evidence_package,
        )

        reuse_allowed = self._decide_reuse_allowed(
            payload=payload,
            plan=plan,
            validation_status=validation_status,
            confidence_score=confidence_score,
        )

        return GenerationResult(
            answer_mode=plan.answer_mode,
            answer_text=answer_text,
            answer_text_short=answer_text_short,
            confidence_score=confidence_score,
            trust_score_at_generation=trust_score,
            validation_status=validation_status,
            deterministic_validation_passed=det_validation.passed,
            semantic_validation_passed=sem_validation.passed,
            reuse_allowed=reuse_allowed,
            reuse_policy_version="reuse_gate_v1",
            citations_json=citations,
            answer_payload_json={
                "strategy_code": plan.strategy_code,
                "plan_warnings": plan.warnings,
                "deterministic_validation_issues": det_validation.issues,
                "semantic_validation_issues": sem_validation.issues,
                "evidence_metrics": evidence_package.metrics_json,
                "evidence_quality": evidence_quality or None,
                "guard_reason": guard_reason or None,
                "documents_builder_debug": documents_answer_payload.get("debug"),
            },
            reuse_decision_payload_json={
                "reuse_allowed": reuse_allowed,
                "reuse_policy_version": "reuse_gate_v1",
                "confidence_score": confidence_score,
                "evidence_quality": evidence_quality or None,
            },
            evidence_items=evidence_items,
            generation_model_name=None,
            generation_prompt_version="grounded_template_v1",
            pipeline_version="generation_pipeline_v1",
        )

    def _extract_submission_channel(
        self,
        payload: GenerationRequest,
        evidence_package: EvidencePackage,
    ) -> Optional[str]:
        constraints = payload.query_constraints_json or {}
        if isinstance(constraints, dict):
            submission_channel = constraints.get("submission_channel")
            if isinstance(submission_channel, str) and submission_channel.strip():
                return submission_channel.strip().lower()

        debug_query_bundle = (evidence_package.debug_payload_json or {}).get("query_bundle") or {}
        if isinstance(debug_query_bundle, dict):
            submission_channel = debug_query_bundle.get("submission_channel")
            if isinstance(submission_channel, str) and submission_channel.strip():
                return submission_channel.strip().lower()

        return None

    def _prepare_documents_table_answer(
            self,
            *,
            payload: GenerationRequest,
            evidence_package: EvidencePackage,
        ) -> dict[str, Any]:
            if payload.intent_type != QuestionIntentEnum.DOCUMENTS_QUESTION:
                return {
                    "answer_text": None,
                    "debug": {
                        "skipped": True,
                        "reason": "not_documents_question",
                    },
                }

            submission_channel = self._extract_submission_channel(
                payload=payload,
                evidence_package=evidence_package,
            )

            result = self.table_documents_answer_builder.build(
                candidates=evidence_package.selected_candidates or [],
                submission_channel=submission_channel,
            )

            answer_text = None
            if result.can_answer:
                answer_text = self.table_documents_answer_builder.render_text(
                    result=result,
                    submission_channel=submission_channel,
                )

            debug_payload = result.debug_payload(
                submission_channel=submission_channel,
            )
            debug_payload["can_answer"] = result.can_answer

            return {
                "answer_text": answer_text,
                "debug": debug_payload,
            }

    # --------------------------------------------------------
    # Evidence hydration
    # --------------------------------------------------------

    async def _hydrate_evidence_objects(
        self,
        evidence_package: EvidencePackage,
    ) -> dict[str, dict[UUID, Any]]:
        """
        Load full ORM objects for selected evidence ids.
        """
        return {
            "documents": await self._load_documents(evidence_package.selected_document_ids),
            "facts": await self._load_legal_facts(evidence_package.selected_fact_ids),
            "tables": await self._load_tables(evidence_package.selected_table_ids),
            "rows": await self._load_rows(evidence_package.selected_row_ids),
            "blocks": await self._load_blocks(evidence_package.selected_block_ids),
        }

    async def _load_documents(self, ids: list[UUID]) -> dict[UUID, Any]:
        if not ids:
            return {}
        stmt: Select[Any] = select(DocumentRegistry).where(DocumentRegistry.document_id.in_(ids))
        result = await self.db.execute(stmt)
        rows = result.scalars().all()
        return {row.document_id: row for row in rows}

    async def _load_legal_facts(self, ids: list[UUID]) -> dict[UUID, Any]:
        if not ids:
            return {}
        stmt: Select[Any] = select(LegalFact).where(LegalFact.fact_id.in_(ids))
        result = await self.db.execute(stmt)
        rows = result.scalars().all()
        return {row.fact_id: row for row in rows}

    async def _load_tables(self, ids: list[UUID]) -> dict[UUID, Any]:
        if not ids:
            return {}
        stmt: Select[Any] = select(DocumentTable).where(DocumentTable.table_id.in_(ids))
        result = await self.db.execute(stmt)
        rows = result.scalars().all()
        return {row.table_id: row for row in rows}

    async def _load_rows(self, ids: list[UUID]) -> dict[UUID, Any]:
        if not ids:
            return {}
        stmt: Select[Any] = select(DocumentTableRow).where(DocumentTableRow.row_id.in_(ids))
        result = await self.db.execute(stmt)
        rows = result.scalars().all()
        return {row.row_id: row for row in rows}

    async def _load_blocks(self, ids: list[UUID]) -> dict[UUID, Any]:
        if not ids:
            return {}
        stmt: Select[Any] = select(DocumentBlock).where(DocumentBlock.block_id.in_(ids))
        result = await self.db.execute(stmt)
        rows = result.scalars().all()
        return {row.block_id: row for row in rows}

    # --------------------------------------------------------
    # Answer planning
    # --------------------------------------------------------

    def _build_answer_plan(
        self,
        *,
        payload: GenerationRequest,
        evidence_package: EvidencePackage,
        hydrated_objects: dict[str, dict[UUID, Any]],
    ) -> AnswerPlan:
        candidates = evidence_package.selected_candidates or []

        if not candidates:
            return AnswerPlan(
                answer_mode=AnswerModeEnum.SAFE_NO_ANSWER,
                strategy_code=evidence_package.strategy_code,
                no_answer_reason_code="no_candidates",
                warnings=["Нет отобранных evidence-кандидатов."],
            )

        primary_candidates = candidates[: min(5, len(candidates))]
        supporting_candidates = candidates[min(5, len(candidates)): min(10, len(candidates))]

        answer_mode = self._select_answer_mode(
            payload=payload,
            primary_candidates=primary_candidates,
        )

        direct_answer_points = self._extract_direct_answer_points(
            payload=payload,
            primary_candidates=primary_candidates,
            hydrated_objects=hydrated_objects,
        )

        if answer_mode != AnswerModeEnum.SAFE_NO_ANSWER and not direct_answer_points:
            return AnswerPlan(
                answer_mode=AnswerModeEnum.SAFE_NO_ANSWER,
                strategy_code=evidence_package.strategy_code,
                primary_candidates=primary_candidates,
                supporting_candidates=supporting_candidates,
                no_answer_reason_code="insufficient_grounded_points",
                warnings=["Недостаточно надёжных grounded points для ответа."],
            )

        return AnswerPlan(
            answer_mode=answer_mode,
            strategy_code=evidence_package.strategy_code,
            primary_candidates=primary_candidates,
            supporting_candidates=supporting_candidates,
            direct_answer_points=direct_answer_points,
            warnings=[],
        )

    def _select_answer_mode(
        self,
        *,
        payload: GenerationRequest,
        primary_candidates: list[RetrievedCandidate],
    ) -> AnswerModeEnum:
        """
        Conservative answer mode selection.
        """
        if not primary_candidates:
            return AnswerModeEnum.SAFE_NO_ANSWER

        top_types = {c.source_type for c in primary_candidates[:3]}

        if payload.intent_type in {
            QuestionIntentEnum.DOCUMENTS_QUESTION,
            QuestionIntentEnum.DEADLINE_QUESTION,
            QuestionIntentEnum.FORM_QUESTION,
        } and ("legal_fact" in top_types or "table_row" in top_types or "table" in top_types):
            return AnswerModeEnum.DIRECT_STRUCTURED

        if payload.intent_type in {
            QuestionIntentEnum.PROCEDURE_QUESTION,
            QuestionIntentEnum.APPEAL_QUESTION,
            QuestionIntentEnum.MIXED_QUESTION,
            QuestionIntentEnum.REJECTION_QUESTION,
            QuestionIntentEnum.ELIGIBILITY_QUESTION,
        }:
            return AnswerModeEnum.GROUNDED_NARRATIVE

        return AnswerModeEnum.GROUNDED_NARRATIVE

    def _extract_direct_answer_points(
        self,
        *,
        payload: GenerationRequest,
        primary_candidates: list[RetrievedCandidate],
        hydrated_objects: dict[str, dict[UUID, Any]],
    ) -> list[str]:
        """
        Extract short grounded facts/points from evidence objects.
        """
        points: list[str] = []

        for candidate in primary_candidates:
            point = self._candidate_to_answer_point(
                candidate=candidate,
                hydrated_objects=hydrated_objects,
                intent_type=payload.intent_type,
            )
            if point and point not in points:
                points.append(point)

        return points[:6]

    def _candidate_to_answer_point(
        self,
        *,
        candidate: RetrievedCandidate,
        hydrated_objects: dict[str, dict[UUID, Any]],
        intent_type: QuestionIntentEnum,
    ) -> Optional[str]:
        if candidate.source_type == "legal_fact":
            fact = hydrated_objects["facts"].get(candidate.source_id)
            if fact is None:
                return candidate.snippet

            fact_type = str(getattr(fact, "fact_type", "") or "").strip()
            value_json = getattr(fact, "value_json", {}) or {}
            condition_json = getattr(fact, "condition_json", {}) or {}
            validity_note = str(getattr(fact, "validity_note", "") or "").strip()

            if fact_type in {"required_documents", "documents"}:
                docs = value_json.get("documents") if isinstance(value_json, dict) else None
                if isinstance(docs, list) and docs:
                    return "Необходимые документы: " + ", ".join(str(x) for x in docs[:8]) + "."
                if validity_note:
                    return validity_note

            if fact_type in {"deadline", "review_period", "payment_deadline"}:
                if validity_note:
                    return validity_note
                if isinstance(value_json, dict) and value_json:
                    return "Срок: " + self._compact_json_dict(value_json) + "."

            if fact_type in {"rejection_reason", "grounds_for_refusal"}:
                if validity_note:
                    return "Основания отказа: " + validity_note.rstrip(".") + "."
                if isinstance(value_json, dict) and value_json:
                    return "Основания отказа: " + self._compact_json_dict(value_json) + "."

            if fact_type in {"eligibility", "condition", "recipient_category"}:
                fragments: list[str] = []
                if isinstance(condition_json, dict) and condition_json:
                    fragments.append(self._compact_json_dict(condition_json))
                if validity_note:
                    fragments.append(validity_note)
                if fragments:
                    return "Условия предоставления: " + "; ".join(fragments) + "."

            if validity_note:
                return validity_note.rstrip(".") + "."

            if isinstance(value_json, dict) and value_json:
                return self._compact_json_dict(value_json) + "."

        elif candidate.source_type == "table_row":
            row = hydrated_objects["rows"].get(candidate.source_id)
            if row is None:
                return self._normalize_sentence(candidate.snippet)

            row_summary = str(getattr(row, "row_summary", "") or "").strip()
            if row_summary:
                return self._normalize_sentence(row_summary)

            normalized_row_json = getattr(row, "normalized_row_json", {}) or {}
            if isinstance(normalized_row_json, dict) and normalized_row_json:
                return self._compact_json_dict(normalized_row_json) + "."

        elif candidate.source_type == "table":
            table = hydrated_objects["tables"].get(candidate.source_id)
            if table is None:
                return self._normalize_sentence(candidate.snippet)

            summary = str(getattr(table, "summary", "") or "").strip()
            title = str(getattr(table, "table_title", "") or "").strip()

            if summary:
                return self._normalize_sentence(summary)
            if title:
                return f"Найдена релевантная таблица: {title}."

        elif candidate.source_type == "block":
            block = hydrated_objects["blocks"].get(candidate.source_id)
            if block is None:
                return self._normalize_sentence(candidate.snippet)

            content = str(getattr(block, "content_clean", "") or "").strip()
            if content:
                return self._normalize_sentence(self._shorten_text(content, limit=260))

        return None

    # --------------------------------------------------------
    # Text composition
    # --------------------------------------------------------

    def _compose_answer_text(
            self,
            *,
            payload: GenerationRequest,
            plan: AnswerPlan,
            evidence_package: EvidencePackage,
            hydrated_objects: dict[str, dict[UUID, Any]],
            documents_answer_payload: Optional[dict[str, Any]] = None,
        ) -> str:
            # Для documents-question сначала всегда пробуем deterministic path.
            if payload.intent_type == QuestionIntentEnum.DOCUMENTS_QUESTION:
                deterministic_text = None
                if isinstance(documents_answer_payload, dict):
                    deterministic_text = documents_answer_payload.get("answer_text")

                if deterministic_text:
                    return deterministic_text

            if plan.answer_mode == AnswerModeEnum.SAFE_NO_ANSWER:
                return self._compose_safe_no_answer_text(
                    payload=payload,
                    plan=plan,
                )

            if plan.answer_mode == AnswerModeEnum.DIRECT_STRUCTURED:
                return self._compose_direct_structured_answer(
                    payload=payload,
                    plan=plan,
                )

            return self._compose_grounded_narrative_answer(
                payload=payload,
                plan=plan,
            )

    def _compose_direct_structured_answer(
        self,
        *,
        payload: GenerationRequest,
        plan: AnswerPlan,
    ) -> str:
        intro = self._select_intro_by_intent(payload.intent_type)
        lines = [intro]

        for point in plan.direct_answer_points[:5]:
            lines.append(f"— {point}")

        lines.append("Ниже приведены источники, на которых основан ответ.")
        return "\n".join(lines)

    def _compose_grounded_narrative_answer(
        self,
        *,
        payload: GenerationRequest,
        plan: AnswerPlan,
    ) -> str:
        intro = self._select_intro_by_intent(payload.intent_type)
        body = " ".join(
            self._normalize_sentence(point)
            for point in plan.direct_answer_points[:4]
            if point
        )

        if not body:
            return self._compose_safe_no_answer_text(
                payload=payload,
                plan=AnswerPlan(
                    answer_mode=AnswerModeEnum.SAFE_NO_ANSWER,
                    strategy_code=plan.strategy_code,
                    no_answer_reason_code="empty_narrative_body",
                    warnings=["Не удалось собрать grounded narrative body."],
                ),
            )

        closing = "Ответ сформирован только по найденным актуальным источникам."
        return f"{intro} {body} {closing}".strip()

    def _compose_safe_no_answer_text(
        self,
        *,
        payload: GenerationRequest,
        plan: AnswerPlan,
    ) -> str:
        base = (
            "Я не могу надёжно ответить на этот вопрос только по найденным источникам. "
            "В доступной выборке недостаточно подтверждённых данных для точного ответа."
        )

        hint = self._safe_no_answer_hint(payload.intent_type)
        if hint:
            return f"{base} {hint}"
        return base

    def _select_intro_by_intent(
        self,
        intent_type: QuestionIntentEnum,
    ) -> str:
        if intent_type == QuestionIntentEnum.DOCUMENTS_QUESTION:
            return "По найденным источникам можно выделить следующий перечень."
        if intent_type == QuestionIntentEnum.DEADLINE_QUESTION:
            return "По найденным источникам установлены следующие сроки."
        if intent_type == QuestionIntentEnum.REJECTION_QUESTION:
            return "По найденным источникам основания отказа сформулированы следующим образом."
        if intent_type == QuestionIntentEnum.ELIGIBILITY_QUESTION:
            return "По найденным источникам применяются следующие условия."
        if intent_type == QuestionIntentEnum.PROCEDURE_QUESTION:
            return "По найденным источникам порядок действий выглядит так."
        if intent_type == QuestionIntentEnum.FORM_QUESTION:
            return "По найденным источникам релевантны следующие сведения по форме или таблице."
        return "По найденным источникам можно сообщить следующее."

    def _safe_no_answer_hint(
        self,
        intent_type: QuestionIntentEnum,
    ) -> Optional[str]:
        if intent_type in {
            QuestionIntentEnum.DOCUMENTS_QUESTION,
            QuestionIntentEnum.FORM_QUESTION,
        }:
            return "Для такого вопроса желательно уточнить меру поддержки или конкретную ситуацию заявителя."
        if intent_type in {
            QuestionIntentEnum.ELIGIBILITY_QUESTION,
            QuestionIntentEnum.AMBIGUOUS_QUESTION,
        }:
            return "Для точного ответа обычно нужны уточняющие условия или категория получателя."
        return None

    # --------------------------------------------------------
    # Citations
    # --------------------------------------------------------

    def _build_citations(
        self,
        *,
        plan: AnswerPlan,
        hydrated_objects: dict[str, dict[UUID, Any]],
    ) -> list[dict[str, Any]]:
        citations: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, UUID]] = set()

        for candidate in plan.primary_candidates[:6]:
            key = (candidate.source_type, candidate.source_id)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            citation = self._candidate_to_citation(candidate, hydrated_objects)
            if citation:
                citations.append(citation)

        return citations

    def _candidate_to_citation(
        self,
        candidate: RetrievedCandidate,
        hydrated_objects: dict[str, dict[UUID, Any]],
    ) -> Optional[dict[str, Any]]:
        document = hydrated_objects["documents"].get(candidate.document_id)
        document_name = getattr(document, "document_name", None) if document else candidate.document_name
        download_url = None
        if document is not None:
            publication_payload = getattr(document, "publication_payload_json", {}) or {}
            download_url = publication_payload.get("download_url")

        citation_text = self._build_citation_text(candidate, hydrated_objects)

        return {
            "source_type": candidate.source_type,
            "source_id": str(candidate.source_id),
            "document_id": str(candidate.document_id),
            "document_name": document_name,
            "display_label": self._build_display_label(candidate, hydrated_objects, document_name),
            "citation_text": citation_text,
            "download_url": download_url,
            "metadata_json": {
                "score": candidate.score,
                "rerank_score": candidate.rerank_score,
            },
        }

    def _build_display_label(
        self,
        candidate: RetrievedCandidate,
        hydrated_objects: dict[str, dict[UUID, Any]],
        document_name: Optional[str],
    ) -> str:
        base = document_name or "Документ"

        if candidate.source_type == "legal_fact":
            fact = hydrated_objects["facts"].get(candidate.source_id)
            fact_type = str(getattr(fact, "fact_type", "") or "").strip()
            if fact_type:
                return f"{base} — факт: {fact_type}"

        if candidate.source_type == "table":
            table = hydrated_objects["tables"].get(candidate.source_id)
            table_title = str(getattr(table, "table_title", "") or "").strip() if table else ""
            if table_title:
                return f"{base} — таблица: {table_title}"

        if candidate.source_type == "table_row":
            return f"{base} — строка таблицы"

        if candidate.source_type == "block":
            block = hydrated_objects["blocks"].get(candidate.source_id)
            clause_number = str(getattr(block, "clause_number", "") or "").strip() if block else ""
            if clause_number:
                return f"{base} — пункт {clause_number}"
            return f"{base} — фрагмент текста"

        return base

    def _build_citation_text(
        self,
        candidate: RetrievedCandidate,
        hydrated_objects: dict[str, dict[UUID, Any]],
    ) -> str:
        if candidate.source_type == "legal_fact":
            fact = hydrated_objects["facts"].get(candidate.source_id)
            if fact is not None:
                validity_note = str(getattr(fact, "validity_note", "") or "").strip()
                if validity_note:
                    return self._shorten_text(validity_note, limit=220) or "Факт"

        if candidate.source_type == "table":
            table = hydrated_objects["tables"].get(candidate.source_id)
            if table is not None:
                title = str(getattr(table, "table_title", "") or "").strip()
                summary = str(getattr(table, "summary", "") or "").strip()
                if title and summary:
                    return f"{title}: {self._shorten_text(summary, limit=180)}"
                if title:
                    return title

        if candidate.source_type == "table_row":
            row = hydrated_objects["rows"].get(candidate.source_id)
            if row is not None:
                row_summary = str(getattr(row, "row_summary", "") or "").strip()
                if row_summary:
                    return self._shorten_text(row_summary, limit=220) or "Строка таблицы"

        if candidate.source_type == "block":
            block = hydrated_objects["blocks"].get(candidate.source_id)
            if block is not None:
                content = str(getattr(block, "content_clean", "") or "").strip()
                if content:
                    return self._shorten_text(content, limit=220) or "Фрагмент текста"

        return candidate.snippet or "Источник"

    # --------------------------------------------------------
    # Evidence items for answer_event
    # --------------------------------------------------------

    def _build_evidence_items(
        self,
        *,
        candidates: list[RetrievedCandidate],
    ) -> list[EvidenceItemInput]:
        items: list[EvidenceItemInput] = []
        seen_keys: set[tuple[str, UUID]] = set()

        for candidate in candidates:
            key = (candidate.source_type, candidate.source_id)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            kwargs: dict[str, Any] = {
                "citation_json": candidate.citation_json or {},
                "role_code": "primary_evidence" if len(items) < 5 else "supporting_evidence",
            }

            if candidate.source_type == "legal_fact":
                kwargs["evidence_item_type"] = "legal_fact"
                kwargs["legal_fact_id"] = candidate.source_id
            elif candidate.source_type == "table":
                kwargs["evidence_item_type"] = "table"
                kwargs["table_id"] = candidate.source_id
            elif candidate.source_type == "table_row":
                kwargs["evidence_item_type"] = "table_row"
                kwargs["table_row_id"] = candidate.source_id
            elif candidate.source_type == "block":
                kwargs["evidence_item_type"] = "block"
                kwargs["block_id"] = candidate.source_id
            else:
                continue

            kwargs["document_id"] = candidate.document_id

            items.append(EvidenceItemInput(**kwargs))

        return items

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    async def _run_deterministic_validation(
        self,
        *,
        question_text: str,
        answer_text: str,
        evidence_package: EvidencePackage,
        citations_json: list[dict[str, Any]],
    ) -> DeterministicValidationResult:
        if self.deterministic_validator is None:
            return self._default_deterministic_validation(
                question_text=question_text,
                answer_text=answer_text,
                evidence_package=evidence_package,
                citations_json=citations_json,
            )

        return await self.deterministic_validator.validate(
            DeterministicValidationInput(
                question_text=question_text,
                answer_text=answer_text,
                evidence_package=evidence_package,
                citations_json=citations_json,
            )
        )

    async def _run_semantic_validation(
        self,
        *,
        question_text: str,
        answer_text: str,
        evidence_package: EvidencePackage,
        answer_mode: AnswerModeEnum,
    ) -> SemanticValidationResult:
        if self.semantic_validator is None:
            return self._default_semantic_validation(
                question_text=question_text,
                answer_text=answer_text,
                evidence_package=evidence_package,
                answer_mode=answer_mode,
            )

        return await self.semantic_validator.validate(
            SemanticValidationInput(
                question_text=question_text,
                answer_text=answer_text,
                evidence_package=evidence_package,
                answer_mode=answer_mode,
            )
        )

    def _default_deterministic_validation(
        self,
        *,
        question_text: str,
        answer_text: str,
        evidence_package: EvidencePackage,
        citations_json: list[dict[str, Any]],
    ) -> DeterministicValidationResult:
        issues: list[str] = []

        if not answer_text.strip():
            issues.append("answer_text_empty")

        if not evidence_package.selected_candidates:
            issues.append("no_selected_candidates")

        if not citations_json and evidence_package.selected_candidates:
            issues.append("citations_missing")

        if len(answer_text) < 30 and evidence_package.selected_candidates:
            issues.append("answer_text_suspiciously_short")

        return DeterministicValidationResult(
            passed=len(issues) == 0,
            issues=issues,
            score=1.0 if len(issues) == 0 else max(0.0, 1.0 - 0.2 * len(issues)),
        )

    def _default_semantic_validation(
        self,
        *,
        question_text: str,
        answer_text: str,
        evidence_package: EvidencePackage,
        answer_mode: AnswerModeEnum,
    ) -> SemanticValidationResult:
        """
        Conservative placeholder semantic validation.

        Later this should be replaced with:
        - contradiction checks
        - unsupported claim checks
        - answer-to-question alignment checks
        """
        issues: list[str] = []

        if answer_mode != AnswerModeEnum.SAFE_NO_ANSWER and not evidence_package.selected_candidates:
            issues.append("semantic_without_evidence")

        if len(answer_text.strip()) < 20:
            issues.append("semantic_answer_too_short")

        return SemanticValidationResult(
            passed=len(issues) == 0,
            issues=issues,
            score=1.0 if len(issues) == 0 else max(0.0, 1.0 - 0.2 * len(issues)),
        )

    def _derive_validation_status(
        self,
        *,
        deterministic_passed: bool,
        semantic_passed: bool,
    ) -> ValidationStatusEnum:
        if deterministic_passed and semantic_passed:
            return ValidationStatusEnum.PASSED
        if deterministic_passed or semantic_passed:
            return ValidationStatusEnum.PARTIAL
        return ValidationStatusEnum.FAILED

    # --------------------------------------------------------
    # Confidence / reuse policy
    # --------------------------------------------------------

    def _calculate_confidence_score(
        self,
        *,
        plan: AnswerPlan,
        evidence_package: EvidencePackage,
        det_validation: DeterministicValidationResult,
        sem_validation: SemanticValidationResult,
    ) -> float:
        if plan.answer_mode == AnswerModeEnum.SAFE_NO_ANSWER:
            evidence_quality = str(
                (evidence_package.metrics_json or {}).get("evidence_quality") or ""
            ).strip().lower()
            if evidence_quality == "insufficient":
                return 0.20
            if evidence_quality == "weak":
                return 0.28
            return 0.35

        evidence_count = len(evidence_package.selected_candidates)
        evidence_component = min(evidence_count / 8.0, 1.0)

        det_score = det_validation.score if det_validation.score is not None else (1.0 if det_validation.passed else 0.0)
        sem_score = sem_validation.score if sem_validation.score is not None else (1.0 if sem_validation.passed else 0.0)

        mode_component = 0.90 if plan.answer_mode == AnswerModeEnum.DIRECT_STRUCTURED else 0.80

        metrics = evidence_package.metrics_json or {}
        strong_candidate_count = int(metrics.get("strong_candidate_count", 0) or 0)
        top_candidate_score = float(metrics.get("top_candidate_score", 0.0) or 0.0)
        top_document_share = float(metrics.get("top_document_share", 0.0) or 0.0)
        evidence_quality = str(metrics.get("evidence_quality", "") or "").strip().lower()

        strong_candidate_component = min(strong_candidate_count / 4.0, 1.0)
        top_candidate_component = min(max(top_candidate_score, 0.0), 1.0)
        document_focus_component = min(max(top_document_share, 0.0), 1.0)

        evidence_quality_component = 0.0
        if evidence_quality == "strong":
            evidence_quality_component = 1.0
        elif evidence_quality == "weak":
            evidence_quality_component = 0.35
        elif evidence_quality == "insufficient":
            evidence_quality_component = 0.0
        else:
            evidence_quality_component = 0.55

        value = (
            0.25 * evidence_component
            + 0.18 * det_score
            + 0.15 * sem_score
            + 0.12 * mode_component
            + 0.10 * strong_candidate_component
            + 0.10 * top_candidate_component
            + 0.05 * document_focus_component
            + 0.05 * evidence_quality_component
        )
        return round(min(max(value, 0.0), 1.0), 4)

    def _calculate_trust_score(
        self,
        *,
        answer_mode: AnswerModeEnum,
        confidence_score: float,
        det_validation: DeterministicValidationResult,
        sem_validation: SemanticValidationResult,
        evidence_package: EvidencePackage,
    ) -> float:
        penalty = 0.0
        if not det_validation.passed:
            penalty += 0.20
        if not sem_validation.passed:
            penalty += 0.15
        if answer_mode == AnswerModeEnum.SAFE_NO_ANSWER:
            penalty += 0.05

        metrics = evidence_package.metrics_json or {}
        evidence_quality = str(metrics.get("evidence_quality", "") or "").strip().lower()
        selected_document_ids_count = int(metrics.get("selected_document_ids_count", 0) or 0)
        top_document_share = float(metrics.get("top_document_share", 0.0) or 0.0)

        bonus = 0.0
        if selected_document_ids_count == 1:
            bonus += 0.06
        if top_document_share >= 0.75:
            bonus += 0.05
        elif top_document_share >= 0.50:
            bonus += 0.02

        if evidence_quality == "strong":
            bonus += 0.05
        elif evidence_quality == "weak":
            bonus -= 0.03
        elif evidence_quality == "insufficient":
            bonus -= 0.08

        return round(min(1.0, max(0.0, confidence_score - penalty + bonus)), 4)

    def _decide_reuse_allowed(
        self,
        *,
        payload: GenerationRequest,
        plan: AnswerPlan,
        validation_status: ValidationStatusEnum,
        confidence_score: float,
    ) -> bool:
        if plan.answer_mode == AnswerModeEnum.SAFE_NO_ANSWER:
            return False

        if validation_status != ValidationStatusEnum.PASSED:
            return False

        if confidence_score < 0.78:
            return False

        # Conservative allow-list for first production phase
        if payload.intent_type in {
            QuestionIntentEnum.DOCUMENTS_QUESTION,
            QuestionIntentEnum.DEADLINE_QUESTION,
            QuestionIntentEnum.PROCEDURE_QUESTION,
            QuestionIntentEnum.FORM_QUESTION,
        }:
            return True

        return False

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _validate_input(
        self,
        payload: GenerationRequest,
    ) -> None:
        if not payload.question_text_raw or not payload.question_text_raw.strip():
            raise ValueError("question_text_raw must not be empty.")

        if not payload.language_code or not payload.language_code.strip():
            raise ValueError("language_code must not be empty.")

    def _build_short_answer(
        self,
        *,
        answer_text: str,
        answer_mode: AnswerModeEnum,
    ) -> Optional[str]:
        if answer_mode == AnswerModeEnum.SAFE_NO_ANSWER:
            return answer_text

        short_text = self._shorten_text(answer_text, limit=350)
        return short_text

    def _build_safe_no_answer_result(
        self,
        *,
        payload: GenerationRequest,
        reason_code: str,
        evidence_package: Optional[EvidencePackage] = None,
    ) -> GenerationResult:
        answer_text = (
            "Я не могу надёжно ответить на этот вопрос только по найденным источникам. "
            "Для точного ответа в текущей выборке недостаточно подтверждённых данных."
        )

        evidence_quality = None
        guard_reason = None
        if evidence_package is not None:
            evidence_quality = (evidence_package.debug_payload_json or {}).get("evidence_quality")
            guard_reason = (evidence_package.debug_payload_json or {}).get("guard_reason")

        return GenerationResult(
            answer_mode=AnswerModeEnum.SAFE_NO_ANSWER,
            answer_text=answer_text,
            answer_text_short=answer_text,
            confidence_score=0.35,
            trust_score_at_generation=0.30,
            validation_status=ValidationStatusEnum.PASSED,
            deterministic_validation_passed=True,
            semantic_validation_passed=True,
            reuse_allowed=False,
            reuse_policy_version="reuse_gate_v1",
            citations_json=[],
            answer_payload_json={
                "reason_code": reason_code,
                "strategy_code": "safe_no_answer",
                "evidence_quality": evidence_quality,
                "guard_reason": guard_reason,
            },
            reuse_decision_payload_json={
                "reuse_allowed": False,
                "reason_code": reason_code,
                "evidence_quality": evidence_quality,
            },
            evidence_items=[],
            generation_model_name=None,
            generation_prompt_version="grounded_template_v1",
            pipeline_version="generation_pipeline_v1",
        )

    def _compact_json_dict(
        self,
        payload: dict[str, Any],
    ) -> str:
        parts: list[str] = []
        for key, value in payload.items():
            key_text = str(key).strip()
            if isinstance(value, list):
                value_text = ", ".join(str(x) for x in value[:6])
            else:
                value_text = str(value).strip()
            if key_text and value_text:
                parts.append(f"{key_text}: {value_text}")
        return "; ".join(parts[:4])

    def _normalize_sentence(
        self,
        value: Optional[str],
    ) -> Optional[str]:
        if value is None:
            return None
        text = " ".join(value.strip().split())
        if not text:
            return None
        if text[-1] not in ".!?":
            text += "."
        return text

    def _shorten_text(
        self,
        value: Optional[str],
        *,
        limit: int,
    ) -> Optional[str]:
        if value is None:
            return None
        text = " ".join(value.strip().split())
        if len(text) <= limit:
            return text
        if limit <= 3:
            return text[:limit]
        return text[: limit - 3].rstrip() + "..."