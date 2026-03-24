# ============================================================
# File: app/services/retrieval/retrieval_orchestrator.py
# Purpose:
#   Central retrieval orchestration for the RAG knowledge base.
#
# Responsibilities:
#   - choose retrieval strategy based on normalized question signature
#   - retrieve evidence candidates from:
#       * legal_facts
#       * document_tables
#       * document_table_rows
#       * document_blocks
#   - combine and rank candidates
#   - build a stable evidence package for downstream answer generation
#
# Design principles:
#   - retrieval is object-based, not chunk-only
#   - conservative evidence selection
#   - deterministic filters first, semantic ranking second
#   - active documents only
#   - transport-agnostic and generation-agnostic
# ============================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import and_, case, desc, func, literal, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.documents import (
    DocumentBlock,
    DocumentRegistry,
    DocumentTable,
    DocumentTableRow,
    LegalFact,
)
from app.db.models.enums import QuestionIntentEnum

logger = logging.getLogger(__name__)


# ============================================================
# Placeholder protocols
# ============================================================

class RetrievalRerankerProtocol:
    """
    Optional reranker for retrieved candidates.
    Expected to return candidates in best-first order, optionally
    updating their rerank_score.
    """

    async def rerank(
        self,
        *,
        question_text: str,
        candidates: list["RetrievedCandidate"],
    ) -> list["RetrievedCandidate"]:
        raise NotImplementedError


# ============================================================
# Exceptions
# ============================================================

class RetrievalOrchestratorError(Exception):
    """Base retrieval orchestrator error."""


class RetrievalValidationError(RetrievalOrchestratorError):
    """Raised when input is invalid."""


# ============================================================
# DTOs
# ============================================================

@dataclass(slots=True)
class RetrievalInput:
    """
    Input for retrieval orchestration.

    Note:
    - question_text_raw: original user question
    - question_text_normalized: normalized form used by routing/lookup
    - query_terms: optional extra search terms prepared upstream
    """
    question_event_id: UUID
    question_text_raw: str
    question_text_normalized: str

    intent_type: QuestionIntentEnum
    measure_code: Optional[str] = None
    subject_category_code: Optional[str] = None

    query_terms: list[str] = field(default_factory=list)
    constraints_json: dict[str, Any] = field(default_factory=dict)

    top_k_facts: int = 10
    top_k_tables: int = 10
    top_k_rows: int = 12
    top_k_blocks: int = 12
    final_top_k: int = 12


@dataclass(slots=True)
class RetrievedCandidate:
    """
    Unified retrieval candidate regardless of source object type.
    """
    source_type: str  # legal_fact | table | table_row | block
    source_id: UUID
    document_id: UUID

    score: float
    rerank_score: Optional[float] = None

    document_name: Optional[str] = None
    doc_uid_base: Optional[str] = None
    revision_date: Optional[str] = None

    measure_code: Optional[str] = None
    subject_category: Optional[str] = None

    title: Optional[str] = None
    snippet: Optional[str] = None
    citation_json: dict[str, Any] = field(default_factory=dict)
    metadata_json: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvidencePackage:
    """
    Final retrieval output for answer generation.
    """
    question_event_id: UUID
    strategy_code: str

    selected_candidates: list[RetrievedCandidate] = field(default_factory=list)

    selected_fact_ids: list[UUID] = field(default_factory=list)
    selected_table_ids: list[UUID] = field(default_factory=list)
    selected_row_ids: list[UUID] = field(default_factory=list)
    selected_block_ids: list[UUID] = field(default_factory=list)
    selected_document_ids: list[UUID] = field(default_factory=list)

    metrics_json: dict[str, Any] = field(default_factory=dict)
    debug_payload_json: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalStrategy:
    """
    Strategy selected based on question signature.
    """
    strategy_code: str
    use_facts: bool = True
    use_tables: bool = True
    use_rows: bool = True
    use_blocks: bool = True

    facts_weight: float = 1.0
    tables_weight: float = 1.0
    rows_weight: float = 1.0
    blocks_weight: float = 1.0


# ============================================================
# Orchestrator
# ============================================================

class RetrievalOrchestrator:
    """
    Central retrieval logic for structured legal knowledge base.
    """

    def __init__(
        self,
        db: AsyncSession,
        *,
        reranker: Optional[RetrievalRerankerProtocol] = None,
    ) -> None:
        self.db = db
        self.reranker = reranker

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    async def retrieve(
        self,
        payload: RetrievalInput,
    ) -> EvidencePackage:
        """
        Main retrieval flow.

        Steps:
        1. Validate input
        2. Select strategy based on intent
        3. Retrieve candidates from relevant object sets
        4. Merge / deduplicate
        5. Optional rerank
        6. Build document-level aggregates
        7. Select final evidence package
        """
        self._validate_input(payload)

        strategy = self._select_strategy(payload)
        query_bundle = self._build_query_bundle(payload)

        raw_candidates: list[RetrievedCandidate] = []

        if strategy.use_facts:
            raw_candidates.extend(
                await self._retrieve_legal_facts(
                    payload=payload,
                    strategy=strategy,
                    query_bundle=query_bundle,
                )
            )

        if strategy.use_tables:
            raw_candidates.extend(
                await self._retrieve_tables(
                    payload=payload,
                    strategy=strategy,
                    query_bundle=query_bundle,
                )
            )

        if strategy.use_rows:
            raw_candidates.extend(
                await self._retrieve_table_rows(
                    payload=payload,
                    strategy=strategy,
                    query_bundle=query_bundle,
                )
            )

        if strategy.use_blocks:
            raw_candidates.extend(
                await self._retrieve_blocks(
                    payload=payload,
                    strategy=strategy,
                    query_bundle=query_bundle,
                )
            )

        merged_candidates = self._merge_and_deduplicate_candidates(raw_candidates)

        merged_candidates = self._apply_semantic_priority(merged_candidates)

        merged_candidates = [
            c
            for c in merged_candidates
            if not self._is_header_noise(c)
        ]

        if payload.intent_type == QuestionIntentEnum.DOCUMENTS_QUESTION:
            merged_candidates = [
                c
                for c in merged_candidates
                if c.source_type != "block"
                or self._has_documents_anchor_match(
                    c,
                    query_bundle["query_terms"],
                )
            ]

        elif payload.intent_type in {
            QuestionIntentEnum.DEADLINE_QUESTION,
            QuestionIntentEnum.FORM_QUESTION,
        }:
            merged_candidates = [
                c
                for c in merged_candidates
                if c.source_type != "block"
                or self._has_meaningful_lexical_match(
                    c,
                    query_bundle["query_terms"],
                )
            ]

        elif payload.intent_type in {
            QuestionIntentEnum.REJECTION_QUESTION,
            QuestionIntentEnum.ELIGIBILITY_QUESTION,
        }:
            merged_candidates = [
                c
                for c in merged_candidates
                if c.source_type != "block"
                or self._has_min_intent_anchor_match(
                    c,
                    query_terms=query_bundle["query_terms"],
                    min_matches=2,
                )
            ]

        merged_candidates = self._apply_intent_semantic_rerank(
            candidates=merged_candidates,
            payload=payload,
            query_bundle=query_bundle,
        )

        # В текущей версии файла _maybe_rerank имеет несовместимую сигнатуру.
        # Пока безопасно пропускаем этот этап и используем локальный semantic rerank.
        reranked_candidates = merged_candidates

        document_stats = self._build_document_aggregates(reranked_candidates)

        priority_document_ids = self._select_priority_documents(
            payload=payload,
            strategy=strategy,
            document_stats=document_stats,
        )

        selected_candidates = self._select_final_candidates(
            payload=payload,
            strategy=strategy,
            candidates=reranked_candidates,
            document_stats=document_stats,
            priority_document_ids=priority_document_ids,
        )

        package = self._build_evidence_package(
            payload=payload,
            strategy=strategy,
            query_bundle=query_bundle,
            raw_candidates=raw_candidates,
            merged_candidates=merged_candidates,
            reranked_candidates=reranked_candidates,
            final_candidates=selected_candidates,
            document_stats=document_stats,
            priority_document_ids=priority_document_ids,
        )

        logger.info(
            "Retrieval completed",
            extra={
                "question_event_id": str(payload.question_event_id),
                "strategy_code": strategy.strategy_code,
                "raw_candidates_count": len(raw_candidates),
                "merged_candidates_count": len(merged_candidates),
                "reranked_candidates_count": len(reranked_candidates),
                "final_candidates_count": len(package.selected_candidates),
                "selected_document_ids_count": len(package.selected_document_ids),
                "evidence_quality": package.debug_payload_json.get("evidence_quality"),
                "guard_reason": package.debug_payload_json.get("guard_reason"),
            },
        )
        return package

    # --------------------------------------------------------
    # Strategy
    # --------------------------------------------------------

    def _select_strategy(
        self,
        payload: RetrievalInput,
    ) -> RetrievalStrategy:
        """
        Select retrieval strategy based on intent.

        Conservative defaults:
        - documents/deadlines/forms -> prioritize facts/tables/rows
        - rejection/eligibility -> prioritize facts + blocks
        - procedure/mixed -> broad search
        """
        intent = payload.intent_type

        if intent == QuestionIntentEnum.DOCUMENTS_QUESTION:
            return RetrievalStrategy(
                strategy_code="documents_priority",
                use_facts=True,
                use_tables=True,
                use_rows=True,
                use_blocks=True,
                facts_weight=1.10,
                tables_weight=1.30,
                rows_weight=1.45,
                blocks_weight=0.65,
            )

        if intent == QuestionIntentEnum.DEADLINE_QUESTION:
            return RetrievalStrategy(
                strategy_code="deadlines_priority",
                use_facts=True,
                use_tables=True,
                use_rows=True,
                use_blocks=True,
                facts_weight=1.25,
                tables_weight=1.10,
                rows_weight=1.10,
                blocks_weight=0.90,
            )

        if intent == QuestionIntentEnum.REJECTION_QUESTION:
            return RetrievalStrategy(
                strategy_code="rejection_priority",
                use_facts=True,
                use_tables=True,
                use_rows=True,
                use_blocks=True,
                facts_weight=1.25,
                tables_weight=1.00,
                rows_weight=1.00,
                blocks_weight=1.10,
            )

        if intent == QuestionIntentEnum.FORM_QUESTION:
            return RetrievalStrategy(
                strategy_code="forms_priority",
                use_facts=False,
                use_tables=True,
                use_rows=True,
                use_blocks=True,
                facts_weight=0.00,
                tables_weight=1.30,
                rows_weight=1.20,
                blocks_weight=0.90,
            )

        if intent == QuestionIntentEnum.ELIGIBILITY_QUESTION:
            return RetrievalStrategy(
                strategy_code="eligibility_priority",
                use_facts=True,
                use_tables=True,
                use_rows=True,
                use_blocks=True,
                facts_weight=1.30,
                tables_weight=0.95,
                rows_weight=1.00,
                blocks_weight=1.10,
            )

        if intent == QuestionIntentEnum.PROCEDURE_QUESTION:
            return RetrievalStrategy(
                strategy_code="procedure_balanced",
                use_facts=True,
                use_tables=True,
                use_rows=True,
                use_blocks=True,
                facts_weight=1.00,
                tables_weight=1.00,
                rows_weight=1.00,
                blocks_weight=1.00,
            )

        return RetrievalStrategy(
            strategy_code="balanced_default",
            use_facts=True,
            use_tables=True,
            use_rows=True,
            use_blocks=True,
            facts_weight=1.00,
            tables_weight=1.00,
            rows_weight=1.00,
            blocks_weight=1.00,
        )

    def _build_query_bundle(
        self,
        payload: RetrievalInput,
    ) -> dict[str, Any]:
        normalized_text = self._normalize_text(
            payload.question_text_normalized or payload.question_text_raw
        )

        upstream_terms = [
            self._normalize_text(x)
            for x in payload.query_terms
            if self._normalize_text(x)
        ]

        table_question_profile = self._detect_table_question_profile(
            question_text=normalized_text,
            intent_type=payload.intent_type,
        )
        submission_channel = self._detect_submission_channel(normalized_text)
        requested_column_hints = self._build_requested_column_hints(
            table_question_profile=table_question_profile,
            submission_channel=submission_channel,
        )
        table_scope_hints = self._infer_table_scope_hints(
            intent_type=payload.intent_type,
            table_question_profile=table_question_profile,
        )

        expanded_terms = self._expand_query_terms(
            question_text=normalized_text,
            intent_type=payload.intent_type,
            measure_code=payload.measure_code,
            submission_channel=submission_channel,
            requested_column_hints=requested_column_hints,
        )

        query_terms = self._deduplicate_preserve_order(
            [
                *expanded_terms,
                *upstream_terms,
                *requested_column_hints,
                *table_scope_hints,
            ]
        )

        return {
            "normalized_text": normalized_text,
            "query_terms": query_terms,
            "measure_code": payload.measure_code,
            "subject_category_code": payload.subject_category_code,
            "table_question_profile": table_question_profile,
            "submission_channel": submission_channel,
            "requested_column_hints": requested_column_hints,
            "table_scope_hints": table_scope_hints,
        }


    def _expand_query_terms(
        self,
        *,
        question_text: str,
        intent_type: QuestionIntentEnum,
        measure_code: Optional[str],
        submission_channel: Optional[str] = None,
        requested_column_hints: Optional[list[str]] = None,
    ) -> list[str]:
        terms: list[str] = []

        base_tokens = self._extract_meaningful_terms(question_text)
        terms.extend(base_tokens)

        if measure_code:
            terms.append(self._normalize_text(measure_code))

        if "едв" in question_text:
            terms.extend(
                [
                    "едв",
                    "ежемесячной денежной выплаты",
                    "ежемесячная денежная выплата",
                    "денежной выплаты",
                ]
            )

        if intent_type == QuestionIntentEnum.DEADLINE_QUESTION:
            terms.extend(
                [
                    "срок",
                    "срок принятия решения",
                    "срок рассмотрения",
                    "рассмотрения заявления",
                    "принятия решения",
                    "рабочих дней",
                    "календарных дней",
                    "дней",
                ]
            )

        elif intent_type == QuestionIntentEnum.DOCUMENTS_QUESTION:
            terms.extend(
                [
                    "документы",
                    "перечень документов",
                    "необходимые документы",
                    "документов необходимых",
                    "представляемые документы",
                    "документы представляемые заявителем",
                    "к заявлению",
                    "заявление",
                    "заявителем",
                    "прилагаемые документы",
                    "предоставления государственной услуги",
                    "наименование документа",
                    "документы необходимые для предоставления государственной услуги",
                ]
            )

            if submission_channel == "epgu":
                terms.extend(
                    [
                        "епгу",
                        "госуслуги",
                        "единый портал",
                        "посредством епгу",
                        "электронной подаче",
                        "в электронной форме",
                        "электронный образ документа",
                        "сведения о документе",
                        "единого портала",
                    ]
                )

            elif submission_channel == "regional_portal":
                terms.extend(
                    [
                        "рпгу",
                        "краевой портал",
                        "региональный портал",
                        "посредством краевого портала",
                        "электронной подаче",
                        "при электронной подаче посредством рпгу",
                        "электронный образ документа",
                        "сведения о документе",
                    ]
                )

            elif submission_channel == "in_person":
                terms.extend(
                    [
                        "лично",
                        "личной подаче",
                        "личном обращении",
                        "при личной подаче",
                        "через представителя",
                        "по доверенности",
                        "через социального работника",
                    ]
                )

            elif submission_channel == "post":
                terms.extend(
                    [
                        "почтой",
                        "почтовым отправлением",
                        "по почте",
                    ]
                )

            elif submission_channel == "mfc":
                terms.extend(
                    [
                        "мфц",
                        "через мфц",
                    ]
                )

        elif intent_type == QuestionIntentEnum.REJECTION_QUESTION:
            terms.extend(
                [
                    "отказ",
                    "основания отказа",
                    "причины отказа",
                    "решение об отказе",
                    "может быть отказано",
                    "отказывается в предоставлении",
                    "отказывается в назначении",
                    "принимается решение об отказе",
                    "непредставление документов",
                    "недостоверные сведения",
                    "отсутствие права",
                ]
            )

        elif intent_type == QuestionIntentEnum.ELIGIBILITY_QUESTION:
            terms.extend(
                [
                    "имеет право",
                    "право на едв",
                    "право на получение",
                    "категории граждан",
                    "категории заявителей",
                    "заявитель",
                    "получатели",
                    "предоставляется",
                    "предоставляется заявителям",
                    "условия предоставления",
                    "при наличии права",
                ]
            )

        elif intent_type == QuestionIntentEnum.PROCEDURE_QUESTION:
            terms.extend(
                [
                    "порядок",
                    "порядок предоставления",
                    "процедура",
                    "последовательность",
                    "предоставление услуги",
                    "назначение",
                    "назначается",
                    "рассмотрение заявления",
                    "принятие решения",
                ]
            )

        if requested_column_hints:
            terms.extend(requested_column_hints)

        return self._deduplicate_preserve_order(
            [self._normalize_text(x) for x in terms if self._normalize_text(x)]
        )[:24]


    def _extract_meaningful_terms(self, text: str) -> list[str]:
        stopwords = {
            "в", "во", "на", "по", "о", "об", "от", "до", "за", "для",
            "и", "или", "а", "но", "как", "какой", "какие", "каким",
            "какая", "каких", "когда", "где", "кто", "что", "это",
            "при", "ли", "из",
        }

        parts = [
            token.strip()
            for token in text.replace('"', " ").replace("(", " ").replace(")", " ").split()
        ]

        result: list[str] = []
        for token in parts:
            token = self._normalize_text(token)
            if not token:
                continue
            if token in stopwords:
                continue
            if len(token) < 3 and token != "едв":
                continue
            result.append(token)

        return result[:8]


    def _deduplicate_preserve_order(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []

        for item in items:
            if not item or item in seen:
                continue
            seen.add(item)
            result.append(item)

        return result

    def _candidate_text_blob(self, candidate: RetrievedCandidate) -> str:
        """
        Build a richer normalized text blob for lexical/semantic checks.

        Important:
        - for table/table_row candidates we must inspect not only title/snippet
          but also metadata_json fields such as row_summary, table title,
          semantic labels and optional column/header hints.
        """
        parts: list[str] = [
            candidate.title or "",
            candidate.snippet or "",
            candidate.document_name or "",
            candidate.measure_code or "",
            candidate.subject_category or "",
        ]

        metadata = candidate.metadata_json or {}

        rich_string_keys = [
            "table_title",
            "table_name",
            "table_label",
            "table_number",
            "appendix_number",
            "row_summary",
            "row_text",
            "table_semantic_type",
            "document_semantic_type",
            "section_title",
            "section_path",
            "header_text",
        ]
        for key in rich_string_keys:
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value)

        rich_list_keys = [
            "column_headers",
            "header_tokens",
            "cells_text",
            "semantic_tags",
            "table_tags",
        ]
        for key in rich_list_keys:
            value = metadata.get(key)
            if isinstance(value, list):
                parts.extend(str(x) for x in value if x)

        # Optional structured cell maps if ingestion already persists them.
        for key in ["cells_by_header", "cells_by_header_normalized"]:
            value = metadata.get(key)
            if isinstance(value, dict):
                for header, cell_value in value.items():
                    parts.append(str(header))
                    if isinstance(cell_value, str):
                        parts.append(cell_value)
                    elif isinstance(cell_value, list):
                        parts.extend(str(x) for x in cell_value if x)

        return self._normalize_text(" ".join(parts))

    def _detect_submission_channel(self, text: str) -> Optional[str]:
        """
        Detect submission channel from user question.

        This is intentionally generic for administrative-regulation style questions.
        """
        text = self._normalize_text(text)

        if any(marker in text for marker in ["епгу", "госуслуг", "единый портал"]):
            return "epgu"

        if any(marker in text for marker in ["рпгу", "краевой портал", "региональный портал"]):
            return "regional_portal"

        if any(marker in text for marker in ["лично", "личный прием", "личном обращении", "личной подаче", "по доверенности", "через представителя"]):
            return "in_person"

        if any(marker in text for marker in ["почтой", "почтов", "почтовым отправлением", "по почте"]):
            return "post"

        if any(marker in text for marker in ["мфц"]):
            return "mfc"

        return None

    def _detect_table_question_profile(
        self,
        *,
        question_text: str,
        intent_type: QuestionIntentEnum,
    ) -> Optional[str]:
        """
        Distinguish document questions:
        - documents_base
        - documents_by_submission_channel
        """
        if intent_type == QuestionIntentEnum.DEADLINE_QUESTION:
            return "deadline"

        if intent_type != QuestionIntentEnum.DOCUMENTS_QUESTION:
            return None

        text = self._normalize_text(question_text)
        documents_markers = [
            "какие документы",
            "какие нужны документы",
            "перечень документов",
            "необходимые документы",
            "что приложить",
            "что предоставить",
            "какие документы нужны",
            "какие документы нужны для",
            "документы для получения",
            "документы для предоставления услуги",
        ]

        if not any(marker in text for marker in documents_markers):
            return None

        submission_channel = self._detect_submission_channel(text)
        if submission_channel:
            return "documents_by_submission_channel"

        return "documents_base"

    def _build_requested_column_hints(
        self,
        *,
        table_question_profile: Optional[str],
        submission_channel: Optional[str],
    ) -> list[str]:
        """
        Hints for downstream retrieval / ranking / debug.
        This does not assume exact DB schema for columns.
        These are normalized semantic hints.
        """
        if table_question_profile == "documents_base":
            return [
                "document_name",
                "наименование документа",
            ]

        if table_question_profile == "deadline":
            return [
                "deadline_value",
                "срок",
                "сроки",
                "срок предоставления",
                "срок предоставления государственной услуги",
                "срок принятия решения",
                "срок рассмотрения",
                "максимальный срок",
                "рабочих дней",
                "календарных дней",
            ]

        if table_question_profile == "documents_by_submission_channel":
            mapping = {
                "epgu": [
                    "document_name",
                    "наименование документа",
                    "epgu_submission",
                    "при электронной подаче посредством епгу",
                    "электронной подаче",
                    "единого портала",
                ],
                "regional_portal": [
                    "document_name",
                    "наименование документа",
                    "regional_portal_submission",
                    "при электронной подаче посредством краевого портала",
                    "электронной подаче",
                    "краевого портала",
                    "при электронной подаче посредством рпгу",
                ],
                "in_person": [
                    "document_name",
                    "наименование документа",
                    "in_person_submission",
                    "при личной подаче",
                    "лично",
                    "через представителя",
                    "по доверенности",
                    "через социального работника",
                ],
                "post": [
                    "document_name",
                    "наименование документа",
                    "post_submission",
                    "почтовым отправлением",
                    "по почте",
                ],
                "mfc": [
                    "document_name",
                    "наименование документа",
                    "mfc_submission",
                    "через мфц",
                    "мфц",
                ],
            }
            return mapping.get(submission_channel or "", ["document_name", "наименование документа"])

        return []

    def _is_required_documents_table_candidate(self, candidate: RetrievedCandidate) -> bool:
        """
        Heuristic: identify candidates belonging to a required-documents table.
        Designed to be generic, not tied to a single regulation.
        """
        text = self._candidate_text_blob(candidate)

        markers = [
            "перечень документов",
            "необходимые документы",
            "документы необходимые для предоставления государственной услуги",
            "документы представляемые заявителем",
            "наименование документа",
            "представляется электронный образ документа",
            "сведения о документе",
            "таблица 2",
            "приложение n 2",
            "приложения n 2",
        ]

        matches = sum(1 for marker in markers if marker in text)
        return matches >= 2
        
    def _is_abbreviation_table_candidate(self, candidate: RetrievedCandidate) -> bool:
        text = self._candidate_text_blob(candidate)

        markers = [
            "условных обозначений",
            "сокращений",
            "условных обозначений и сокращений",
            "перечень условных обозначений",
        ]
        return any(marker in text for marker in markers)
        
    def _has_table_semantic_type(
        self,
        candidate: RetrievedCandidate,
        expected_type: str,
    ) -> bool:
        metadata = candidate.metadata_json or {}
        actual = str(metadata.get("table_semantic_type") or "").strip().lower()
        return actual == expected_type.strip().lower()
        
    def _looks_like_service_documents_row(self, candidate: RetrievedCandidate) -> bool:
        text = self._candidate_text_blob(candidate)

        markers = [
            "документы информация необходимые для предоставления государственной услуги",
            "документы (информация), необходимые для предоставления государственной услуги",
            "исчерпывающий перечень документов",
            "способ подачи в уполномоченное учреждение",
        ]

        if any(marker in text for marker in markers):
            # Если это уже реальная строка с документом, не считаем её service-row
            real_doc_markers = [
                "запрос заявление",
                "паспорт",
                "документ подтверждающий полномочия",
                "вступившее в законную силу решение суда",
                "справка",
            ]
            if any(marker in text for marker in real_doc_markers):
                return False
            return True

        return False
        
    def _looks_like_service_deadline_row(self, candidate: RetrievedCandidate) -> bool:
        metadata = candidate.metadata_json or {}
        cells = metadata.get("cells_by_semantic_key") or metadata.get("cells_by_header_key") or {}
        if not isinstance(cells, dict):
            return False

        service_values = {
            "срок",
            "сроки",
            "срок предоставления",
            "срок предоставления государственной услуги",
            "максимальный срок",
            "рабочих дней",
            "календарных дней",
        }

        for value in cells.values():
            normalized = self._normalize_text(value)
            if normalized in service_values:
                return True

        text_blob = self._normalize_text(self._candidate_text_blob(candidate))
        if text_blob.startswith("срок ") and len(text_blob.split()) <= 4:
            return True

        return False

    def _has_submission_channel_match(
        self,
        candidate: RetrievedCandidate,
        submission_channel: Optional[str],
    ) -> bool:
        """
        Check whether candidate text mentions the requested submission channel.
        """
        if not submission_channel:
            return False

        text = self._candidate_text_blob(candidate)

        channel_markers: dict[str, list[str]] = {
            "epgu": [
                "епгу",
                "посредством епгу",
                "электронной подаче",
                "в электронной форме",
                "электронный образ документа",
                "единого портала",
                "сведения о документе",
            ],
            "regional_portal": [
                "краевой портал",
                "посредством краевого портала",
                "электронной подаче",
                "в электронной форме",
                "электронный образ документа",
                "посредством рпгу",
                "сведения о документе",
            ],
            "in_person": [
                "лично",
                "при личной подаче",
                "личном обращении",
                "через представителя",
                "по доверенности",
                "через социального работника",                
            ],
            "post": [
                "почтой",
                "почтовым отправлением",
                "по почте",
            ],
            "mfc": [
                "мфц",
                "через мфц",
            ],
        }

        return any(marker in text for marker in channel_markers.get(submission_channel, []))

    def _infer_table_scope_hints(
        self,
        *,
        intent_type: QuestionIntentEnum,
        table_question_profile: Optional[str],
    ) -> list[str]:
        """
        Generic semantic hints about expected table family.
        """
        if intent_type == QuestionIntentEnum.DOCUMENTS_QUESTION and table_question_profile:
            return [
                "required_documents",
                "documents_for_service",
            ]
        return []

    # --------------------------------------------------------
    # Branch retrieval
    # --------------------------------------------------------

    async def _retrieve_legal_facts(
        self,
        *,
        payload: RetrievalInput,
        strategy: RetrievalStrategy,
        query_bundle: dict[str, Any],
    ) -> list[RetrievedCandidate]:
        """
        Retrieve from normalized legal facts.

        Priority:
        - active documents only
        - measure_code exact match if present
        - subject_category exact match if present
        - fact_type boosts depending on intent
        """
        text_terms = query_bundle["query_terms"]
        measure_code = query_bundle["measure_code"]
        subject_category = query_bundle["subject_category_code"]

        stmt = (
            select(
                LegalFact.fact_id.label("source_id"),
                LegalFact.document_id.label("document_id"),
                DocumentRegistry.document_name.label("document_name"),
                DocumentRegistry.doc_uid_base.label("doc_uid_base"),
                DocumentRegistry.revision_date.label("revision_date"),
                LegalFact.measure_code.label("measure_code"),
                LegalFact.subject_category.label("subject_category"),
                LegalFact.fact_type.label("title"),
                LegalFact.validity_note.label("snippet"),
                LegalFact.citation_json.label("citation_json"),
                LegalFact.metadata_json.label("metadata_json"),
                (
                    self._fact_match_score_expr(
                        text_terms=text_terms,
                        measure_code=measure_code,
                        subject_category=subject_category,
                        intent_type=payload.intent_type,
                    ) * strategy.facts_weight
                ).label("score"),
            )
            .join(DocumentRegistry, DocumentRegistry.document_id == LegalFact.document_id)
            .where(DocumentRegistry.status == "active")
        )

        if measure_code:
            stmt = stmt.where(
                or_(
                    LegalFact.measure_code == measure_code,
                    LegalFact.measure_code.is_(None),
                )
            )

        if subject_category:
            stmt = stmt.where(
                or_(
                    LegalFact.subject_category == subject_category,
                    LegalFact.subject_category.is_(None),
                )
            )

        stmt = stmt.order_by(desc("score")).limit(payload.top_k_facts)

        result = await self.db.execute(stmt)
        rows = result.mappings().all()

        candidates: list[RetrievedCandidate] = []
        for row in rows:
            score = float(row["score"] or 0.0)
            if score <= 0:
                continue

            candidates.append(
                RetrievedCandidate(
                    source_type="legal_fact",
                    source_id=row["source_id"],
                    document_id=row["document_id"],
                    score=score,
                    document_name=row["document_name"],
                    doc_uid_base=row["doc_uid_base"],
                    revision_date=self._datetime_to_iso(row["revision_date"]),
                    measure_code=row["measure_code"],
                    subject_category=row["subject_category"],
                    title=row["title"],
                    snippet=row["snippet"],
                    citation_json=row["citation_json"] or {},
                    metadata_json=row["metadata_json"] or {},
                )
            )
        return candidates

    async def _retrieve_tables(
        self,
        *,
        payload: RetrievalInput,
        strategy: RetrievalStrategy,
        query_bundle: dict[str, Any],
    ) -> list[RetrievedCandidate]:
        """
        Retrieve tables by title/summary/preview plus document filters.
        """
        text_terms = query_bundle["query_terms"]

        stmt = (
            select(
                DocumentTable.table_id.label("source_id"),
                DocumentTable.document_id.label("document_id"),
                DocumentRegistry.document_name.label("document_name"),
                DocumentRegistry.doc_uid_base.label("doc_uid_base"),
                DocumentRegistry.revision_date.label("revision_date"),
                literal(None).label("measure_code"),
                literal(None).label("subject_category"),
                DocumentTable.table_title.label("title"),
                DocumentTable.summary.label("snippet"),
                DocumentTable.citation_json.label("citation_json"),
                DocumentTable.metadata_json.label("metadata_json"),
                (
                    self._table_match_score_expr(text_terms=text_terms) * strategy.tables_weight
                ).label("score"),
            )
            .join(DocumentRegistry, DocumentRegistry.document_id == DocumentTable.document_id)
            .where(DocumentRegistry.status == "active")
        )

        if payload.measure_code:
            stmt = stmt.where(
                or_(
                    DocumentTable.metadata_json["measure_code"].astext == payload.measure_code,
                    DocumentTable.metadata_json["measure_code"].astext.is_(None),
                )
            )

        stmt = stmt.order_by(desc("score")).limit(payload.top_k_tables)

        result = await self.db.execute(stmt)
        rows = result.mappings().all()

        candidates: list[RetrievedCandidate] = []
        for row in rows:
            score = float(row["score"] or 0.0)
            if score <= 0:
                continue

            candidates.append(
                RetrievedCandidate(
                    source_type="table",
                    source_id=row["source_id"],
                    document_id=row["document_id"],
                    score=score,
                    document_name=row["document_name"],
                    doc_uid_base=row["doc_uid_base"],
                    revision_date=self._datetime_to_iso(row["revision_date"]),
                    title=row["title"],
                    snippet=row["snippet"],
                    citation_json=row["citation_json"] or {},
                    metadata_json=row["metadata_json"] or {},
                )
            )
        return candidates

    async def _retrieve_table_rows(
        self,
        *,
        payload: RetrievalInput,
        strategy: RetrievalStrategy,
        query_bundle: dict[str, Any],
    ) -> list[RetrievedCandidate]:
        """
        Retrieve row-level evidence.

        Important for:
        - documents_question
        - form_question
        - deadline_question
        where exact row selection often matters.
        """
        text_terms = query_bundle["query_terms"]

        stmt = (
            select(
                DocumentTableRow.row_id.label("source_id"),
                DocumentTableRow.document_id.label("document_id"),
                DocumentRegistry.document_name.label("document_name"),
                DocumentRegistry.doc_uid_base.label("doc_uid_base"),
                DocumentRegistry.revision_date.label("revision_date"),
                literal(None).label("measure_code"),
                literal(None).label("subject_category"),
                literal("table_row").label("title"),
                DocumentTableRow.row_summary.label("snippet"),
                DocumentTableRow.citation_json.label("citation_json"),
                DocumentTableRow.metadata_json.label("metadata_json"),
                (
                    self._row_match_score_expr(text_terms=text_terms) * strategy.rows_weight
                ).label("score"),
            )
            .join(DocumentRegistry, DocumentRegistry.document_id == DocumentTableRow.document_id)
            .where(DocumentRegistry.status == "active")
        )

        if payload.measure_code:
            stmt = stmt.where(
                or_(
                    DocumentTableRow.metadata_json["measure_code"].astext == payload.measure_code,
                    DocumentTableRow.metadata_json["measure_code"].astext.is_(None),
                )
            )

        stmt = stmt.order_by(desc("score")).limit(payload.top_k_rows)

        result = await self.db.execute(stmt)
        rows = result.mappings().all()

        candidates: list[RetrievedCandidate] = []
        for row in rows:
            score = float(row["score"] or 0.0)
            if score <= 0:
                continue

            candidates.append(
                RetrievedCandidate(
                    source_type="table_row",
                    source_id=row["source_id"],
                    document_id=row["document_id"],
                    score=score,
                    document_name=row["document_name"],
                    doc_uid_base=row["doc_uid_base"],
                    revision_date=self._datetime_to_iso(row["revision_date"]),
                    title=row["title"],
                    snippet=row["snippet"],
                    citation_json=row["citation_json"] or {},
                    metadata_json=row["metadata_json"] or {},
                )
            )
        return candidates

    async def _retrieve_blocks(
        self,
        *,
        payload: RetrievalInput,
        strategy: RetrievalStrategy,
        query_bundle: dict[str, Any],
    ) -> list[RetrievedCandidate]:
        """
        Retrieve paragraph/list/heading level blocks.

        Blocks remain useful for:
        - explanatory norms
        - procedure descriptions
        - appeal rules
        - narrative fallback when no exact structured object exists
        """
        text_terms = query_bundle["query_terms"]

        stmt = (
            select(
                DocumentBlock.block_id.label("source_id"),
                DocumentBlock.document_id.label("document_id"),
                DocumentRegistry.document_name.label("document_name"),
                DocumentRegistry.doc_uid_base.label("doc_uid_base"),
                DocumentRegistry.revision_date.label("revision_date"),
                literal(None).label("measure_code"),
                literal(None).label("subject_category"),
                DocumentBlock.block_type.label("title"),
                DocumentBlock.content_clean.label("snippet"),
                DocumentBlock.citation_json.label("citation_json"),
                DocumentBlock.metadata_json.label("metadata_json"),
                (
                    self._block_match_score_expr(
                        text_terms=text_terms,
                        intent_type=payload.intent_type,
                    ) * strategy.blocks_weight
                ).label("score"),
            )
            .join(DocumentRegistry, DocumentRegistry.document_id == DocumentBlock.document_id)
            .where(DocumentRegistry.status == "active")
        )

        stmt = stmt.order_by(desc("score")).limit(payload.top_k_blocks)

        result = await self.db.execute(stmt)
        rows = result.mappings().all()

        candidates: list[RetrievedCandidate] = []
        for row in rows:
            score = float(row["score"] or 0.0)
            if score <= 0:
                continue

            snippet = self._shorten_text(row["snippet"], limit=800)

            candidates.append(
                RetrievedCandidate(
                    source_type="block",
                    source_id=row["source_id"],
                    document_id=row["document_id"],
                    score=score,
                    document_name=row["document_name"],
                    doc_uid_base=row["doc_uid_base"],
                    revision_date=self._datetime_to_iso(row["revision_date"]),
                    title=row["title"],
                    snippet=snippet,
                    citation_json=row["citation_json"] or {},
                    metadata_json=row["metadata_json"] or {},
                )
            )
        return candidates

    # --------------------------------------------------------
    # SQL scoring expressions
    # --------------------------------------------------------

    def _fact_match_score_expr(
        self,
        *,
        text_terms: list[str],
        measure_code: Optional[str],
        subject_category: Optional[str],
        intent_type: QuestionIntentEnum,
    ):
        """
        Deterministic SQL-side score for legal facts.

        Current implementation uses lightweight ILIKE matching.
        Later can be upgraded with:
        - pg_trgm similarity
        - vector search
        - FTS ranking
        """
        score = literal(0.0)

        if measure_code:
            score = score + case(
                (LegalFact.measure_code == measure_code, 1.0),
                else_=0.0,
            )

        if subject_category:
            score = score + case(
                (LegalFact.subject_category == subject_category, 0.7),
                else_=0.0,
            )

        if intent_type == QuestionIntentEnum.DOCUMENTS_QUESTION:
            score = score + case(
                (LegalFact.fact_type.in_(["required_documents", "documents"]), 0.9),
                else_=0.0,
            )
        elif intent_type == QuestionIntentEnum.DEADLINE_QUESTION:
            score = score + case(
                (LegalFact.fact_type.in_(["deadline", "review_period", "payment_deadline"]), 0.9),
                else_=0.0,
            )
        elif intent_type == QuestionIntentEnum.REJECTION_QUESTION:
            score = score + case(
                (LegalFact.fact_type.in_(["rejection_reason", "grounds_for_refusal"]), 0.9),
                else_=0.0,
            )
        elif intent_type == QuestionIntentEnum.ELIGIBILITY_QUESTION:
            score = score + case(
                (LegalFact.fact_type.in_(["eligibility", "condition", "recipient_category"]), 0.9),
                else_=0.0,
            )

        for term in text_terms[:6]:
            like_term = f"%{term}%"
            score = score + case(
                (
                    or_(
                        LegalFact.validity_note.ilike(like_term),
                        func.cast(LegalFact.value_json, literal("TEXT").type).ilike(like_term),
                        func.cast(LegalFact.condition_json, literal("TEXT").type).ilike(like_term),
                    ),
                    0.35,
                ),
                else_=0.0,
            )

        return score

    def _table_match_score_expr(
        self,
        *,
        text_terms: list[str],
    ):
        score = literal(0.0)
        for term in text_terms[:6]:
            like_term = f"%{term}%"
            score = score + case(
                (
                    or_(
                        DocumentTable.table_title.ilike(like_term),
                        DocumentTable.summary.ilike(like_term),
                        DocumentTable.markdown_preview.ilike(like_term),
                        DocumentTable.table_type.ilike(like_term),
                    ),
                    0.40,
                ),
                else_=0.0,
            )

        score = score + case(
            (DocumentTable.table_type.in_(["eligibility_table", "documents_table", "deadline_table", "form_table"]), 0.25),
            else_=0.0,
        )
        return score

    def _row_match_score_expr(
        self,
        *,
        text_terms: list[str],
    ):
        score = literal(0.0)
        for term in text_terms[:6]:
            like_term = f"%{term}%"
            score = score + case(
                (
                    or_(
                        DocumentTableRow.row_summary.ilike(like_term),
                        func.cast(DocumentTableRow.row_json, literal("TEXT").type).ilike(like_term),
                        func.cast(DocumentTableRow.normalized_row_json, literal("TEXT").type).ilike(like_term),
                    ),
                    0.45,
                ),
                else_=0.0,
            )
        return score

    def _block_match_score_expr(
        self,
        *,
        text_terms: list[str],
        intent_type: QuestionIntentEnum,
    ):
        score = literal(0.0)

        for term in text_terms[:6]:
            like_term = f"%{term}%"
            score = score + case(
                (
                    DocumentBlock.content_clean.ilike(like_term),
                    0.35,
                ),
                else_=0.0,
            )

        if intent_type in {
            QuestionIntentEnum.PROCEDURE_QUESTION,
            QuestionIntentEnum.APPEAL_QUESTION,
        }:
            score = score + case(
                (DocumentBlock.block_type.in_(["paragraph", "list", "heading"]), 0.20),
                else_=0.0,
            )
        else:
            score = score + case(
                (DocumentBlock.block_type.in_(["paragraph", "list"]), 0.10),
                else_=0.0,
            )

        return score

    # --------------------------------------------------------
    # Candidate merge / rerank / selection
    # --------------------------------------------------------

    def _merge_and_deduplicate_candidates(
        self,
        candidates: list[RetrievedCandidate],
    ) -> list[RetrievedCandidate]:
        """
        Deduplicate exact same source objects and keep highest score.
        """
        merged: dict[tuple[str, UUID], RetrievedCandidate] = {}

        for candidate in candidates:
            key = (candidate.source_type, candidate.source_id)
            existing = merged.get(key)

            if existing is None:
                merged[key] = candidate
                continue

            if candidate.score > existing.score:
                merged[key] = candidate

        result = list(merged.values())
        result.sort(
            key=lambda x: (
                self._candidate_effective_score(x),
                x.source_type == "legal_fact",
                x.source_type == "table_row",
                x.source_type == "table",
                x.source_type == "block",
            ),
            reverse=True,
        )
        return result
      
    def _is_header_noise(
        self,
        candidate: RetrievedCandidate,
    ) -> bool:
        if candidate.source_type != "block":
            return False

        text = self._candidate_text_blob(candidate)
        snippet = (candidate.snippet or "").strip()
        metadata = candidate.metadata_json or {}

        if not text:
            return True

        style_name = str(metadata.get("style_name") or "").lower()
        is_heading_style = bool(metadata.get("is_heading_style"))

        header_patterns = [
            "министерство",
            "правительство",
            "красноярского края",
            "российской федерации",
            "постановление",
            "приказ",
            "утвержден",
            "утверждён",
            "административного регламента",
            "государственной услуги по предоставлению",
            "министр",
        ]

        exact_noise = {
            "министр",
            "и.л.пастухова",
            "красноярского края",
        }

        if snippet.lower() in exact_noise:
            return True

        if len(snippet) <= 40:
            return True

        if is_heading_style:
            return True

        if "title" in style_name:
            return True

        upper_ratio = self._uppercase_ratio(snippet)
        if len(snippet) <= 180 and upper_ratio >= 0.60:
            return True

        for pattern in header_patterns:
            if pattern in text and len(snippet) <= 220:
                return True

        return False


    def _uppercase_ratio(self, text: str) -> float:
        letters = [ch for ch in text if ch.isalpha()]
        if not letters:
            return 0.0

        upper = [ch for ch in letters if ch.isupper()]
        return len(upper) / len(letters)


    def _apply_semantic_priority(
        self,
        candidates: list[RetrievedCandidate],
    ) -> list[RetrievedCandidate]:
        for candidate in candidates:
            if candidate.source_type == "table_row":
                candidate.score *= 1.45
            elif candidate.source_type == "legal_fact":
                candidate.score *= 1.20
            elif candidate.source_type == "table":
                candidate.score *= 1.05
            elif candidate.source_type == "block":
                candidate.score *= 0.85

        return candidates

    async def _maybe_rerank(
        self,
        *,
        question_text: str,
        candidates: list[RetrievedCandidate],
    ) -> list[RetrievedCandidate]:
        if not candidates:
            return []

        if self.reranker is None:
            return sorted(
                candidates,
                key=self._candidate_effective_score,
                reverse=True,
            )

        reranked = await self.reranker.rerank(
            question_text=question_text,
            candidates=candidates,
        )

        return sorted(
            reranked,
            key=self._candidate_effective_score,
            reverse=True,
        )

    def _build_document_aggregates(
        self,
        candidates: list[RetrievedCandidate],
    ) -> dict[UUID, dict[str, Any]]:
        """
        Build document-level aggregate stats from reranked candidates.

        Goal:
        - prefer evidence concentrated in one strong document
        - avoid noisy spread across many weak documents
        """
        aggregates: dict[UUID, dict[str, Any]] = {}

        for index, candidate in enumerate(candidates):
            effective_score = self._candidate_effective_score(candidate)

            stats = aggregates.setdefault(
                candidate.document_id,
                {
                    "document_id": candidate.document_id,
                    "document_name": candidate.document_name,
                    "doc_uid_base": candidate.doc_uid_base,
                    "candidate_count": 0,
                    "best_score": 0.0,
                    "total_score": 0.0,
                    "weighted_total_score": 0.0,
                    "source_types": set(),
                    "first_rank_index": index,
                },
            )

            stats["candidate_count"] += 1
            stats["best_score"] = max(stats["best_score"], effective_score)
            stats["total_score"] += effective_score
            stats["weighted_total_score"] += effective_score / (1.0 + (index * 0.15))
            stats["source_types"].add(candidate.source_type)
            stats["first_rank_index"] = min(stats["first_rank_index"], index)

        for stats in aggregates.values():
            stats["source_type_count"] = len(stats["source_types"])
            stats["aggregate_score"] = (
                stats["weighted_total_score"]
                + (stats["best_score"] * 0.60)
                + (min(stats["candidate_count"], 4) * 0.12)
                + (stats["source_type_count"] * 0.08)
            )
            stats["source_types"] = sorted(stats["source_types"])

        return aggregates
        
    def _has_meaningful_lexical_match(
        self,
        candidate: RetrievedCandidate,
        query_terms: list[str],
    ) -> bool:
        text = self._candidate_text_blob(candidate)

        matched_terms = 0
        for term in query_terms:
            if not term or len(term) < 3:
                continue
            if term in text:
                matched_terms += 1

        return matched_terms >= 1

    def _select_priority_documents(
        self,
        *,
        payload: RetrievalInput,
        strategy: RetrievalStrategy,
        document_stats: dict[UUID, dict[str, Any]],
    ) -> list[UUID]:
        """
        Pick 1-2 priority documents before final candidate balancing.

        Conservative logic:
        - prefer one dominant document
        - allow second document only if it is reasonably competitive
        """
        if not document_stats:
            return []

        ranked_stats = sorted(
            document_stats.values(),
            key=lambda item: (
                item["aggregate_score"],
                item["best_score"],
                item["candidate_count"],
                -item["first_rank_index"],
            ),
            reverse=True,
        )

        first = ranked_stats[0]
        selected_document_ids = [first["document_id"]]

        if len(ranked_stats) == 1:
            return selected_document_ids

        second = ranked_stats[1]

        first_score = float(first["aggregate_score"])
        second_score = float(second["aggregate_score"])

        second_is_competitive = second_score >= max(0.75, first_score * 0.62)

        strategy_allows_two_docs = strategy.strategy_code in {
            "procedure_balanced",
            "balanced_default",
            "rejection_priority",
            "eligibility_priority",
        }

        if second_is_competitive and strategy_allows_two_docs:
            selected_document_ids.append(second["document_id"])

        return selected_document_ids

    def _select_final_candidates(
            self,
            *,
            payload: RetrievalInput,
            strategy: RetrievalStrategy,
            candidates: list[RetrievedCandidate],
            document_stats: dict[UUID, dict[str, Any]],
            priority_document_ids: list[UUID],
        ) -> list[RetrievedCandidate]:
            """
            Select final balanced evidence set.

            Special rule for DOCUMENTS_QUESTION:
            - prefer table_row evidence from the priority document
            - allow a noticeably larger pool of row candidates
            - suppress noisy blocks/facts because deterministic builder
              needs rows, not a mixed bag of snippets
            """
            if not candidates:
                return []

            is_documents_question = payload.intent_type == QuestionIntentEnum.DOCUMENTS_QUESTION

            if is_documents_question:
                # Для questions типа "какие документы нужны..."
                # builder должен получить достаточный пул row-кандидатов.
                type_caps = {
                    "legal_fact": 1,
                    "table": min(2, payload.final_top_k),
                    "table_row": min(max(10, payload.final_top_k), 14),
                    "block": 1,
                }
            else:
                type_caps = {
                    "legal_fact": max(2, min(5, payload.final_top_k)),
                    "table": max(2, min(4, payload.final_top_k)),
                    "table_row": max(2, min(5, payload.final_top_k)),
                    "block": max(2, min(5, payload.final_top_k)),
                }

            priority_document_set = set(priority_document_ids)
            min_score_threshold = self._get_min_candidate_score_threshold(
                payload=payload,
                strategy=strategy,
                document_stats=document_stats,
            )

            priority_candidates: list[RetrievedCandidate] = []
            non_priority_candidates: list[RetrievedCandidate] = []

            for candidate in candidates:
                if candidate.document_id in priority_document_set:
                    priority_candidates.append(candidate)
                else:
                    non_priority_candidates.append(candidate)

            ordered_candidates = [*priority_candidates, *non_priority_candidates]

            # Для documents-question поднимаем наверх table_row из documents-table,
            # чтобы сначала отбирать именно answer-bearing rows.
            if is_documents_question:
                ordered_candidates.sort(
                    key=lambda candidate: (
                        0 if (
                            candidate.document_id in priority_document_set
                            and candidate.source_type == "table_row"
                            and self._has_table_semantic_type(candidate, "documents")
                        ) else
                        1 if (
                            candidate.document_id in priority_document_set
                            and candidate.source_type == "table"
                            and self._has_table_semantic_type(candidate, "documents")
                        ) else
                        2 if candidate.document_id in priority_document_set else 3,
                        -self._candidate_effective_score(candidate),
                    )
                )

            selected: list[RetrievedCandidate] = []
            selected_keys: set[tuple[str, UUID]] = set()
            type_counts: dict[str, int] = {}
            document_counts: dict[UUID, int] = {}

            for candidate in ordered_candidates:
                candidate_key = (candidate.source_type, candidate.source_id)
                if candidate_key in selected_keys:
                    continue

                effective_score = self._candidate_effective_score(candidate)
                if effective_score < min_score_threshold:
                    continue

                current_type_count = type_counts.get(candidate.source_type, 0)
                if current_type_count >= type_caps.get(candidate.source_type, payload.final_top_k):
                    continue

                current_doc_count = document_counts.get(candidate.document_id, 0)
                if is_documents_question:
                    max_per_document = 12 if candidate.document_id in priority_document_set else 4
                else:
                    max_per_document = 6 if candidate.document_id in priority_document_set else 3

                if current_doc_count >= max_per_document:
                    continue

                selected.append(candidate)
                selected_keys.add(candidate_key)
                type_counts[candidate.source_type] = current_type_count + 1
                document_counts[candidate.document_id] = current_doc_count + 1

                if len(selected) >= payload.final_top_k:
                    break

            # Fallback: если балансировка оказалась слишком строгой,
            # добираем ещё кандидатов без type/doc caps, но только до final_top_k.
            if len(selected) < min(3, payload.final_top_k):
                for candidate in ordered_candidates:
                    if len(selected) >= payload.final_top_k:
                        break

                    candidate_key = (candidate.source_type, candidate.source_id)
                    if candidate_key in selected_keys:
                        continue

                    selected.append(candidate)
                    selected_keys.add(candidate_key)

            return selected

    def _build_evidence_package(
        self,
        *,
        payload: RetrievalInput,
        strategy: RetrievalStrategy,
        query_bundle: dict[str, Any],
        raw_candidates: list[RetrievedCandidate],
        merged_candidates: list[RetrievedCandidate],
        reranked_candidates: list[RetrievedCandidate],
        final_candidates: list[RetrievedCandidate],
        document_stats: dict[UUID, dict[str, Any]],
        priority_document_ids: list[UUID],
    ) -> EvidencePackage:
        fact_ids: list[UUID] = []
        table_ids: list[UUID] = []
        row_ids: list[UUID] = []
        block_ids: list[UUID] = []
        document_ids: list[UUID] = []

        seen_documents: set[UUID] = set()

        for candidate in final_candidates:
            if candidate.source_type == "legal_fact":
                fact_ids.append(candidate.source_id)
            elif candidate.source_type == "table":
                table_ids.append(candidate.source_id)
            elif candidate.source_type == "table_row":
                row_ids.append(candidate.source_id)
            elif candidate.source_type == "block":
                block_ids.append(candidate.source_id)

            if candidate.document_id not in seen_documents:
                seen_documents.add(candidate.document_id)
                document_ids.append(candidate.document_id)

        evidence_quality, guard_reason = self._classify_evidence_quality(
            candidates=final_candidates,
            priority_document_ids=priority_document_ids,
            document_stats=document_stats,
        )

        ranked_document_stats = sorted(
            document_stats.values(),
            key=lambda item: item["aggregate_score"],
            reverse=True,
        )

        top_candidate_score = self._candidate_effective_score(final_candidates[0]) if final_candidates else 0.0
        second_candidate_score = self._candidate_effective_score(final_candidates[1]) if len(final_candidates) > 1 else 0.0

        top_document_score = float(ranked_document_stats[0]["aggregate_score"]) if ranked_document_stats else 0.0
        second_document_score = float(ranked_document_stats[1]["aggregate_score"]) if len(ranked_document_stats) > 1 else 0.0

        top_document_id = ranked_document_stats[0]["document_id"] if ranked_document_stats else None
        top_document_selected_count = sum(
            1 for candidate in final_candidates if candidate.document_id == top_document_id
        ) if top_document_id is not None else 0

        top_document_share = (
            round(top_document_selected_count / len(final_candidates), 4)
            if final_candidates
            else 0.0
        )

        strong_candidate_threshold = 0.75
        strong_candidate_count = sum(
            1 for candidate in final_candidates
            if self._candidate_effective_score(candidate) >= strong_candidate_threshold
        )

        metrics_json = {
            "raw_candidates_count": len(raw_candidates),
            "merged_candidates_count": len(merged_candidates),
            "reranked_candidates_count": len(reranked_candidates),
            "final_candidates_count": len(final_candidates),
            "selected_fact_ids_count": len(fact_ids),
            "selected_table_ids_count": len(table_ids),
            "selected_row_ids_count": len(row_ids),
            "selected_block_ids_count": len(block_ids),
            "selected_document_ids_count": len(document_ids),
            "distinct_document_count_before_selection": len(document_stats),
            "priority_document_count": len(priority_document_ids),
            "top_candidate_score": round(top_candidate_score, 4),
            "second_candidate_score": round(second_candidate_score, 4),
            "top_document_score": round(top_document_score, 4),
            "second_document_score": round(second_document_score, 4),
            "top_document_share": top_document_share,
            "strong_candidate_count": strong_candidate_count,
            "evidence_quality": evidence_quality,
        }

        debug_payload_json = {
            "strategy_code": strategy.strategy_code,
            "priority_document_ids": [str(x) for x in priority_document_ids],
            "evidence_quality": evidence_quality,
            "guard_reason": guard_reason,
            "query_bundle": {
                "normalized_text": query_bundle.get("normalized_text"),
                "query_terms": query_bundle.get("query_terms"),
                "table_question_profile": query_bundle.get("table_question_profile"),
                "submission_channel": query_bundle.get("submission_channel"),
                "requested_column_hints": query_bundle.get("requested_column_hints"),
                "table_scope_hints": query_bundle.get("table_scope_hints"),
            },
            "selected_candidates_preview": [
                {
                    "source_type": c.source_type,
                    "source_id": str(c.source_id),
                    "document_id": str(c.document_id),
                    "score": c.score,
                    "rerank_score": c.rerank_score,
                    "effective_score": round(self._candidate_effective_score(c), 4),
                    "document_name": c.document_name,
                    "title": c.title,
                    "snippet": self._shorten_text(c.snippet, limit=300),
                    "citation_json": c.citation_json,
                    "metadata_preview": {
                        "table_title": (c.metadata_json or {}).get("table_title"),
                        "table_name": (c.metadata_json or {}).get("table_name"),
                        "table_number": (c.metadata_json or {}).get("table_number"),
                        "appendix_number": (c.metadata_json or {}).get("appendix_number"),
                        "row_summary": (c.metadata_json or {}).get("row_summary"),
                        "table_semantic_type": (c.metadata_json or {}).get("table_semantic_type"),
                        "column_headers": (c.metadata_json or {}).get("column_headers"),
                        "cells_text": (c.metadata_json or {}).get("cells_text"),
                        "cells_by_header": (c.metadata_json or {}).get("cells_by_header"),
                        "cells_by_header_normalized": (c.metadata_json or {}).get("cells_by_header_normalized"),
                        "cells_by_header_key": (c.metadata_json or {}).get("cells_by_header_key"),
                    },
                }
                for c in final_candidates
            ],
            "document_stats_preview": [
                {
                    "document_id": str(item["document_id"]),
                    "document_name": item["document_name"],
                    "aggregate_score": round(float(item["aggregate_score"]), 4),
                    "best_score": round(float(item["best_score"]), 4),
                    "candidate_count": int(item["candidate_count"]),
                    "source_type_count": int(item["source_type_count"]),
                    "source_types": item["source_types"],
                }
                for item in ranked_document_stats[:5]
            ],
        }

        return EvidencePackage(
            question_event_id=payload.question_event_id,
            strategy_code=strategy.strategy_code,
            selected_candidates=final_candidates,
            selected_fact_ids=fact_ids,
            selected_table_ids=table_ids,
            selected_row_ids=row_ids,
            selected_block_ids=block_ids,
            selected_document_ids=document_ids,
            metrics_json=metrics_json,
            debug_payload_json=debug_payload_json,
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    def _validate_input(
        self,
        payload: RetrievalInput,
    ) -> None:
        if not payload.question_text_raw or not payload.question_text_raw.strip():
            raise RetrievalValidationError("question_text_raw must not be empty.")

        if (
            payload.top_k_facts < 0
            or payload.top_k_tables < 0
            or payload.top_k_rows < 0
            or payload.top_k_blocks < 0
        ):
            raise RetrievalValidationError("top_k values must be >= 0.")

        if payload.final_top_k < 1:
            raise RetrievalValidationError("final_top_k must be >= 1.")

    # --------------------------------------------------------
    # Quality helpers
    # --------------------------------------------------------

    def _candidate_effective_score(
        self,
        candidate: RetrievedCandidate,
    ) -> float:
        return float(
            candidate.rerank_score if candidate.rerank_score is not None else candidate.score
        )

    def _get_min_candidate_score_threshold(
        self,
        *,
        payload: RetrievalInput,
        strategy: RetrievalStrategy,
        document_stats: dict[UUID, dict[str, Any]],
    ) -> float:
        """
        Conservative threshold for final evidence selection.

        This is intentionally low enough not to kill structured evidence,
        but high enough to reduce obviously noisy tails.
        """
        base_threshold = 0.35

        if strategy.strategy_code in {
            "documents_priority",
            "deadlines_priority",
            "forms_priority",
        }:
            base_threshold = 0.45
        elif strategy.strategy_code in {
            "eligibility_priority",
            "rejection_priority",
        }:
            base_threshold = 0.40

        if not document_stats:
            return base_threshold

        ranked = sorted(
            document_stats.values(),
            key=lambda item: item["aggregate_score"],
            reverse=True,
        )
        top_document_best_score = float(ranked[0]["best_score"])

        adaptive_floor = min(base_threshold, max(0.25, top_document_best_score * 0.35))
        return adaptive_floor

    def _classify_evidence_quality(
        self,
        *,
        candidates: list[RetrievedCandidate],
        priority_document_ids: list[UUID],
        document_stats: dict[UUID, dict[str, Any]],
    ) -> tuple[str, str]:
        """
        Classify evidence package strength without changing public API.

        Returns:
            (evidence_quality, guard_reason)
        """
        if not candidates:
            return ("insufficient", "no_candidates")

        top_score = self._candidate_effective_score(candidates[0])
        if top_score < 0.25:
            return ("insufficient", "top_score_too_low")

        if len(candidates) == 1:
            return ("weak", "single_candidate_only")

        distinct_document_count = len({candidate.document_id for candidate in candidates})
        if distinct_document_count > 2 and len(priority_document_ids) <= 1:
            return ("weak", "document_spread_too_wide")

        top_document_id = candidates[0].document_id
        top_document_selected_count = sum(
            1 for candidate in candidates if candidate.document_id == top_document_id
        )
        top_document_share = top_document_selected_count / len(candidates)

        if top_document_share < 0.40 and distinct_document_count > 2:
            return ("weak", "low_document_concentration")

        strong_candidate_count = sum(
            1 for candidate in candidates
            if self._candidate_effective_score(candidate) >= 0.75
        )
        if strong_candidate_count == 0 and top_score < 0.55:
            return ("weak", "no_strong_candidates")

        ranked_document_stats = sorted(
            document_stats.values(),
            key=lambda item: item["aggregate_score"],
            reverse=True,
        )
        if len(ranked_document_stats) >= 2:
            top_document_score = float(ranked_document_stats[0]["aggregate_score"])
            second_document_score = float(ranked_document_stats[1]["aggregate_score"])
            if second_document_score > top_document_score * 0.95 and distinct_document_count > 2:
                return ("weak", "no_clear_document_leader")

        return ("strong", "ok")

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _normalize_text(
        self,
        value: Optional[str],
    ) -> str:
        if not value:
            return ""
        return " ".join(value.strip().lower().split())

    def _shorten_text(
        self,
        value: Optional[str],
        *,
        limit: int,
    ) -> Optional[str]:
        if value is None:
            return None

        text = value.strip()
        if len(text) <= limit:
            return text

        if limit <= 3:
            return text[:limit]

        return text[: limit - 3].rstrip() + "..."

    def _datetime_to_iso(
        self,
        value: Any,
    ) -> Optional[str]:
        if value is None:
            return None
        try:
            return value.isoformat()
        except AttributeError:
            return str(value)
    
    def _apply_intent_semantic_rerank(
        self,
        *,
        candidates: list[RetrievedCandidate],
        payload: RetrievalInput,
        query_bundle: dict[str, Any],
    ) -> list[RetrievedCandidate]:
        """
        Local intent-aware rerank.

        Goal:
        - strengthen answer-bearing rows/tables
        - suppress noisy blocks
        - especially improve document-table retrieval
        """
        if not candidates:
            return candidates

        intent_type = payload.intent_type
        table_question_profile = query_bundle.get("table_question_profile")
        submission_channel = query_bundle.get("submission_channel")
        requested_column_hints = query_bundle.get("requested_column_hints") or []

        reranked: list[RetrievedCandidate] = []

        for candidate in candidates:
            score = candidate.rerank_score if candidate.rerank_score is not None else candidate.score
            text = self._candidate_text_blob(candidate)

            # -------------------------------
            # Global source priors
            # -------------------------------
            if candidate.source_type == "table_row":
                score += 0.05
            elif candidate.source_type == "table":
                score += 0.03

            # -------------------------------
            # Documents questions
            # -------------------------------
            if intent_type == QuestionIntentEnum.DOCUMENTS_QUESTION:
                # Strong positive signal: extractor already classified table as documents
                if self._has_table_semantic_type(candidate, "documents"):
                    if candidate.source_type == "table_row":
                        score += 0.65
                    elif candidate.source_type == "table":
                        score += 0.35

                # Strong negative signal: abbreviations table is not an answer to document list question
                if self._is_abbreviation_table_candidate(candidate):
                    if candidate.source_type == "table_row":
                        score -= 1.10
                    elif candidate.source_type == "table":
                        score -= 0.90
                    else:
                        score -= 0.40

                # Penalize service/header-like rows inside document tables
                if self._looks_like_service_documents_row(candidate):
                    if candidate.source_type == "table_row":
                        score -= 0.60
                    elif candidate.source_type == "table":
                        score -= 0.30

                # Existing lexical/table heuristics
                if self._is_required_documents_table_candidate(candidate):
                    if candidate.source_type == "table_row":
                        score += 0.45
                    elif candidate.source_type == "table":
                        score += 0.25
                    elif candidate.source_type == "legal_fact":
                        score += 0.08
                    elif candidate.source_type == "block":
                        score -= 0.20

                # Additional boosts if candidate explicitly contains requested column hints
                for hint in requested_column_hints:
                    hint_norm = self._normalize_text(hint)
                    if hint_norm and hint_norm in text:
                        score += 0.06

                # Channel-specific questions
                if table_question_profile == "documents_by_submission_channel":
                    if self._has_submission_channel_match(candidate, submission_channel):
                        if candidate.source_type == "table_row":
                            score += 0.35
                        elif candidate.source_type == "table":
                            score += 0.20
                        else:
                            score += 0.08
                    else:
                        if candidate.source_type == "block":
                            score -= 0.10
                            
            elif intent_type == QuestionIntentEnum.DEADLINE_QUESTION:
                metadata = candidate.metadata_json or {}
                text_norm = self._normalize_text(text)

                if self._has_table_semantic_type(candidate, "deadlines") or self._has_table_semantic_type(candidate, "deadline"):
                    if candidate.source_type == "table_row":
                        score += 0.45
                    elif candidate.source_type == "table":
                        score += 0.22

                cells = metadata.get("cells_by_semantic_key") or {}
                if isinstance(cells, dict) and cells.get("deadline_value"):
                    score += 0.25

                if any(marker in text_norm for marker in [
                    "срок",
                    "в течение",
                    "рабочих дней",
                    "календарных дней",
                    "не позднее",
                    "не более",
                ]):
                    score += 0.18 if candidate.source_type == "table_row" else 0.08

                if self._looks_like_service_deadline_row(candidate):
                    if candidate.source_type == "table_row":
                        score -= 0.55
                    elif candidate.source_type == "table":
                        score -= 0.25
                    else:
                        score -= 0.10

                question_norm = self._normalize_text(
                    payload.question_text_normalized or payload.question_text_raw
                )
                for term in [
                    "принятия решения",
                    "рассмотрения заявления",
                    "предоставления услуги",
                    "регистрации заявления",
                    "выплаты",
                    "перечисления",
                ]:
                    if term in question_norm and term in text_norm:
                        score += 0.10

            # -------------------------------
            # Rejection
            # -------------------------------
            elif intent_type == QuestionIntentEnum.REJECTION_QUESTION:
                if any(marker in text for marker in [
                    "основания отказа",
                    "отказа в предоставлении",
                    "отказа в приеме",
                    "непредставление документов",
                    "недостоверные сведения",
                ]):
                    score += 0.18

            # -------------------------------
            # Eligibility
            # -------------------------------
            elif intent_type == QuestionIntentEnum.ELIGIBILITY_QUESTION:
                if any(marker in text for marker in [
                    "имеет право",
                    "категории заявителей",
                    "заявителями являются",
                    "право на получение",
                ]):
                    score += 0.18

            candidate.rerank_score = round(score, 6)
            reranked.append(candidate)

        reranked.sort(
            key=lambda item: (
                item.rerank_score if item.rerank_score is not None else item.score,
                item.score,
                1 if item.source_type == "table_row" else 0,
                1 if item.source_type == "table" else 0,
            ),
            reverse=True,
        )
        return reranked
    
    def _has_min_intent_anchor_match(
        self,
        candidate: RetrievedCandidate,
        *,
        query_terms: list[str],
        min_matches: int = 2,
    ) -> bool:
        text = self._candidate_text_blob(candidate)
        matched = 0

        for term in query_terms:
            if not term or len(term) < 3:
                continue
            if term in text:
                matched += 1
                if matched >= min_matches:
                    return True

        return False
        
    def _has_documents_anchor_match(
        self,
        candidate: RetrievedCandidate,
        query_terms: list[str],
    ) -> bool:
        text = self._candidate_text_blob(candidate)

        strong_terms = [
            "документы",
            "перечень документов",
            "необходимые документы",
            "документов необходимых",
            "заявление",
            "заявителем",
            "представляемые документы",
            "прилагаемые документы",
            "наименование документа",
            "электронный образ документа",
            "сведения о документе",
            "таблица 2",
            "приложение n 2",
            "приложения n 2",
        ]

        matches = 0

        for term in strong_terms:
            if term in text:
                matches += 1

        for term in query_terms:
            if len(term) >= 4 and term in text:
                matches += 1

        if "наименование документа" in text:
            matches += 1

        return matches >= 2
        
    def _deadline_candidate_rank_key(self, candidate: RetrievedCandidate) -> tuple[int, float]:
        metadata = candidate.metadata_json or {}
        cells = metadata.get("cells_by_semantic_key") or {}

        has_deadline_value = isinstance(cells, dict) and bool(cells.get("deadline_value"))
        is_deadline_row = candidate.source_type == "table_row" and has_deadline_value
        is_block = candidate.source_type == "block"

        priority_bucket = 2
        if is_deadline_row:
            priority_bucket = 0
        elif is_block:
            priority_bucket = 1

        return (priority_bucket, -float(candidate.score or 0.0))