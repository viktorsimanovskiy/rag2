from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(slots=True)
class DeadlineAnswerItem:
    deadline_value: str
    scope_text: str
    source_type: str
    citation_json: dict[str, Any] = field(default_factory=dict)

    fact_type: str | None = None
    deadline_kind: str = "other"
    kind_confidence: float = 0.0
    is_service_core_deadline: bool = False
    candidate_score: float = 0.0

    table_title: str | None = None
    table_number: str | None = None

    source_row_ids: list[str] = field(default_factory=list)
    source_block_ids: list[str] = field(default_factory=list)
    source_fact_ids: list[str] = field(default_factory=list)
    source_scores: list[float] = field(default_factory=list)


@dataclass(slots=True)
class DeadlinesAnswerBuildResult:
    can_answer: bool
    question_deadline_kind: str

    primary_item: DeadlineAnswerItem | None = None
    alternative_items: list[DeadlineAnswerItem] = field(default_factory=list)

    dropped_rows_debug: list[dict[str, Any]] = field(default_factory=list)
    merged_items_debug: list[dict[str, Any]] = field(default_factory=list)

    ambiguity_reason: str | None = None
    reason: str | None = None

    def debug_payload(self) -> dict[str, Any]:
        return {
            "can_answer": self.can_answer,
            "question_deadline_kind": self.question_deadline_kind,
            "reason": self.reason,
            "ambiguity_reason": self.ambiguity_reason,
            "primary_item": self._item_to_debug_dict(self.primary_item),
            "alternative_items": [
                self._item_to_debug_dict(item)
                for item in self.alternative_items
            ],
            "dropped_rows_debug": self.dropped_rows_debug,
            "merged_items_debug": self.merged_items_debug,
        }

    def _item_to_debug_dict(
        self,
        item: DeadlineAnswerItem | None,
    ) -> dict[str, Any] | None:
        if item is None:
            return None

        return {
            "deadline_value": item.deadline_value,
            "scope_text": item.scope_text,
            "source_type": item.source_type,
            "fact_type": item.fact_type,
            "deadline_kind": item.deadline_kind,
            "kind_confidence": item.kind_confidence,
            "is_service_core_deadline": item.is_service_core_deadline,
            "candidate_score": item.candidate_score,
            "table_title": item.table_title,
            "table_number": item.table_number,
            "citation_json": item.citation_json,
            "source_row_ids": item.source_row_ids,
            "source_block_ids": item.source_block_ids,
            "source_fact_ids": item.source_fact_ids,
            "source_scores": item.source_scores,
        }


class TableDeadlinesAnswerBuilder:
    """
    Deterministic builder for deadline questions.

    Цель:
    - выбрать один основной срок по типу вопроса;
    - не смешивать procedural noise с целевым сроком;
    - уметь брать сроки из legal_fact, table_row и block.
    """

    _IGNORED_SCOPE_KEYS = {
        "deadline_value",
        "row_number",
    }

    _SERVICE_VALUES = {
        "срок",
        "сроки",
        "срок предоставления",
        "срок предоставления государственной услуги",
        "максимальный срок",
        "максимальный срок предоставления государственной услуги",
        "рабочих дней",
        "календарных дней",
    }

    _DEADLINE_KIND_LABELS: dict[str, str] = {
        "decision": "принятия решения",
        "notification": "уведомления",
        "payment": "выплаты",
        "registration": "регистрации",
        "correction": "исправления ошибок",
        "other": "срока",
    }

    _BLOCK_DEADLINE_PATTERNS = [
        re.compile(
            r"в течение\s+\d+\s+(?:рабоч(?:их|его)|календарн(?:ых|ого))\s+дн(?:я|ей)",
            re.IGNORECASE,
        ),
        re.compile(r"в течение\s+\d+\s+дн(?:я|ей)", re.IGNORECASE),
        re.compile(
            r"не более\s+\d+\s+(?:рабоч(?:их|его)|календарн(?:ых|ого))\s+дн(?:я|ей)",
            re.IGNORECASE,
        ),
        re.compile(
            r"не позднее\s+26(?:-го)?\s+числа(?:\s+месяца)?",
            re.IGNORECASE,
        ),
        re.compile(r"не позднее\s+[^.;]{3,140}", re.IGNORECASE),
    ]

    _OFFTOPIC_MARKERS = (
        "исправлении опечаток",
        "опечаток и ошибок",
        "исправлении ошибок",
        "об отсутствии ошибок",
        "уведомления об отсутствии ошибок",
        "нового документа взамен",
        "новый документ",
        "проверки подписи",
        "подлинности простой электронной подписи",
        "усиленной квалифицированной электронной подписи",
        "статьи 9",
        "статьи 11",
        "федерального закона n 63-фз",
    )

    _NOTIFICATION_STRONG_MARKERS = (
        "уведомление о принятом решении",
        "о принятом решении заявитель",
        "заявитель уведомляется",
        "заявитель или представитель уведомляется",
        "направляется заявителю",
        "направляет заявителю",
        "уведомление направляется",
        "уведомляет заявителя",
        "уведомляет представителя",
    )
    _NOTIFICATION_WEAK_MARKERS = (
        "уведомление",
        "уведомляет",
        "уведомить",
        "извещает",
        "извещение",
        "сообщение о решении",
        "информирует",
    )

    _PAYMENT_STRONG_MARKERS = (
        "выплачивает",
        "выплачивается",
        "выплата",
        "выплаты",
        "перечисление",
        "зачисление",
        "ежемесячно",
        "не позднее 26-го числа",
        "не позднее 26 числа",
        "26-го числа текущего месяца",
        "26 числа текущего месяца",
    )

    _DECISION_STRONG_MARKERS = (
        "решение о предоставлении",
        "решение о назначении",
        "решение принимается",
        "принятия решения",
        "принятие решения",
        "рассмотрения заявления",
        "рассмотрение заявления",
        "назначении едв",
        "назначение едв",
    )

    _REGISTRATION_MARKERS = (
        "регистрация запроса",
        "регистрация заявления",
        "регистрируется",
        "зарегистрировано",
    )

    _CORRECTION_MARKERS = (
        "исправление ошибок",
        "исправлении ошибок",
        "исправление опечаток",
        "исправлении опечаток",
        "об отсутствии ошибок",
        "опечаток и ошибок",
    )

    def build(
        self,
        *,
        candidates: list[Any],
        question_text: Optional[str] = None,
    ) -> DeadlinesAnswerBuildResult:
        normalized_question = self._normalize(question_text)
        question_deadline_kind = self._question_deadline_kind(normalized_question) or "other"

        raw_items: list[DeadlineAnswerItem] = []
        dropped_rows_debug: list[dict[str, Any]] = []

        for candidate in candidates:
            item = self._build_item_from_candidate(candidate)
            if item is None:
                dropped_rows_debug.append(
                    {
                        "source_type": self._candidate_source_type(candidate),
                        "source_id": str(self._candidate_attr(candidate, "source_id", "") or ""),
                        "reason": "candidate_not_converted",
                    }
                )
                continue
            raw_items.append(item)

        if not raw_items:
            return DeadlinesAnswerBuildResult(
                can_answer=False,
                question_deadline_kind=question_deadline_kind,
                reason="no_deadline_items",
                dropped_rows_debug=dropped_rows_debug,
                merged_items_debug=[],
            )

        merged_items, merge_debug = self._merge_similar_items(raw_items)

        if not merged_items:
            return DeadlinesAnswerBuildResult(
                can_answer=False,
                question_deadline_kind=question_deadline_kind,
                reason="no_merged_deadline_items",
                dropped_rows_debug=dropped_rows_debug,
                merged_items_debug=merge_debug,
            )

        ranked_items = sorted(
            merged_items,
            key=lambda item: self._rank_item(
                item=item,
                question_text=normalized_question,
            ),
            reverse=True,
        )

        primary_item = self._select_primary_item(
            ranked_items=ranked_items,
            question_deadline_kind=question_deadline_kind,
        )
        alternative_items = [
            item for item in ranked_items
            if item is not primary_item
        ]

        ambiguity_reason: str | None = None
        if alternative_items and any(
            self._normalize(item.deadline_value) != self._normalize(primary_item.deadline_value)
            for item in alternative_items
        ):
            ambiguity_reason = "multiple_distinct_deadlines"

        return DeadlinesAnswerBuildResult(
            can_answer=True,
            question_deadline_kind=question_deadline_kind,
            primary_item=primary_item,
            alternative_items=alternative_items,
            dropped_rows_debug=dropped_rows_debug,
            merged_items_debug=[
                *merge_debug,
                {
                    "raw_items_count": len(raw_items),
                    "merged_items_count": len(merged_items),
                    "ranked_items_preview": [
                        {
                            "deadline_value": item.deadline_value,
                            "scope_text": item.scope_text,
                            "source_type": item.source_type,
                            "fact_type": item.fact_type,
                            "deadline_kind": item.deadline_kind,
                            "kind_confidence": item.kind_confidence,
                            "is_service_core_deadline": item.is_service_core_deadline,
                            "candidate_score": item.candidate_score,
                            "rank_score": self._rank_item(
                                item=item,
                                question_text=normalized_question,
                            ),
                        }
                        for item in ranked_items[:10]
                    ],
                },
            ],
            ambiguity_reason=ambiguity_reason,
            reason=None,
        )

    def render_text(
        self,
        *,
        result: DeadlinesAnswerBuildResult,
    ) -> Optional[str]:
        if not result.can_answer or result.primary_item is None:
            return None

        primary = result.primary_item
        render_kind = self._resolve_final_deadline_kind(
            item=primary,
            question_deadline_kind=result.question_deadline_kind,
        )

        primary_label = self._DEADLINE_KIND_LABELS.get(
            render_kind,
            self._DEADLINE_KIND_LABELS["other"],
        )

        if primary.scope_text:
            return (
                f"Срок {primary_label} по найденным источникам: "
                f"{primary.deadline_value} ({primary.scope_text})."
            )

        return f"Срок {primary_label} по найденным источникам: {primary.deadline_value}."
        
    def _should_render_single_primary(
        self,
        *,
        primary: DeadlineAnswerItem,
        alternatives: list[DeadlineAnswerItem],
        question_deadline_kind: str,
    ) -> bool:
        if not alternatives:
            return True

        primary_kind = self._resolve_render_deadline_kind(
            item=primary,
            question_deadline_kind=question_deadline_kind,
        )

        visible_alternatives = self._select_visible_alternatives(
            primary=primary,
            alternatives=alternatives,
            question_deadline_kind=question_deadline_kind,
        )
        if not visible_alternatives:
            return True

        first_alt = visible_alternatives[0]
        first_alt_kind = self._resolve_render_deadline_kind(
            item=first_alt,
            question_deadline_kind=question_deadline_kind,
        )

        if (
            primary_kind == first_alt_kind
            and self._normalize(primary.deadline_value) == self._normalize(first_alt.deadline_value)
        ):
            return True

        if primary.candidate_score >= first_alt.candidate_score + 0.35:
            return True

        if primary.kind_confidence >= 0.90 and first_alt.kind_confidence <= 0.55:
            return True

        return False

    def _select_visible_alternatives(
        self,
        *,
        primary: DeadlineAnswerItem,
        alternatives: list[DeadlineAnswerItem],
        question_deadline_kind: str,
    ) -> list[DeadlineAnswerItem]:
        primary_kind = self._resolve_render_deadline_kind(
            item=primary,
            question_deadline_kind=question_deadline_kind,
        )
        primary_value = self._normalize(primary.deadline_value)

        visible: list[DeadlineAnswerItem] = []

        for item in alternatives:
            item_kind = self._resolve_render_deadline_kind(
                item=item,
                question_deadline_kind=question_deadline_kind,
            )
            item_value = self._normalize(item.deadline_value)

            if item_value == primary_value and item_kind == primary_kind:
                continue

            if question_deadline_kind != "other" and item_kind == "correction":
                continue

            if item.candidate_score < 0.15:
                continue

            visible.append(item)
            if len(visible) >= 2:
                break

        return visible

    def _render_bulleted_item(
        self,
        item: DeadlineAnswerItem,
        *,
        question_deadline_kind: str,
    ) -> str:
        render_kind = self._render_deadline_kind(
            item=item,
            question_deadline_kind=question_deadline_kind,
        )
        label = self._DEADLINE_KIND_LABELS.get(
            render_kind,
            self._DEADLINE_KIND_LABELS["other"],
        )

        if item.scope_text:
            return f"- {item.deadline_value} — срок {label} ({item.scope_text})"
        return f"- {item.deadline_value} — срок {label}"
        
    def _fact_type_to_deadline_kind(
        self,
        fact_type: str | None,
    ) -> str:
        mapping = {
            "decision_deadline": "decision",
            "notification_deadline": "notification",
            "payment_deadline": "payment",
            "registration_deadline": "registration",
            "correction_deadline": "correction",
        }
        return mapping.get(self._normalize(fact_type), "other")

    def _resolve_final_deadline_kind(
        self,
        *,
        item: DeadlineAnswerItem,
        question_deadline_kind: str,
    ) -> str:
        fact_type_kind = self._fact_type_to_deadline_kind(item.fact_type)
        if fact_type_kind != "other":
            return fact_type_kind

        if item.kind_confidence >= 0.60 and item.deadline_kind != "other":
            return item.deadline_kind

        if question_deadline_kind and question_deadline_kind != "other":
            return question_deadline_kind

        if item.deadline_kind and item.deadline_kind != "other":
            return item.deadline_kind

        return "other"

    # --------------------------------------------------------
    # Candidate conversion
    # --------------------------------------------------------

    def _build_item_from_candidate(
        self,
        candidate: Any,
    ) -> DeadlineAnswerItem | None:
        source_type = self._candidate_source_type(candidate)

        if source_type == "legal_fact":
            return self._build_item_from_legal_fact_candidate(candidate)

        if source_type == "table_row":
            return self._build_item_from_table_row_candidate(candidate)

        if source_type == "block":
            return self._build_item_from_block_candidate(candidate)

        return None

    def _build_item_from_legal_fact_candidate(
        self,
        candidate: Any,
    ) -> DeadlineAnswerItem | None:
        payload = self._candidate_payload(candidate)
        value_json = payload.get("value_json") or {}
        metadata_json = payload.get("metadata_json") or {}
        condition_json = payload.get("condition_json") or {}

        source_text = self._clean(
            value_json.get("source_text")
            or self._candidate_text(candidate)
            or ""
        )

        deadline_value = self._clean(
            value_json.get("deadline_value")
            or value_json.get("value")
            or metadata_json.get("deadline_value")
            or ""
        )

        if not deadline_value:
            deadline_value = self._extract_deadline_value_from_text(source_text)

        if not deadline_value:
            return None
        if not self._looks_like_deadline_value(deadline_value):
            return None

        fact_type = self._candidate_fact_type(candidate)

        scope_text = self._clean(
            metadata_json.get("deadline_scope_text")
            or condition_json.get("deadline_scope_text")
            or value_json.get("deadline_scope_text")
            or self._scope_text_from_fact_type(fact_type)
            or ""
        )

        is_service_core_deadline = bool(
            metadata_json.get("is_service_core_deadline")
            or condition_json.get("is_service_core_deadline")
        )

        deadline_kind = self._fact_type_to_deadline_kind(fact_type)
        kind_confidence = 1.0 if deadline_kind != "other" else 0.0

        if deadline_kind == "other":
            deadline_kind = self._question_deadline_kind(
                " ".join(
                    x for x in [
                        deadline_value,
                        scope_text,
                        source_text,
                        fact_type or "",
                    ]
                    if x
                )
            ) or "other"
            if deadline_kind != "other":
                kind_confidence = 0.70

        source_id = str(self._candidate_attr(candidate, "source_id", "") or "")
        source_score = self._candidate_score(candidate)

        return DeadlineAnswerItem(
            deadline_value=deadline_value,
            scope_text=scope_text,
            source_type="legal_fact",
            citation_json=self._candidate_citation_json(candidate),
            fact_type=fact_type,
            deadline_kind=deadline_kind,
            kind_confidence=kind_confidence,
            is_service_core_deadline=is_service_core_deadline,
            candidate_score=source_score,
            table_title=None,
            table_number=None,
            source_fact_ids=[source_id] if source_id else [],
            source_scores=[source_score],
        )
        
    def _build_item_from_table_row_candidate(
        self,
        candidate: Any,
    ) -> DeadlineAnswerItem | None:
        metadata_json = self._candidate_metadata_json(candidate)
        cells = (
            metadata_json.get("cells_by_semantic_key")
            or metadata_json.get("cells_by_header_key")
            or {}
        )
        if not isinstance(cells, dict):
            cells = {}

        deadline_value = self._extract_deadline_value(cells)
        if not deadline_value:
            deadline_value = self._extract_deadline_value_from_text(
                self._candidate_text(candidate)
            )

        deadline_value = self._clean(deadline_value)
        if not deadline_value:
            return None
        if self._is_service_value(deadline_value):
            return None
        if not self._looks_like_deadline_value(deadline_value):
            return None

        scope_text = self._extract_scope_text(cells)
        if not scope_text:
            scope_text = self._candidate_text(candidate)
        scope_text = self._clean(scope_text) or ""

        deadline_kind, kind_confidence = self._classify_deadline_kind(
            text=" ".join([deadline_value, scope_text]),
        )

        source_id = str(self._candidate_attr(candidate, "source_id", "") or "")
        source_score = self._candidate_score(candidate)

        return DeadlineAnswerItem(
            deadline_value=deadline_value,
            scope_text=scope_text,
            source_type="table_row",
            citation_json=self._candidate_citation_json(candidate),
            fact_type=None,
            deadline_kind=deadline_kind,
            kind_confidence=kind_confidence,
            is_service_core_deadline=self._table_row_is_service_core(scope_text),
            candidate_score=source_score,
            table_title=self._clean(metadata_json.get("table_title")),
            table_number=self._clean(metadata_json.get("table_number")),
            source_row_ids=[source_id] if source_id else [],
            source_scores=[source_score],
        )

    def _build_item_from_block_candidate(
        self,
        candidate: Any,
    ) -> DeadlineAnswerItem | None:
        text = self._clean(self._candidate_text(candidate))
        if not text:
            return None

        if self._is_offtopic_deadline_block(text):
            return None

        deadline_value = self._extract_deadline_value_from_text(text)
        if not deadline_value:
            return None

        deadline_kind, kind_confidence = self._classify_deadline_kind(text=text)
        scope_text = self._extract_block_scope_text(text=text, deadline_kind=deadline_kind)

        source_id = str(self._candidate_attr(candidate, "source_id", "") or "")
        source_score = self._candidate_score(candidate)

        return DeadlineAnswerItem(
            deadline_value=deadline_value,
            scope_text=scope_text or text,
            source_type="block",
            citation_json=self._candidate_citation_json(candidate),
            fact_type=None,
            deadline_kind=deadline_kind,
            kind_confidence=kind_confidence,
            is_service_core_deadline=self._block_is_service_core(text, deadline_kind),
            candidate_score=source_score,
            source_block_ids=[source_id] if source_id else [],
            source_scores=[source_score],
        )

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    def _rank_item(
        self,
        *,
        item: DeadlineAnswerItem,
        question_text: str,
    ) -> float:
        question_kind = self._question_deadline_kind(question_text) or "other"
        score = 0.0

        if item.source_type == "legal_fact":
            score += 50.0
        elif item.source_type == "table_row":
            score += 25.0
        elif item.source_type == "block":
            score += 15.0

        if item.is_service_core_deadline:
            score += 65.0

        score += self._fact_type_bonus(
            deadline_kind=item.deadline_kind,
            question_kind=question_kind,
        )

        score += self._question_scope_bonus(
            question_text=question_text,
            item=item,
        )

        score += self._deadline_specificity_score(item.deadline_value)

        score += min(max(item.candidate_score, 0.0), 1.0) * 10.0

        if question_kind == "notification":
            score += self._notification_specificity_bonus(item)
        elif question_kind == "decision":
            score += self._decision_specificity_bonus(item)
        elif question_kind == "payment":
            score += self._payment_specificity_bonus(item)
        elif question_kind == "registration":
            score += self._registration_specificity_bonus(item)
        elif question_kind == "correction":
            score += self._correction_specificity_bonus(item)

        return score
        
    def _select_primary_item(
        self,
        *,
        ranked_items: list[DeadlineAnswerItem],
        question_deadline_kind: str,
    ) -> DeadlineAnswerItem:
        if not ranked_items:
            raise ValueError("ranked_items must not be empty")

        if question_deadline_kind == "other":
            return ranked_items[0]

        same_kind_items = [
            item
            for item in ranked_items
            if self._render_deadline_kind(
                item=item,
                question_deadline_kind=question_deadline_kind,
            ) == question_deadline_kind
        ]
        if not same_kind_items:
            return ranked_items[0]

        return max(
            same_kind_items,
            key=lambda item: self._primary_selection_score(
                item=item,
                question_deadline_kind=question_deadline_kind,
            ),
        )

    def _primary_selection_score(
        self,
        *,
        item: DeadlineAnswerItem,
        question_deadline_kind: str,
    ) -> float:
        text = self._normalize(
            " ".join(
                x for x in [
                    item.deadline_value,
                    item.scope_text,
                    item.fact_type or "",
                ]
                if x
            )
        )

        score = 0.0

        if item.source_type == "legal_fact":
            score += 40.0
        elif item.source_type == "table_row":
            score += 20.0
        elif item.source_type == "block":
            score += 10.0

        if item.is_service_core_deadline:
            score += 45.0

        score += min(max(item.kind_confidence, 0.0), 1.0) * 25.0
        score += min(max(item.candidate_score, 0.0), 1.0) * 10.0

        if self._fact_type_to_deadline_kind(item.fact_type) == question_deadline_kind:
            score += 30.0

        if self._is_suspension_related(text):
            score -= 120.0

        if question_deadline_kind == "notification":
            if self._has_exact_notification_decision_marker(text):
                score += 35.0
            if "уведом" in text:
                score += 12.0
            if "принятия решения" in text and "уведом" not in text:
                score -= 20.0

        elif question_deadline_kind == "decision":
            if "максимальный срок предоставления государственной услуги" in text:
                score += 40.0
            elif "срок предоставления государственной услуги" in text:
                score += 30.0

            if "решение о предоставлении" in text or "решение о назначении" in text:
                score += 18.0

            if "уведом" in text:
                score -= 20.0

        elif question_deadline_kind == "payment":
            if "не позднее 26-го числа" in text or "не позднее 26 числа" in text:
                score += 45.0
            if "выплат" in text or "перечисл" in text or "зачисл" in text:
                score += 25.0
            if "уведом" in text:
                score -= 40.0
            if (
                "принятия решения" in text
                and not (
                    "выплат" in text
                    or "перечисл" in text
                    or "зачисл" in text
                    or "26" in text
                )
            ):
                score -= 35.0

        elif question_deadline_kind == "registration":
            if "регистрац" in text or "регистрир" in text:
                score += 25.0

        elif question_deadline_kind == "correction":
            if "опечат" in text or "ошиб" in text:
                score += 25.0

        return score

    def _is_suspension_related(
        self,
        text: str,
    ) -> bool:
        norm = self._normalize(text)
        return (
            "приостанов" in norm
            or "приостанавлива" in norm
            or "приостановлении рассмотрения" in norm
        )

    def _fact_type_bonus(
        self,
        *,
        deadline_kind: str,
        question_kind: str,
    ) -> float:
        if question_kind == "other":
            weights = {
                "decision": 30.0,
                "notification": 25.0,
                "payment": 25.0,
                "registration": 15.0,
                "correction": -10.0,
                "other": 0.0,
            }
            return weights.get(deadline_kind, 0.0)

        if deadline_kind == question_kind:
            return 90.0

        if question_kind == "notification":
            cross = {
                "decision": 10.0,
                "payment": -20.0,
                "registration": -10.0,
                "correction": -80.0,
                "other": -15.0,
            }
            return cross.get(deadline_kind, 0.0)

        if question_kind == "decision":
            cross = {
                "notification": 5.0,
                "payment": -25.0,
                "registration": -5.0,
                "correction": -60.0,
                "other": -10.0,
            }
            return cross.get(deadline_kind, 0.0)

        if question_kind == "payment":
            cross = {
                "notification": -20.0,
                "decision": -20.0,
                "registration": -10.0,
                "correction": -60.0,
                "other": -10.0,
            }
            return cross.get(deadline_kind, 0.0)

        if question_kind == "registration":
            cross = {
                "notification": -10.0,
                "decision": -5.0,
                "payment": -15.0,
                "correction": -40.0,
                "other": -5.0,
            }
            return cross.get(deadline_kind, 0.0)

        if question_kind == "correction":
            cross = {
                "notification": -30.0,
                "decision": -30.0,
                "payment": -30.0,
                "registration": -15.0,
                "other": -10.0,
            }
            return cross.get(deadline_kind, 0.0)

        return 0.0

    def _question_scope_bonus(
        self,
        *,
        question_text: str,
        item: DeadlineAnswerItem,
    ) -> float:
        question_norm = self._normalize(question_text)
        scope_norm = self._normalize(item.scope_text)
        if not question_norm or not scope_norm:
            return 0.0

        bonus = 0.0
        for token in question_norm.split():
            if len(token) < 4:
                continue
            if token in scope_norm:
                bonus += 0.04

        return min(bonus, 0.20)

    def _notification_specificity_bonus(
        self,
        item: DeadlineAnswerItem,
    ) -> float:
        text = self._normalize(" ".join([item.deadline_value, item.scope_text, item.fact_type or ""]))
        score = 0.0

        if self._has_exact_notification_decision_marker(text):
            score += 40.0

        if "уведом" in text:
            score += 12.0

        if "о принятом решении" in text or "о решении" in text:
            score += 15.0

        if "об отсутствии ошибок" in text:
            score -= 70.0

        if "опечат" in text or "ошиб" in text:
            score -= 55.0

        if "проверки подписи" in text or "электронной подписи" in text:
            score -= 60.0

        return score

    def _decision_specificity_bonus(
        self,
        item: DeadlineAnswerItem,
    ) -> float:
        text = self._normalize(" ".join([item.deadline_value, item.scope_text, item.fact_type or ""]))
        score = 0.0

        if "решение о предоставлении" in text or "решение о назначении" in text:
            score += 22.0

        if "рассмотрения заявления" in text or "принятия решения" in text:
            score += 14.0

        if "уведом" in text and "о принятом решении" not in text:
            score -= 18.0

        if "об отсутствии ошибок" in text:
            score -= 60.0

        if self._is_suspension_related(text):
            score -= 90.0

        return score

    def _payment_specificity_bonus(
        self,
        item: DeadlineAnswerItem,
    ) -> float:
        text = self._normalize(" ".join([item.deadline_value, item.scope_text, item.fact_type or ""]))
        score = 0.0

        if "26-го числа" in text or "26 числа" in text:
            score += 18.0
        if "выплат" in text or "перечисл" in text or "зачисл" in text:
            score += 14.0
        if "уведом" in text:
            score -= 18.0
        return score

    def _registration_specificity_bonus(
        self,
        item: DeadlineAnswerItem,
    ) -> float:
        text = self._normalize(" ".join([item.deadline_value, item.scope_text, item.fact_type or ""]))
        if "регистрац" in text or "регистрир" in text:
            return 18.0
        return 0.0

    def _correction_specificity_bonus(
        self,
        item: DeadlineAnswerItem,
    ) -> float:
        text = self._normalize(" ".join([item.deadline_value, item.scope_text, item.fact_type or ""]))
        if "опечат" in text or "ошиб" in text:
            return 20.0
        return 0.0

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    def _merge_similar_items(
        self,
        items: list[DeadlineAnswerItem],
    ) -> tuple[list[DeadlineAnswerItem], list[dict[str, Any]]]:
        merged: dict[tuple[str, str, str], DeadlineAnswerItem] = {}
        debug: list[dict[str, Any]] = []

        for item in items:
            merge_key = (
                self._normalize(item.deadline_value),
                self._normalize(item.scope_text),
                self._normalize(item.deadline_kind),
            )

            existing = merged.get(merge_key)
            if existing is None:
                merged[merge_key] = DeadlineAnswerItem(
                    deadline_value=item.deadline_value,
                    scope_text=item.scope_text,
                    source_type=item.source_type,
                    citation_json=item.citation_json or {},
                    fact_type=item.fact_type,
                    deadline_kind=item.deadline_kind,
                    kind_confidence=item.kind_confidence,
                    is_service_core_deadline=item.is_service_core_deadline,
                    candidate_score=item.candidate_score,
                    table_title=item.table_title,
                    table_number=item.table_number,
                    source_row_ids=list(item.source_row_ids),
                    source_block_ids=list(item.source_block_ids),
                    source_fact_ids=list(item.source_fact_ids),
                    source_scores=list(item.source_scores),
                )
                debug.append(
                    {
                        "action": "create",
                        "merge_key": merge_key,
                        "deadline_value": item.deadline_value,
                        "scope_text": item.scope_text,
                        "deadline_kind": item.deadline_kind,
                        "source_type": item.source_type,
                    }
                )
                continue

            existing.candidate_score = max(existing.candidate_score, item.candidate_score)
            existing.kind_confidence = max(existing.kind_confidence, item.kind_confidence)
            existing.is_service_core_deadline = (
                existing.is_service_core_deadline or item.is_service_core_deadline
            )

            if existing.source_type != "legal_fact" and item.source_type == "legal_fact":
                existing.source_type = "legal_fact"

            if not existing.fact_type and item.fact_type:
                existing.fact_type = item.fact_type

            if item.scope_text and len(item.scope_text) > len(existing.scope_text or ""):
                existing.scope_text = item.scope_text

            if not existing.citation_json and item.citation_json:
                existing.citation_json = item.citation_json

            if not existing.table_title and item.table_title:
                existing.table_title = item.table_title
            if not existing.table_number and item.table_number:
                existing.table_number = item.table_number

            existing.source_row_ids.extend(x for x in item.source_row_ids if x not in existing.source_row_ids)
            existing.source_block_ids.extend(x for x in item.source_block_ids if x not in existing.source_block_ids)
            existing.source_fact_ids.extend(x for x in item.source_fact_ids if x not in existing.source_fact_ids)
            existing.source_scores.extend(item.source_scores)

            debug.append(
                {
                    "action": "merge",
                    "merge_key": merge_key,
                    "deadline_value": existing.deadline_value,
                    "scope_text": existing.scope_text,
                    "deadline_kind": existing.deadline_kind,
                    "source_type": existing.source_type,
                }
            )

        return list(merged.values()), debug

    # --------------------------------------------------------
    # Extraction helpers
    # --------------------------------------------------------

    def _extract_deadline_value(
        self,
        cells: dict[str, Any],
    ) -> Optional[str]:
        direct_value = self._clean(cells.get("deadline_value"))
        if direct_value:
            return direct_value

        for key, value in cells.items():
            key_norm = self._normalize(key)
            if "срок" in key_norm or "рабочих дней" in key_norm or "календарных дней" in key_norm:
                cleaned = self._clean(value)
                if cleaned:
                    return cleaned
        return None

    def _extract_scope_text(
        self,
        cells: dict[str, Any],
    ) -> Optional[str]:
        parts: list[str] = []

        for key, value in cells.items():
            if key in self._IGNORED_SCOPE_KEYS:
                continue

            cleaned_value = self._clean(value)
            if not cleaned_value:
                continue
            if self._is_service_value(cleaned_value):
                continue
            if self._looks_like_deadline_value(cleaned_value):
                continue

            pretty_key = self._pretty_label(key)
            if pretty_key:
                parts.append(f"{pretty_key}: {cleaned_value}")
            else:
                parts.append(cleaned_value)

        if not parts:
            return None

        if len(parts) == 1:
            return parts[0]

        return "; ".join(parts[:3])

    def _extract_deadline_value_from_text(
        self,
        text: str,
    ) -> Optional[str]:
        for pattern in self._BLOCK_DEADLINE_PATTERNS:
            match = pattern.search(text or "")
            if match:
                return self._clean(match.group(0))
        return None

    def _extract_block_scope_text(
        self,
        *,
        text: str,
        deadline_kind: str,
    ) -> Optional[str]:
        norm = self._normalize(text)

        if deadline_kind == "notification":
            if "о принятом решении" in norm:
                return "уведомление о принятом решении"
            if "уведомление" in norm:
                return "уведомление о решении"
            return "уведомление"

        if deadline_kind == "decision":
            if "решение о предоставлении" in norm:
                return "принятие решения о предоставлении"
            if "решение о назначении" in norm:
                return "принятие решения о назначении"
            return "принятие решения"

        if deadline_kind == "payment":
            if "едв" in norm:
                return "выплата ЕДВ"
            return "выплата"

        if deadline_kind == "registration":
            return "регистрация заявления"

        if deadline_kind == "correction":
            return "исправление ошибок и опечаток"

        return None

    # --------------------------------------------------------
    # Classification helpers
    # --------------------------------------------------------

    def _question_deadline_kind(
        self,
        question_text: str,
    ) -> str | None:
        text = self._normalize(question_text)

        if any(x in text for x in ("выплат", "перечисл", "деньги", "26-го числа", "26 числа")):
            return "payment"

        if any(x in text for x in ("уведом", "сообщ", "извест", "о решении")):
            return "notification"

        if any(x in text for x in ("зарегистр", "регистрац")):
            return "registration"

        if any(x in text for x in ("опечат", "ошиб")):
            return "correction"

        if any(x in text for x in ("примут решение", "принятия решения", "когда примут", "срок предоставления")):
            return "decision"

        return None

    def _classify_deadline_kind(
        self,
        *,
        text: Optional[str],
    ) -> tuple[str, float]:
        norm = self._normalize(text)
        if not norm:
            return ("other", 0.0)

        scores = {
            "decision": 0.0,
            "notification": 0.0,
            "payment": 0.0,
            "registration": 0.0,
            "correction": 0.0,
        }

        scores["notification"] += self._score_by_markers(norm, self._NOTIFICATION_STRONG_MARKERS, 2.3)
        scores["notification"] += self._score_by_markers(norm, self._NOTIFICATION_WEAK_MARKERS, 0.7)

        scores["payment"] += self._score_by_markers(norm, self._PAYMENT_STRONG_MARKERS, 2.2)
        scores["decision"] += self._score_by_markers(norm, self._DECISION_STRONG_MARKERS, 1.9)
        scores["registration"] += self._score_by_markers(norm, self._REGISTRATION_MARKERS, 2.0)
        scores["correction"] += self._score_by_markers(norm, self._CORRECTION_MARKERS, 2.4)

        if scores["notification"] > 0 and "принятия решения" in norm:
            scores["decision"] *= 0.55

        if scores["payment"] > 0 and "принятия решения" in norm:
            scores["decision"] *= 0.45

        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        winner, winner_score = ordered[0]
        if winner_score <= 0:
            return ("other", 0.0)

        total = sum(scores.values()) or 1.0
        confidence = round(winner_score / total, 3)

        if len(ordered) >= 2 and ordered[0][1] - ordered[1][1] < 0.35:
            confidence = round(min(confidence, 0.58), 3)

        return (winner, confidence)
        
    def _fact_type_to_deadline_kind(
        self,
        fact_type: str | None,
    ) -> str:
        mapping = {
            "decision_deadline": "decision",
            "notification_deadline": "notification",
            "payment_deadline": "payment",
            "registration_deadline": "registration",
            "correction_deadline": "correction",
        }
        return mapping.get(self._normalize(fact_type), "other")
        
    def _render_deadline_kind(
        self,
        *,
        item: DeadlineAnswerItem,
        question_deadline_kind: str,
    ) -> str:
        return self._resolve_final_deadline_kind(
            item=item,
            question_deadline_kind=question_deadline_kind,
        )

    # --------------------------------------------------------
    # Utility helpers
    # --------------------------------------------------------

    def _candidate_attr(
        self,
        candidate: Any,
        name: str,
        default: Any = None,
    ) -> Any:
        if isinstance(candidate, dict):
            return candidate.get(name, default)
        return getattr(candidate, name, default)

    def _candidate_source_type(
        self,
        candidate: Any,
    ) -> str | None:
        return (
            self._candidate_attr(candidate, "source_type")
            or self._candidate_attr(candidate, "evidence_item_type")
            or self._candidate_attr(candidate, "item_type")
        )

    def _candidate_payload(
        self,
        candidate: Any,
    ) -> dict[str, Any]:
        payload = (
            self._candidate_attr(candidate, "payload_json")
            or self._candidate_attr(candidate, "payload")
            or {}
        )
        return payload if isinstance(payload, dict) else {}

    def _candidate_metadata_json(
        self,
        candidate: Any,
    ) -> dict[str, Any]:
        payload = self._candidate_payload(candidate)
        metadata = (
            self._candidate_attr(candidate, "metadata_json")
            or payload.get("metadata_json")
            or {}
        )
        return metadata if isinstance(metadata, dict) else {}

    def _candidate_citation_json(
        self,
        candidate: Any,
    ) -> dict[str, Any]:
        payload = self._candidate_payload(candidate)
        citation = (
            self._candidate_attr(candidate, "citation_json")
            or payload.get("citation_json")
            or {}
        )
        return citation if isinstance(citation, dict) else {}
        
    def _candidate_fact_type(
        self,
        candidate: Any,
    ) -> str | None:
        payload = self._candidate_payload(candidate)
        value_json = payload.get("value_json") or {}
        metadata_json = payload.get("metadata_json") or {}
        condition_json = payload.get("condition_json") or {}

        fact_type = (
            self._candidate_attr(candidate, "fact_type")
            or payload.get("fact_type")
            or value_json.get("fact_type")
            or metadata_json.get("fact_type")
            or condition_json.get("fact_type")
            or self._candidate_attr(candidate, "title")
        )
        fact_type = self._clean(fact_type)
        return fact_type or None
        
    def _extract_deadline_value_from_text(
        self,
        text: str,
    ) -> str:
        source = self._clean(text)
        if not source:
            return ""

        patterns = [
            r"(в течение\s+\d+\s+(?:рабоч(?:их|его)|календарн(?:ых|ого))\s+дн(?:ей|я))",
            r"(не позднее\s+\d+(?:-го)?\s+числа[^.]*)",
            r"(не более\s+\d+\s+(?:рабоч(?:их|его)|календарн(?:ых|ого))\s+дн(?:ей|я))",
            r"(до\s+\d+\s+(?:рабоч(?:их|его)|календарн(?:ых|ого))\s+дн(?:ей|я))",
            r"(составляет\s+\d+\s+(?:рабоч(?:их|его)|календарн(?:ых|ого))\s+дн(?:ей|я))",
        ]

        for pattern in patterns:
            match = re.search(pattern, source, flags=re.IGNORECASE)
            if match:
                return self._clean(match.group(1))

        return ""
        
    def _scope_text_from_fact_type(
        self,
        fact_type: str | None,
    ) -> str:
        mapping = {
            "decision_deadline": "принятие решения о предоставлении",
            "notification_deadline": "уведомление о решении",
            "payment_deadline": "выплата",
            "registration_deadline": "регистрация заявления",
            "correction_deadline": "исправление ошибок",
        }
        return mapping.get(self._normalize(fact_type), "")

    def _candidate_text(
        self,
        candidate: Any,
    ) -> str:
        payload = self._candidate_payload(candidate)
        metadata = self._candidate_metadata_json(candidate)

        parts: list[str] = [
            self._clean(self._candidate_attr(candidate, "title", "")) or "",
            self._clean(self._candidate_attr(candidate, "snippet", "")) or "",
            self._clean(payload.get("validity_note")) or "",
            self._clean(metadata.get("row_summary")) or "",
            self._clean(metadata.get("row_text")) or "",
        ]

        return self._clean(" ".join(x for x in parts if x)) or ""

    def _candidate_score(
        self,
        candidate: Any,
    ) -> float:
        raw = (
            self._candidate_attr(candidate, "rerank_score")
            or self._candidate_attr(candidate, "effective_score")
            or self._candidate_attr(candidate, "score")
            or self._candidate_attr(candidate, "source_score")
            or 0.0
        )
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0

    def _deadline_specificity_score(
        self,
        value: str,
    ) -> int:
        text = self._normalize(value)
        score = 0
        if "рабоч" in text:
            score += 3
        if "календар" in text:
            score += 2
        if any(ch.isdigit() for ch in value):
            score += 2
        if "не более" in text or "не позднее" in text:
            score += 1
        if "26" in text:
            score += 1
        return score

    def _looks_like_deadline_value(
        self,
        text: str,
    ) -> bool:
        text_norm = self._normalize(text)
        if not text_norm:
            return False

        deadline_markers = [
            "дней",
            "дня",
            "рабочих",
            "календарных",
            "не позднее",
            "не более",
            "числа",
        ]
        return any(marker in text_norm for marker in deadline_markers) and any(ch.isdigit() for ch in text)

    def _is_service_value(
        self,
        text: str,
    ) -> bool:
        return self._normalize(text) in {self._normalize(x) for x in self._SERVICE_VALUES}

    def _is_offtopic_deadline_block(
        self,
        text: str,
    ) -> bool:
        norm = self._normalize(text)
        return any(marker in norm for marker in self._OFFTOPIC_MARKERS)

    def _has_exact_notification_decision_marker(
        self,
        text: str,
    ) -> bool:
        norm = self._normalize(text)
        return (
            "уведом" in norm
            and (
                "о принятом решении" in norm
                or "о решении" in norm
                or "направляется заявителю" in norm
                or "направляет заявителю" in norm
                or "заявитель уведомляется" in norm
            )
        )

    def _block_is_service_core(
        self,
        text: str,
        deadline_kind: str,
    ) -> bool:
        norm = self._normalize(text)

        if deadline_kind == "notification" and self._has_exact_notification_decision_marker(norm):
            return True

        if deadline_kind == "decision" and (
            "решение о предоставлении" in norm
            or "решение о назначении" in norm
            or "принятия решения" in norm
        ):
            return True

        if deadline_kind == "payment" and (
            "не позднее 26-го числа" in norm
            or "не позднее 26 числа" in norm
            or "выплат" in norm
        ):
            return True

        if deadline_kind == "registration" and "регистрац" in norm:
            return True

        if deadline_kind == "correction" and ("опечат" in norm or "ошиб" in norm):
            return True

        return False

    def _table_row_is_service_core(
        self,
        scope_text: str,
    ) -> bool:
        text = self._normalize(scope_text)

        if self._has_exact_notification_decision_marker(text):
            return True

        if any(
            marker in text
            for marker in (
                "предоставления государственной услуги",
                "принятия решения",
                "решение о предоставлении",
                "решение о назначении",
                "регистрация заявления",
                "регистрация запроса",
                "выплата",
                "26-го числа",
                "26 числа",
            )
        ):
            return True

        if any(
            marker in text
            for marker in (
                "межведомствен",
                "опросн",
                "обратн",
                "доработк",
                "представить лично",
                "опечаток и ошибок",
                "об отсутствии ошибок",
            )
        ):
            return False

        return False

    def _score_by_markers(
        self,
        norm: str,
        markers: tuple[str, ...],
        weight: float,
    ) -> float:
        score = 0.0
        for marker in markers:
            if marker in norm:
                score += weight
        return score

    def _pretty_label(
        self,
        key: Any,
    ) -> Optional[str]:
        if key is None:
            return None
        text = str(key).strip().replace("_", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text or None

    def _clean(
        self,
        value: Any,
    ) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        text = re.sub(r"\s+", " ", text).strip()
        return text or None

    def _normalize(
        self,
        value: Any,
    ) -> str:
        cleaned = self._clean(value)
        if not cleaned:
            return ""
        return cleaned.lower().replace("ё", "е")