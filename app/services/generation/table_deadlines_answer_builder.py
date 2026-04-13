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
                question_kind=question_deadline_kind,
                question_text=normalized_question,
            ),
            reverse=True,
        )

        primary_item = ranked_items[0]
        alternative_items = ranked_items[1:]

        ambiguity_reason: str | None = None
        if alternative_items and any(
            self._normalize(item.deadline_value) != self._normalize(primary_item.deadline_value)
            for item in alternative_items[:3]
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
                                question_kind=question_deadline_kind,
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
        render_kind = self._render_deadline_kind(
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

        if not deadline_value or not self._looks_like_deadline_value(deadline_value):
            return None

        fact_type = self._candidate_fact_type(candidate)

        scope_text = self._clean(
            metadata_json.get("deadline_scope_text")
            or condition_json.get("deadline_scope_text")
            or value_json.get("deadline_scope_text")
            or self._scope_text_from_fact_type(fact_type)
            or ""
        ) or ""

        deadline_kind = self._fact_type_to_deadline_kind(fact_type)
        kind_confidence = 1.0 if deadline_kind != "other" else 0.0

        if deadline_kind == "other":
            deadline_kind, kind_confidence = self._classify_deadline_kind(
                text=" ".join(
                    x for x in [
                        deadline_value,
                        scope_text,
                        source_text,
                        fact_type or "",
                    ]
                    if x
                ),
            )

        source_id = str(self._candidate_attr(candidate, "source_id", "") or "")
        source_score = self._candidate_score(candidate)

        is_service_core_deadline = bool(
            metadata_json.get("is_service_core_deadline")
            or condition_json.get("is_service_core_deadline")
            or self._block_is_service_core(
                text=" ".join(x for x in [scope_text, source_text] if x),
                deadline_kind=deadline_kind,
            )
        )

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
        payload = self._candidate_payload(candidate)
        metadata_json = payload.get("metadata_json") or {}
        cells = (
            payload.get("cells_json")
            or payload.get("cells")
            or metadata_json.get("cells_json")
            or {}
        )
        if not isinstance(cells, dict):
            cells = {}

        deadline_value = self._extract_deadline_value(cells)
        if not deadline_value:
            deadline_value = self._extract_deadline_value_from_text(self._candidate_text(candidate) or "")

        if not deadline_value or not self._looks_like_deadline_value(deadline_value):
            return None

        scope_text = self._clean(
            metadata_json.get("deadline_scope_text")
            or self._extract_scope_text(cells)
            or payload.get("row_summary")
            or self._candidate_text(candidate)
            or ""
        ) or ""

        fact_type = self._candidate_fact_type(candidate)
        deadline_kind = self._fact_type_to_deadline_kind(fact_type)
        kind_confidence = 1.0 if deadline_kind != "other" else 0.0

        if deadline_kind == "other":
            deadline_kind, kind_confidence = self._classify_deadline_kind(
                text=" ".join(x for x in [deadline_value, scope_text] if x),
            )

        row_id = str(self._candidate_attr(candidate, "source_id", "") or "")
        source_score = self._candidate_score(candidate)

        return DeadlineAnswerItem(
            deadline_value=deadline_value,
            scope_text=scope_text,
            source_type="table_row",
            citation_json=self._candidate_citation_json(candidate),
            fact_type=fact_type,
            deadline_kind=deadline_kind,
            kind_confidence=kind_confidence,
            is_service_core_deadline=bool(
                metadata_json.get("is_service_core_deadline")
                or self._table_row_is_service_core(scope_text)
            ),
            candidate_score=source_score,
            table_title=self._clean(
                self._candidate_attr(candidate, "title")
                or metadata_json.get("table_title")
            ),
            table_number=self._clean(metadata_json.get("table_number")),
            source_row_ids=[row_id] if row_id else [],
            source_scores=[source_score],
        )

    def _build_item_from_block_candidate(
        self,
        candidate: Any,
    ) -> DeadlineAnswerItem | None:
        text = self._candidate_text(candidate) or ""
        if not text:
            return None

        deadline_value = self._extract_deadline_value_from_text(text)
        if not deadline_value:
            return None
        if not self._looks_like_deadline_value(deadline_value):
            return None

        deadline_kind, kind_confidence = self._classify_deadline_kind(text=text)
        if deadline_kind == "other":
            return None

        if self._is_offtopic_deadline_block(text) and deadline_kind != "correction":
            return None

        block_id = str(self._candidate_attr(candidate, "source_id", "") or "")
        source_score = self._candidate_score(candidate)
        scope_text = self._extract_block_scope_text(
            text=text,
            deadline_kind=deadline_kind,
        ) or ""

        return DeadlineAnswerItem(
            deadline_value=deadline_value,
            scope_text=scope_text,
            source_type="block",
            citation_json=self._candidate_citation_json(candidate),
            fact_type=self._candidate_fact_type(candidate),
            deadline_kind=deadline_kind,
            kind_confidence=kind_confidence,
            is_service_core_deadline=self._block_is_service_core(
                text=text,
                deadline_kind=deadline_kind,
            ),
            candidate_score=source_score,
            table_title=None,
            table_number=None,
            source_block_ids=[block_id] if block_id else [],
            source_scores=[source_score],
        )

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    def _rank_item(
        self,
        *,
        item: DeadlineAnswerItem,
        question_kind: str,
        question_text: str,
    ) -> float:
        score = 0.0

        if item.source_type == "legal_fact":
            score += 50.0
        elif item.source_type == "table_row":
            score += 25.0
        elif item.source_type == "block":
            score += 15.0

        if item.is_service_core_deadline:
            score += 65.0

        score += self._kind_alignment_score(
            deadline_kind=self._render_deadline_kind(
                item=item,
                question_deadline_kind=question_kind,
            ),
            question_kind=question_kind,
        )

        score += self._question_scope_bonus(
            question_text=question_text,
            item=item,
        )

        score += self._deadline_specificity_score(item.deadline_value)
        score += min(max(item.candidate_score, 0.0), 1.0) * 10.0
        score += min(max(item.kind_confidence, 0.0), 1.0) * 15.0

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

        if self._is_suspension_related(text):
            score -= 120.0

        if question_kind == "notification":
            if self._has_exact_notification_decision_marker(text):
                score += 35.0
            if "уведом" in text:
                score += 12.0
            if "принятия решения" in text and "уведом" not in text:
                score -= 20.0

        elif question_kind == "decision":
            if "максимальный срок предоставления государственной услуги" in text:
                score += 40.0
            elif "срок предоставления государственной услуги" in text:
                score += 30.0

            if "решение о предоставлении" in text or "решение о назначении" in text:
                score += 18.0

            if "уведом" in text:
                score -= 20.0

        elif question_kind == "payment":
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

        elif question_kind == "registration":
            if "регистрац" in text or "регистрир" in text:
                score += 25.0

        elif question_kind == "correction":
            if "опечат" in text or "ошиб" in text:
                score += 25.0

        return score

    def _kind_alignment_score(
        self,
        *,
        deadline_kind: str,
        question_kind: str,
    ) -> float:
        if question_kind == "other":
            return 0.0

        if deadline_kind == question_kind:
            return 40.0

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
                "notification": -15.0,
                "decision": -20.0,
                "payment": -20.0,
                "registration": -20.0,
                "other": -5.0,
            }
            return cross.get(deadline_kind, 0.0)

        return 0.0

    def _question_scope_bonus(
        self,
        *,
        question_text: str,
        item: DeadlineAnswerItem,
    ) -> float:
        norm_question = self._normalize(question_text)
        norm_scope = self._normalize(item.scope_text)
        norm_fact_type = self._normalize(item.fact_type)

        score = 0.0

        if "уведом" in norm_question and "уведом" in norm_scope:
            score += 18.0
        if "решени" in norm_question and (
            "решени" in norm_scope or "decision" in norm_fact_type
        ):
            score += 12.0
        if (
            "выплат" in norm_question
            or "придет" in norm_question
            or "придёт" in norm_question
            or "поступ" in norm_question
            or "получу" in norm_question
        ) and (
            "выплат" in norm_scope
            or "payment" in norm_fact_type
            or "26" in norm_scope
        ):
            score += 18.0
        if "регистрац" in norm_question and "регистрац" in norm_scope:
            score += 18.0
        if (
            "ошиб" in norm_question
            or "опечат" in norm_question
        ) and (
            "ошиб" in norm_scope
            or "опечат" in norm_scope
            or "correction" in norm_fact_type
        ):
            score += 18.0

        return score

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    def _merge_similar_items(
        self,
        items: list[DeadlineAnswerItem],
    ) -> tuple[list[DeadlineAnswerItem], list[dict[str, Any]]]:
        merged: dict[tuple[str, str, str, str, str], DeadlineAnswerItem] = {}
        debug: list[dict[str, Any]] = []

        for item in items:
            merge_key = (
                self._normalize(item.deadline_value),
                self._normalize(item.scope_text),
                self._normalize(item.deadline_kind),
                self._normalize(item.source_type),
                self._normalize(item.fact_type),
            )

            if merge_key not in merged:
                merged[merge_key] = DeadlineAnswerItem(
                    deadline_value=item.deadline_value,
                    scope_text=item.scope_text,
                    source_type=item.source_type,
                    citation_json=dict(item.citation_json),
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
                        "action": "new",
                        "merge_key": merge_key,
                        "deadline_value": item.deadline_value,
                        "scope_text": item.scope_text,
                        "deadline_kind": item.deadline_kind,
                        "source_type": item.source_type,
                        "fact_type": item.fact_type,
                    }
                )
                continue

            existing = merged[merge_key]
            existing.candidate_score = max(existing.candidate_score, item.candidate_score)
            existing.kind_confidence = max(existing.kind_confidence, item.kind_confidence)
            existing.is_service_core_deadline = (
                existing.is_service_core_deadline or item.is_service_core_deadline
            )

            if not existing.citation_json and item.citation_json:
                existing.citation_json = dict(item.citation_json)

            if item.scope_text and len(item.scope_text) > len(existing.scope_text or ""):
                existing.scope_text = item.scope_text

            if not existing.table_title and item.table_title:
                existing.table_title = item.table_title
            if not existing.table_number and item.table_number:
                existing.table_number = item.table_number

            existing.source_row_ids.extend(
                x for x in item.source_row_ids if x not in existing.source_row_ids
            )
            existing.source_block_ids.extend(
                x for x in item.source_block_ids if x not in existing.source_block_ids
            )
            existing.source_fact_ids.extend(
                x for x in item.source_fact_ids if x not in existing.source_fact_ids
            )
            existing.source_scores.extend(item.source_scores)

            debug.append(
                {
                    "action": "merge",
                    "merge_key": merge_key,
                    "deadline_value": existing.deadline_value,
                    "scope_text": existing.scope_text,
                    "deadline_kind": existing.deadline_kind,
                    "source_type": existing.source_type,
                    "fact_type": existing.fact_type,
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
        source = self._clean(text)
        if not source:
            return None

        count_token = (
            r"(?:"
            r"\d+|"
            r"один|одна|одно|одного|одной|одни|одних|"
            r"два|две|двух|"
            r"три|трех|трёх|"
            r"четыре|четырех|четырёх|"
            r"пять|пяти|"
            r"шесть|шести|"
            r"семь|семи|"
            r"восемь|восьми|"
            r"девять|девяти|"
            r"десять|десяти"
            r")"
        )

        patterns = [
            re.compile(
                rf"в течение\s+{count_token}\s+(?:рабоч(?:их|его)|календарн(?:ых|ого))\s+дн(?:я|ей)",
                re.IGNORECASE,
            ),
            re.compile(
                rf"не более\s+{count_token}\s+(?:рабоч(?:их|его)|календарн(?:ых|ого))\s+дн(?:я|ей)",
                re.IGNORECASE,
            ),
            re.compile(
                rf"не позднее\s+{count_token}\s+(?:рабоч(?:их|его)|календарн(?:ых|ого))\s+дн(?:я|ей)",
                re.IGNORECASE,
            ),
            re.compile(
                r"не позднее\s+26(?:-го)?\s+числа(?:\s+месяца)?",
                re.IGNORECASE,
            ),
            re.compile(
                r"не позднее\s+[^.;]{3,140}",
                re.IGNORECASE,
            ),
        ]

        for pattern in patterns:
            match = pattern.search(source)
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
            if "максимальный срок предоставления государственной услуги" in norm:
                return "предоставление государственной услуги"
            if "решение о предоставлении" in norm or "решение о назначении" in norm:
                return "принятие решения о предоставлении"
            return "принятие решения"

        if deadline_kind == "payment":
            return "выплата"

        if deadline_kind == "registration":
            return "регистрация заявления"

        if deadline_kind == "correction":
            return "исправление ошибок"

        return None

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

    # --------------------------------------------------------
    # Classification helpers
    # --------------------------------------------------------

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

    def _question_deadline_kind(
        self,
        text: str,
    ) -> str:
        norm = self._normalize(text)
        if not norm:
            return "other"

        payment_exact_markers = (
            "когда придет",
            "когда придёт",
            "когда придут",
            "когда поступит",
            "когда поступят",
            "когда выплатят",
            "когда перечислят",
            "когда зачислят",
            "когда мне придет",
            "когда мне придёт",
            "когда мне выплатят",
            "когда мне перечислят",
            "когда мне зачислят",
            "когда я получу",
            "когда получу",
            "когда придет едв",
            "когда придёт едв",
            "когда мне придет едв",
            "когда мне придёт едв",
            "когда будет выплата",
            "когда поступят деньги",
            "когда придут деньги",
        )
        if any(marker in norm for marker in payment_exact_markers):
            return "payment"

        if "уведом" in norm or "сообщ" in norm:
            return "notification"

        if "регистрац" in norm or "зарегистрир" in norm:
            return "registration"

        if "опечат" in norm or "ошиб" in norm or "исправлен" in norm:
            return "correction"

        decision_exact_markers = (
            "срок принятия решения",
            "когда примут решение",
            "когда будет решение",
            "срок рассмотрения",
            "максимальный срок предоставления",
            "срок предоставления государственной услуги",
        )
        if any(marker in norm for marker in decision_exact_markers):
            return "decision"

        if (
            "выплат" in norm
            or "перечисл" in norm
            or "зачисл" in norm
            or "поступ" in norm
            or "придет" in norm
            or "придёт" in norm
            or "получу" in norm
        ):
            return "payment"

        return "other"

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

        scores["notification"] += self._score_by_markers(
            norm,
            self._NOTIFICATION_STRONG_MARKERS,
            3.0,
        )
        scores["notification"] += self._score_by_markers(
            norm,
            self._NOTIFICATION_WEAK_MARKERS,
            1.0,
        )
        scores["payment"] += self._score_by_markers(
            norm,
            self._PAYMENT_STRONG_MARKERS,
            3.0,
        )
        scores["decision"] += self._score_by_markers(
            norm,
            self._DECISION_STRONG_MARKERS,
            2.5,
        )
        scores["registration"] += self._score_by_markers(
            norm,
            self._REGISTRATION_MARKERS,
            3.0,
        )
        scores["correction"] += self._score_by_markers(
            norm,
            self._CORRECTION_MARKERS,
            3.0,
        )

        if self._has_exact_notification_decision_marker(norm):
            scores["notification"] += 5.0

        if "не позднее 26" in norm:
            scores["payment"] += 5.0

        ordered = sorted(
            scores.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )
        best_kind, best_score = ordered[0]
        second_score = ordered[1][1] if len(ordered) > 1 else 0.0

        if best_score <= 0:
            return ("other", 0.0)

        confidence = min(1.0, max(0.0, (best_score - second_score + 1.0) / 5.0))
        return (best_kind, confidence)

    # --------------------------------------------------------
    # Candidate helpers
    # --------------------------------------------------------

    def _candidate_source_type(
        self,
        candidate: Any,
    ) -> str:
        return self._normalize(self._candidate_attr(candidate, "source_type")) or ""

    def _candidate_payload(
        self,
        candidate: Any,
    ) -> dict[str, Any]:
        payload = self._candidate_attr(candidate, "payload_json")
        return payload if isinstance(payload, dict) else {}

    def _candidate_attr(
        self,
        candidate: Any,
        name: str,
        default: Any = None,
    ) -> Any:
        if isinstance(candidate, dict):
            return candidate.get(name, default)
        return getattr(candidate, name, default)

    def _candidate_text(
        self,
        candidate: Any,
    ) -> str:
        payload = self._candidate_payload(candidate)
        parts = [
            self._clean(self._candidate_attr(candidate, "snippet")),
            self._clean(self._candidate_attr(candidate, "text")),
            self._clean(payload.get("text")),
            self._clean(payload.get("row_summary")),
            self._clean(payload.get("title")),
        ]
        return self._clean(" ".join(x for x in parts if x)) or ""

    def _candidate_citation_json(
        self,
        candidate: Any,
    ) -> dict[str, Any]:
        value = self._candidate_attr(candidate, "citation_json")
        if isinstance(value, dict):
            return dict(value)
        return {}

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

    # --------------------------------------------------------
    # Utility helpers
    # --------------------------------------------------------

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

        deadline_markers = (
            "дней",
            "дня",
            "рабочих",
            "рабочего",
            "календарных",
            "календарного",
            "не позднее",
            "не более",
            "числа",
        )
        number_words = (
            "один",
            "одна",
            "одно",
            "одного",
            "одной",
            "два",
            "две",
            "двух",
            "три",
            "трех",
            "трёх",
            "четыре",
            "четырех",
            "четырёх",
            "пять",
            "пяти",
            "шесть",
            "шести",
            "семь",
            "семи",
            "восемь",
            "восьми",
            "девять",
            "девяти",
            "десять",
            "десяти",
        )

        has_number = any(ch.isdigit() for ch in text) or any(word in text_norm for word in number_words)
        return has_number and any(marker in text_norm for marker in deadline_markers)

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
            or "срок предоставления государственной услуги" in norm
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