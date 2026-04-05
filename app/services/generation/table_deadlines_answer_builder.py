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
    is_service_core_deadline: bool = False
    candidate_score: float = 0.0

    table_title: str | None = None
    table_number: str | None = None


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
            "is_service_core_deadline": item.is_service_core_deadline,
            "candidate_score": item.candidate_score,
            "table_title": item.table_title,
            "table_number": item.table_number,
            "citation_json": item.citation_json,
        }


class TableDeadlinesAnswerBuilder:
    """
    Deterministic builder for deadline questions.

    Что меняем по сравнению с предыдущей версией:
    - block-кандидаты остаются полноправным источником срока;
    - классификация kind становится контекстной: payment / notification
      должны выигрывать у простого упоминания "принятия решения", если
      это лишь опорная точка для последующего этапа;
    - render_text по возможности отдаёт один основной срок, а не свалку
      всех найденных сроков разных этапов.
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
        "other": "срока",
    }

    # Общие deadline-паттерны. Порядок важен: сначала более длинные и
    # предметные формулы, затем более короткие.
    _BLOCK_DEADLINE_PATTERNS = [
        re.compile(
            r"не позднее\s+26(?:-го)?\s+числа\s+месяца[^.;]{0,160}",
            re.IGNORECASE,
        ),
        re.compile(
            r"в течение\s+\d+\s+(?:рабоч(?:их|его)|календарн(?:ых|ого))\s+дн(?:я|ей)",
            re.IGNORECASE,
        ),
        re.compile(r"в течение\s+\d+\s+дн(?:я|ей)", re.IGNORECASE),
        re.compile(
            r"не более\s+\d+\s+(?:рабоч(?:их|его)|календарн(?:ых|ого))\s+дн(?:я|ей)",
            re.IGNORECASE,
        ),
        re.compile(r"не позднее\s+[^.;]{3,160}", re.IGNORECASE),
    ]

    # Для жёсткого отсечения нерелевантных procedural block-ов, которые
    # попадали в shortlist, но не отвечали на вопрос о сроке ЕДВ как услуги.
    _OFFTOPIC_BLOCK_MARKERS = (
        "исправлении опечаток",
        "опечаток и ошибок",
        "исправлении ошибок",
        "отказе в приеме к рассмотрению документов",
        "новый документ взамен",
        "новый документ",
        "получения нового документа",
        "уведомление об отсутствии ошибок",
        "уведомления об отсутствии ошибок",
        "отсутствии ошибок",
        "отсутствия ошибок",
        "содержащего опечатки и ошибки",
        "проверки подписи",
        "подлинности простой электронной подписи",
        "усиленной квалифицированной электронной подписи",
        "статьи 9",
        "статьи 11",
        "федерального закона n 63-фз",
    )

    # Маркеры kind-а разбиты по силе. Это позволяет не считать любое
    # упоминание "принятия решения" признаком decision, если основной смысл
    # блока - уведомление или выплата после уже принятого решения.
    _PAYMENT_STRONG_MARKERS = (
        "выплачивает",
        "выплачивается",
        "выплата",
        "выплаты",
        "ежемесячно",
        "перечисление",
        "перечисления",
        "зачисление",
        "зачисления",
        "доставки выплаты",
        "осуществления выплаты",
        "через отделение почтовой связи",
        "российскую кредитную организацию",
        "не позднее 26-го числа",
        "не позднее 26 числа",
        "не позднее 26-го числа месяца",
        "не позднее 26 числа месяца",
        "26-го числа текущего месяца",
        "26 числа текущего месяца",
    )
    _PAYMENT_WEAK_MARKERS = (
        "предоставление едв",
        "возобновление едв",
        "получателю",
        "денежн",
    )

    _NOTIFICATION_STRONG_MARKERS = (
        "направляет",
        "направить",
        "уведомление",
        "уведомляет",
        "уведомить",
        "извещение",
        "извещает",
        "сообщение о решении",
        "сообщает заявителю",
        "информирование",
        "информирует",
    )
    _NOTIFICATION_WEAK_MARKERS = (
        "заявителю",
        "ветерану труда края уведомление",
        "направление уведомления",
    )

    _DECISION_STRONG_MARKERS = (
        "решение о предоставлении",
        "решение о назначении",
        "решение принимается",
        "принимается уполномоченным учреждением",
        "принятие решения",
        "принятия решения",
        "назначении едв",
        "назначение едв",
        "рассмотрения заявления",
        "рассмотрение заявления",
        "рассмотрения документов",
        "регистрации заявления",
    )
    _DECISION_WEAK_MARKERS = (
        "назначении",
        "назначение",
        "решение",
        "рассмотрение",
    )

    def build(
        self,
        *,
        candidates: list[Any],
        question_text: Optional[str] = None,
    ) -> DeadlinesAnswerBuildResult:
        raw_items: list[DeadlineAnswerItem] = []
        dropped_rows_debug: list[dict[str, Any]] = []

        normalized_question = self._normalize(question_text)
        question_deadline_kind = self._question_deadline_kind(normalized_question)

        for candidate in candidates:
            item = self._build_item_from_candidate(candidate)
            if item is None:
                continue
            raw_items.append(item)

        if not raw_items:
            return DeadlinesAnswerBuildResult(
                can_answer=False,
                question_deadline_kind=question_deadline_kind or "other",
                reason="no_deadline_items",
                dropped_rows_debug=dropped_rows_debug,
                merged_items_debug=[
                    {
                        "raw_items_count": 0,
                        "merged_items_count": 0,
                        "ranked_items_preview": [],
                    }
                ],
            )

        merged_items, merge_debug = self._merge_similar_items(raw_items)

        if not merged_items:
            return DeadlinesAnswerBuildResult(
                can_answer=False,
                question_deadline_kind=question_deadline_kind or "other",
                reason="no_merged_deadline_items",
                dropped_rows_debug=dropped_rows_debug,
                merged_items_debug=[
                    *merge_debug,
                    {
                        "raw_items_count": len(raw_items),
                        "merged_items_count": 0,
                        "ranked_items_preview": [],
                    },
                ],
            )

        ranked_items = sorted(
            merged_items,
            key=lambda item: self._rank_item(
                item=item,
                question_text=normalized_question,
            ),
            reverse=True,
        )

        primary_item = ranked_items[0]
        alternative_items = ranked_items[1:]

        ambiguity_reason: str | None = None
        if alternative_items and any(
            self._normalize(item.deadline_value) != self._normalize(primary_item.deadline_value)
            for item in alternative_items
        ):
            ambiguity_reason = "multiple_distinct_deadlines"

        return DeadlinesAnswerBuildResult(
            can_answer=True,
            question_deadline_kind=question_deadline_kind or "other",
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
        
    def _build_item_from_candidate(
        self,
        candidate: Any,
    ) -> DeadlineAnswerItem | None:
        source_type = self._candidate_source_type(candidate)

        if source_type == "legal_fact":
            return self._build_item_from_legal_fact_candidate(candidate)

        if source_type == "table_row":
            return self._build_item_from_table_row_candidate(candidate)

        return None
        
    def _build_item_from_table_row_candidate(
        self,
        candidate: Any,
    ) -> DeadlineAnswerItem | None:
        deadline_value = self._extract_deadline_value(candidate)
        if not deadline_value:
            return None

        if self._is_service_value(deadline_value):
            return None
        if not self._looks_like_deadline_value(deadline_value):
            return None

        scope_text = self._extract_scope_text(candidate)
        scope_text = self._clean(scope_text) if scope_text else ""

        citation_json = self._candidate_citation_json(candidate)
        metadata_json = self._candidate_metadata_json(candidate)

        return DeadlineAnswerItem(
            deadline_value=self._clean(deadline_value),
            scope_text=scope_text or "",
            source_type="table_row",
            citation_json=citation_json,
            fact_type=None,
            is_service_core_deadline=self._table_row_is_service_core(scope_text),
            candidate_score=self._candidate_score(candidate),
            table_title=metadata_json.get("table_title"),
            table_number=metadata_json.get("table_number"),
        )
        
    def _build_item_from_legal_fact_candidate(
        self,
        candidate: Any,
    ) -> DeadlineAnswerItem | None:
        payload = self._candidate_payload(candidate)
        value_json = payload.get("value_json") or {}
        metadata_json = payload.get("metadata_json") or {}
        condition_json = payload.get("condition_json") or {}

        deadline_value = (
            value_json.get("deadline_value")
            or value_json.get("value")
            or ""
        )
        deadline_value = self._clean(deadline_value)

        if not deadline_value:
            return None
        if not self._looks_like_deadline_value(deadline_value):
            return None

        scope_text = (
            metadata_json.get("deadline_scope_text")
            or condition_json.get("deadline_scope_text")
            or condition_json.get("heading_text")
            or value_json.get("source_text")
            or self._candidate_text(candidate)
            or ""
        )
        scope_text = self._clean(scope_text)

        fact_type = (
            self._candidate_attr(candidate, "fact_type")
            or payload.get("fact_type")
            or metadata_json.get("fact_type")
        )

        is_service_core_deadline = bool(
            metadata_json.get("is_service_core_deadline")
            or condition_json.get("is_service_core_deadline")
        )

        return DeadlineAnswerItem(
            deadline_value=deadline_value,
            scope_text=scope_text,
            source_type="legal_fact",
            citation_json=self._candidate_citation_json(candidate),
            fact_type=fact_type,
            is_service_core_deadline=is_service_core_deadline,
            candidate_score=self._candidate_score(candidate),
            table_title=None,
            table_number=None,
        )
        
    def _rank_item(
        self,
        *,
        item: DeadlineAnswerItem,
        question_text: str,
    ) -> float:
        score = 0.0

        # 1. legal_fact приоритетнее table_row
        if item.source_type == "legal_fact":
            score += 50.0
        elif item.source_type == "table_row":
            score += 20.0

        # 2. core-сроки приоритетнее procedural/fallback
        if item.is_service_core_deadline:
            score += 80.0

        # 3. совпадение типа вопроса и типа срока
        score += self._fact_type_bonus(
            fact_type=item.fact_type,
            question_text=question_text,
        )

        # 4. старый бонус за совпадение scope с вопросом сохраняем
        score += self._question_scope_bonus(
            question_text=question_text,
            scope_text=item.scope_text,
        )

        # 5. чем “сильнее” формулировка срока, тем лучше
        score += self._deadline_specificity_score(item.deadline_value)

        # 6. retrieval/rerank score как дополнительный, но не главный сигнал
        score += min(max(item.candidate_score, 0.0), 1.0) * 10.0

        return score
        
    def _fact_type_bonus(
        self,
        *,
        fact_type: str | None,
        question_text: str,
    ) -> float:
        question_kind = self._question_deadline_kind(question_text)

        if not fact_type:
            return 0.0

        generic_weights = {
            "decision_deadline": 60.0,
            "notification_deadline": 45.0,
            "payment_deadline": 40.0,
            "registration_deadline": 20.0,
            "correction_deadline": 5.0,
            "applicant_action_deadline": -20.0,
            "internal_procedure_deadline": -30.0,
        }

        decision_weights = {
            "decision_deadline": 100.0,
            "notification_deadline": 25.0,
            "payment_deadline": 10.0,
            "registration_deadline": 10.0,
            "correction_deadline": -10.0,
            "applicant_action_deadline": -40.0,
            "internal_procedure_deadline": -50.0,
        }

        notification_weights = {
            "decision_deadline": 20.0,
            "notification_deadline": 100.0,
            "payment_deadline": 5.0,
            "registration_deadline": 5.0,
            "correction_deadline": -10.0,
            "applicant_action_deadline": -40.0,
            "internal_procedure_deadline": -50.0,
        }

        payment_weights = {
            "decision_deadline": 10.0,
            "notification_deadline": 10.0,
            "payment_deadline": 100.0,
            "registration_deadline": 0.0,
            "correction_deadline": -20.0,
            "applicant_action_deadline": -50.0,
            "internal_procedure_deadline": -50.0,
        }

        registration_weights = {
            "decision_deadline": 10.0,
            "notification_deadline": 10.0,
            "payment_deadline": 0.0,
            "registration_deadline": 100.0,
            "correction_deadline": 0.0,
            "applicant_action_deadline": -30.0,
            "internal_procedure_deadline": -20.0,
        }

        correction_weights = {
            "decision_deadline": 0.0,
            "notification_deadline": 0.0,
            "payment_deadline": -20.0,
            "registration_deadline": 0.0,
            "correction_deadline": 100.0,
            "applicant_action_deadline": -20.0,
            "internal_procedure_deadline": -20.0,
        }

        by_kind = {
            "decision": decision_weights,
            "notification": notification_weights,
            "payment": payment_weights,
            "registration": registration_weights,
            "correction": correction_weights,
            None: generic_weights,
        }

        weights = by_kind.get(question_kind, generic_weights)
        return weights.get(fact_type, 0.0)
        
    def _question_deadline_kind(
        self,
        question_text: str,
    ) -> str | None:
        text = self._clean(question_text).lower()

        if any(x in text for x in ("выплат", "перечисл", "деньги", "26-го числа", "26 числа")):
            return "payment"

        if any(x in text for x in ("уведом", "сообщ", "извест", "когда ответят")):
            return "notification"

        if any(x in text for x in ("зарегистр", "регистрац")):
            return "registration"

        if any(x in text for x in ("опечат", "ошиб")):
            return "correction"

        if any(x in text for x in ("примут решение", "принятия решения", "когда примут", "срок предоставления")):
            return "decision"

        return None
        
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

    def _candidate_text(
        self,
        candidate: Any,
    ) -> str:
        payload = self._candidate_payload(candidate)
        return (
            self._candidate_attr(candidate, "content_text")
            or self._candidate_attr(candidate, "text")
            or payload.get("validity_note")
            or ""
        )

    def _candidate_score(
        self,
        candidate: Any,
    ) -> float:
        raw = (
            self._candidate_attr(candidate, "effective_score")
            or self._candidate_attr(candidate, "score")
            or self._candidate_attr(candidate, "source_score")
            or 0.0
        )
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0

    def render_text(
        self,
        *,
        result: DeadlinesAnswerBuildResult,
    ) -> Optional[str]:
        if not result.can_answer or result.primary_item is None:
            return None

        primary = result.primary_item
        question_kind = result.question_deadline_kind
        primary_label = self._DEADLINE_KIND_LABELS.get(
            primary.deadline_kind,
            self._DEADLINE_KIND_LABELS["other"],
        )

        # Главный режим: пользователь спросил про конкретный этап, и у нас
        # есть сильный primary этого же типа. Тогда отвечаем одним сроком,
        # не засоряя ответ уведомлением/выплатой из соседних пунктов.
        if self._should_render_single_primary(
            primary=primary,
            alternatives=result.alternative_items,
            question_deadline_kind=question_kind,
        ):
            if primary.scope_text:
                return (
                    f"Срок {primary_label} по найденным источникам: "
                    f"{primary.deadline_value} ({primary.scope_text})."
                )
            return f"Срок {primary_label} по найденным источникам: {primary.deadline_value}."

        # Осторожный fallback: выводим только близкие альтернативы. При
        # вопросе про конкретный этап не показываем альтернативы других типов,
        # чтобы не подменять ответ списком чужих сроков.
        visible_alternatives = self._select_visible_alternatives(
            primary=primary,
            alternatives=result.alternative_items,
            question_deadline_kind=question_kind,
        )

        if not visible_alternatives:
            if primary.scope_text:
                return (
                    f"Срок {primary_label} по найденным источникам: "
                    f"{primary.deadline_value} ({primary.scope_text})."
                )
            return f"Срок {primary_label} по найденным источникам: {primary.deadline_value}."

        lines: list[str] = ["По найденным источникам установлены следующие сроки:"]
        lines.append(self._render_bulleted_item(primary))
        for item in visible_alternatives:
            lines.append(self._render_bulleted_item(item))
        lines.append("")
        lines.append("Конкретный срок зависит от того, о каком действии или этапе процедуры идёт речь.")
        return "\n".join(lines)

    def _build_item_from_table_row(
        self,
        *,
        candidate: Any,
        normalized_question: str,
        question_deadline_kind: str,
        dropped_rows_debug: list[dict[str, Any]],
    ) -> Optional[DeadlineAnswerItem]:
        row_id = str(getattr(candidate, "source_id", "") or "")
        metadata = getattr(candidate, "metadata_json", None) or {}
        table_semantic_type = self._clean(metadata.get("table_semantic_type"))
        score = self._extract_candidate_score(candidate)

        cells = metadata.get("cells_by_semantic_key") or metadata.get("cells_by_header_key") or {}
        if not isinstance(cells, dict):
            dropped_rows_debug.append({"row_id": row_id, "reason": "cells_not_dict"})
            return None

        deadline_value = self._extract_deadline_value(cells)
        if not deadline_value:
            dropped_rows_debug.append(
                {
                    "row_id": row_id,
                    "reason": "empty_deadline_value",
                    "table_semantic_type": table_semantic_type,
                }
            )
            return None

        if self._is_service_value(deadline_value):
            dropped_rows_debug.append(
                {
                    "row_id": row_id,
                    "reason": "service_header_row",
                    "deadline_value": deadline_value,
                }
            )
            return None

        if not self._looks_like_deadline_value(deadline_value):
            dropped_rows_debug.append(
                {
                    "row_id": row_id,
                    "reason": "not_deadline_like_value",
                    "deadline_value": deadline_value,
                }
            )
            return None

        if table_semantic_type and table_semantic_type.lower() not in {"deadlines", "deadline"}:
            dropped_rows_debug.append(
                {
                    "row_id": row_id,
                    "reason": "not_deadlines_table",
                    "table_semantic_type": table_semantic_type,
                }
            )
            return None

        scope_text = self._extract_scope_text(cells)
        deadline_kind, kind_confidence = self._classify_deadline_kind(
            text=" ".join([deadline_value or "", scope_text or ""]),
        )
        priority_score = score + self._question_scope_bonus(
            question_text_normalized=normalized_question,
            question_deadline_kind=question_deadline_kind,
            item_deadline_kind=deadline_kind,
            scope_text=scope_text,
        )

        return DeadlineAnswerItem(
            deadline_value=deadline_value,
            scope_text=scope_text,
            deadline_kind=deadline_kind,
            kind_confidence=kind_confidence,
            source_row_ids=[row_id] if row_id else [],
            source_table_types=[table_semantic_type] if table_semantic_type else [],
            source_scores=[priority_score],
        )

    def _build_item_from_block(
        self,
        *,
        candidate: Any,
        normalized_question: str,
        question_deadline_kind: str,
        dropped_rows_debug: list[dict[str, Any]],
    ) -> Optional[DeadlineAnswerItem]:
        block_id = str(getattr(candidate, "source_id", "") or "")
        text = self._clean(getattr(candidate, "snippet", None) or getattr(candidate, "title", None) or "")
        if not text:
            dropped_rows_debug.append({"row_id": block_id, "reason": "empty_block_text"})
            return None

        if self._is_offtopic_deadline_block(text):
            dropped_rows_debug.append({"row_id": block_id, "reason": "offtopic_deadline_block"})
            return None

        if not self._has_block_deadline_marker(text):
            dropped_rows_debug.append({"row_id": block_id, "reason": "block_without_deadline_marker"})
            return None

        deadline_value = self._extract_deadline_value_from_block(text)
        if not deadline_value:
            dropped_rows_debug.append({"row_id": block_id, "reason": "block_deadline_not_extracted"})
            return None

        deadline_kind, kind_confidence = self._classify_deadline_kind(text=text)
        scope_text = self._extract_block_scope_text(text=text, deadline_kind=deadline_kind)
        score = self._extract_candidate_score(candidate)
        priority_score = score + self._question_scope_bonus(
            question_text_normalized=normalized_question,
            question_deadline_kind=question_deadline_kind,
            item_deadline_kind=deadline_kind,
            scope_text=scope_text,
        )

        return DeadlineAnswerItem(
            deadline_value=deadline_value,
            scope_text=scope_text,
            deadline_kind=deadline_kind,
            kind_confidence=kind_confidence,
            source_block_ids=[block_id] if block_id else [],
            source_scores=[priority_score],
        )

    def _merge_similar_items(
        self,
        items: list[DeadlineAnswerItem],
    ) -> tuple[list[DeadlineAnswerItem], list[dict[str, Any]]]:
        merged: dict[tuple[str, str], DeadlineAnswerItem] = {}
        debug: list[dict[str, Any]] = []

        for item in items:
            normalized_value = self._normalize(item.deadline_value)
            normalized_scope = self._normalize(item.scope_text)

            # Мержим по deadline_value + scope_text.
            # Если scope пустой, всё равно не теряем item.
            merge_key = (normalized_value, normalized_scope)

            existing = merged.get(merge_key)
            if existing is None:
                merged[merge_key] = DeadlineAnswerItem(
                    deadline_value=item.deadline_value,
                    scope_text=item.scope_text,
                    source_type=item.source_type,
                    citation_json=item.citation_json or {},
                    fact_type=item.fact_type,
                    is_service_core_deadline=item.is_service_core_deadline,
                    candidate_score=item.candidate_score,
                    table_title=item.table_title,
                    table_number=item.table_number,
                )
                debug.append(
                    {
                        "action": "create",
                        "merge_key": merge_key,
                        "deadline_value": item.deadline_value,
                        "scope_text": item.scope_text,
                        "source_type": item.source_type,
                        "fact_type": item.fact_type,
                        "is_service_core_deadline": item.is_service_core_deadline,
                        "candidate_score": item.candidate_score,
                    }
                )
                continue

            # 1. Сохраняем максимальный score.
            existing.candidate_score = max(existing.candidate_score, item.candidate_score)

            # 2. Если любой из item core -> merged тоже core.
            existing.is_service_core_deadline = (
                existing.is_service_core_deadline or item.is_service_core_deadline
            )

            # 3. legal_fact приоритетнее table_row.
            if existing.source_type != "legal_fact" and item.source_type == "legal_fact":
                existing.source_type = "legal_fact"
                existing.fact_type = item.fact_type or existing.fact_type
                if item.citation_json:
                    existing.citation_json = item.citation_json
                if item.scope_text and (
                    not existing.scope_text
                    or len(item.scope_text) > len(existing.scope_text)
                ):
                    existing.scope_text = item.scope_text

            # 4. Если fact_type ещё пустой — дозаполняем.
            if not existing.fact_type and item.fact_type:
                existing.fact_type = item.fact_type

            # 5. Берём более содержательный scope_text.
            if item.scope_text and (
                not existing.scope_text
                or len(self._clean(item.scope_text)) > len(self._clean(existing.scope_text))
            ):
                existing.scope_text = item.scope_text

            # 6. Если у existing нет citation_json — дозаполняем.
            if not existing.citation_json and item.citation_json:
                existing.citation_json = item.citation_json

            # 7. Если table metadata пустые — дозаполняем.
            if not existing.table_title and item.table_title:
                existing.table_title = item.table_title
            if not existing.table_number and item.table_number:
                existing.table_number = item.table_number

            debug.append(
                {
                    "action": "merge",
                    "merge_key": merge_key,
                    "existing_source_type": existing.source_type,
                    "incoming_source_type": item.source_type,
                    "deadline_value": existing.deadline_value,
                    "scope_text": existing.scope_text,
                    "fact_type": existing.fact_type,
                    "is_service_core_deadline": existing.is_service_core_deadline,
                    "candidate_score": existing.candidate_score,
                }
            )

        merged_items = list(merged.values())
        return merged_items, debug

    def _primary_sort_key(
        self,
        *,
        item: DeadlineAnswerItem,
        question_deadline_kind: str,
    ) -> tuple[int, float, float, float, int, int]:
        kind_bucket = 1
        if question_deadline_kind == "other":
            kind_bucket = 0
        elif item.deadline_kind == question_deadline_kind:
            kind_bucket = 0

        # table_row оставляем слегка предпочтительным, но не настолько,
        # чтобы подавлять хороший block по точному kind-совпадению.
        source_bonus = 0.05 if item.source_row_ids else 0.0

        return (
            kind_bucket,
            -(self._best_score(item) + source_bonus),
            -item.kind_confidence,
            -self._deadline_specificity_score(item.deadline_value),
            -self._same_kind_source_count(item),
            len(item.scope_text or ""),
        )

    def _extract_deadline_value(self, cells: dict[str, Any]) -> Optional[str]:
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

    def _extract_scope_text(self, cells: dict[str, Any]) -> Optional[str]:
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

    def _extract_deadline_value_from_block(self, text: str) -> Optional[str]:
        for pattern in self._BLOCK_DEADLINE_PATTERNS:
            match = pattern.search(text)
            if match:
                return self._clean(match.group(0))
        return None

    def _extract_block_scope_text(self, *, text: str, deadline_kind: str) -> Optional[str]:
        text_norm = self._normalize(text)

        if deadline_kind == "payment":
            if "едв" in text_norm:
                return "выплата ЕДВ"
            return "выплата"

        if deadline_kind == "notification":
            if "о назначении едв" in text_norm:
                return "уведомление о назначении ЕДВ"
            if "о предоставлении едв" in text_norm:
                return "уведомление о предоставлении ЕДВ"
            if "уведомление" in text_norm:
                return "уведомление о решении"
            return "уведомление"

        if deadline_kind == "decision":
            if "решение о предоставлении едв" in text_norm:
                return "принятие решения о предоставлении ЕДВ"
            if "решение о назначении едв" in text_norm:
                return "принятие решения о назначении ЕДВ"
            return "принятие решения"

        if "предоставлении услуги" in text_norm:
            return "предоставление услуги"
        return None

    def _detect_question_deadline_kind(self, question_text_normalized: str) -> str:
        kind, _ = self._classify_deadline_kind(text=question_text_normalized)
        return kind

    def _classify_deadline_kind(self, *, text: Optional[str]) -> tuple[str, float]:
        norm = self._normalize(text)
        if not norm:
            return ("other", 0.0)

        payment_score = self._score_by_markers(norm, self._PAYMENT_STRONG_MARKERS, 2.3)
        payment_score += self._score_by_markers(norm, self._PAYMENT_WEAK_MARKERS, 0.7)

        notification_score = self._score_by_markers(norm, self._NOTIFICATION_STRONG_MARKERS, 2.1)
        notification_score += self._score_by_markers(norm, self._NOTIFICATION_WEAK_MARKERS, 0.6)

        decision_score = self._score_by_markers(norm, self._DECISION_STRONG_MARKERS, 1.6)
        decision_score += self._score_by_markers(norm, self._DECISION_WEAK_MARKERS, 0.35)

        # Контекстные поправки.
        # Для payment/question block-а фраза "со дня принятия решения" не делает
        # срок decision, если в тексте есть явная выплата/перечисление.
        if payment_score > 0 and "принятия решения" in norm:
            decision_score *= 0.45

        # Аналогично для уведомления: наличие "принятия решения" - лишь опорная
        # точка после которой отправляют уведомление.
        if notification_score > 0 and "принятия решения" in norm:
            decision_score *= 0.55

        # Если вопрос/текст напрямую содержит "срок выплаты" или "срок уведомления",
        # усиливаем соответствующий kind.
        if "срок выплаты" in norm or "срок выплаты едв" in norm:
            payment_score += 1.5
        if "срок уведомления" in norm or "срок уведомления о решении" in norm:
            notification_score += 1.5
        if "срок принятия решения" in norm:
            decision_score += 1.5

        scores = {
            "decision": decision_score,
            "notification": notification_score,
            "payment": payment_score,
        }
        winner = max(scores, key=scores.get)
        best_score = scores[winner]
        if best_score <= 0:
            return ("other", 0.0)

        total = sum(scores.values()) or 1.0
        confidence = round(best_score / total, 3)

        # Если различие между лучшим и вторым кандидатом минимально,
        # не притворяемся уверенными и снижаем confidence.
        ordered_scores = sorted(scores.values(), reverse=True)
        if len(ordered_scores) >= 2 and ordered_scores[0] - ordered_scores[1] < 0.35:
            confidence = round(min(confidence, 0.58), 3)

        return (winner, confidence)

    def _question_scope_bonus(
        self,
        *,
        question_text_normalized: str,
        question_deadline_kind: str,
        item_deadline_kind: str,
        scope_text: Optional[str],
    ) -> float:
        bonus = 0.0
        if question_deadline_kind != "other" and item_deadline_kind == question_deadline_kind:
            bonus += 0.45
        elif question_deadline_kind != "other" and item_deadline_kind != "other":
            bonus -= 0.22

        if not question_text_normalized or not scope_text:
            return bonus

        scope_norm = self._normalize(scope_text)
        if not scope_norm:
            return bonus

        matched_terms = 0
        for term in question_text_normalized.split():
            if len(term) < 4:
                continue
            if term in scope_norm:
                matched_terms += 1
        if matched_terms >= 1:
            bonus += min(0.12, matched_terms * 0.04)
        return bonus

    def _deadline_specificity_score(self, value: str) -> int:
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

    def _best_score(self, item: DeadlineAnswerItem) -> float:
        if not item.source_scores:
            return 0.0
        return max(item.source_scores)

    def _same_kind_source_count(self, item: DeadlineAnswerItem) -> int:
        return len(item.source_row_ids) + len(item.source_block_ids)

    def _render_bulleted_item(self, item: DeadlineAnswerItem) -> str:
        label = self._DEADLINE_KIND_LABELS.get(item.deadline_kind, self._DEADLINE_KIND_LABELS["other"])
        if item.scope_text:
            return f"— {item.deadline_value} ({label}; {item.scope_text})"
        return f"— {item.deadline_value} ({label})"

    def _select_visible_alternatives(
        self,
        *,
        primary: DeadlineAnswerItem,
        alternatives: list[DeadlineAnswerItem],
        question_deadline_kind: str,
    ) -> list[DeadlineAnswerItem]:
        visible: list[DeadlineAnswerItem] = []
        primary_score = self._best_score(primary)

        for item in alternatives:
            # При конкретном вопросе показываем только альтернативы того же kind.
            if question_deadline_kind != "other" and item.deadline_kind != question_deadline_kind:
                continue

            # Даже для same-kind альтернатив показываем только действительно
            # близкие варианты, а не весь хвост.
            if primary_score - self._best_score(item) > 0.18:
                continue

            # Не плодим почти одинаковые строки с тем же сроком и тем же типом.
            if (
                self._normalize(item.deadline_value) == self._normalize(primary.deadline_value)
                and item.deadline_kind == primary.deadline_kind
            ):
                continue

            visible.append(item)
            if len(visible) >= 2:
                break

        return visible

    def _should_render_single_primary(
        self,
        *,
        primary: DeadlineAnswerItem,
        alternatives: list[DeadlineAnswerItem],
        question_deadline_kind: str,
    ) -> bool:
        if question_deadline_kind == "other":
            return not alternatives

        if primary.deadline_kind != question_deadline_kind:
            return False

        # Если следующая same-kind альтернатива сильно слабее, не раздуваем ответ.
        same_kind_alternatives = [
            item for item in alternatives if item.deadline_kind == question_deadline_kind
        ]
        if not same_kind_alternatives:
            return True

        next_item = same_kind_alternatives[0]
        if self._best_score(primary) - self._best_score(next_item) >= 0.18:
            return True

        # Высокая уверенность в kind + нормальный отрыв по score.
        if primary.kind_confidence >= 0.62 and self._best_score(primary) - self._best_score(next_item) >= 0.10:
            return True

        return False

    def _extract_candidate_score(self, candidate: Any) -> float:
        rerank_score = getattr(candidate, "rerank_score", None)
        if isinstance(rerank_score, (int, float)):
            return float(rerank_score)
        score = getattr(candidate, "score", None)
        if isinstance(score, (int, float)):
            return float(score)
        return 0.0

    def _has_block_deadline_marker(self, text: str) -> bool:
        norm = self._normalize(text)
        if not norm:
            return False
        if any(pattern.search(text) for pattern in self._BLOCK_DEADLINE_PATTERNS):
            return True
        return any(
            marker in norm
            for marker in (
                "рабочих дней",
                "календарных дней",
                "в течение",
                "не позднее",
                "не более",
            )
        )

    def _looks_like_deadline_value(self, text: str) -> bool:
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
            "до",
            "числа",
        ]
        return any(marker in text_norm for marker in deadline_markers) and any(ch.isdigit() for ch in text)

    def _is_service_value(self, text: str) -> bool:
        return self._normalize(text) in {self._normalize(x) for x in self._SERVICE_VALUES}

    def _is_offtopic_deadline_block(self, text: str) -> bool:
        norm = self._normalize(text)
        if not norm:
            return False
        return any(marker in norm for marker in self._OFFTOPIC_BLOCK_MARKERS)

    def _score_by_markers(self, norm: str, markers: tuple[str, ...], weight: float) -> float:
        score = 0.0
        for marker in markers:
            if marker in norm:
                score += weight
        return score

    def _pretty_label(self, key: Any) -> Optional[str]:
        if key is None:
            return None
        text = str(key).strip().replace("_", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text or None

    def _clean(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        text = re.sub(r"\s+", " ", text).strip()
        return text or None

    def _normalize(self, value: Any) -> str:
        cleaned = self._clean(value)
        if not cleaned:
            return ""
        return cleaned.lower().replace("ё", "е")
        
    def _table_row_is_service_core(
        self,
        scope_text: str,
    ) -> bool:
        text = self._clean(scope_text).lower()

        if any(
            marker in text
            for marker in (
                "предоставления государственной услуги",
                "принятия решения",
                "о принятом решении",
                "уведомляется",
                "выплата",
                "не позднее 26-го числа",
                "регистрация запроса",
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
            )
        ):
            return False

        return False
