from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(slots=True)
class DeadlineAnswerItem:
    deadline_value: str
    scope_text: Optional[str] = None
    deadline_kind: str = "other"
    kind_confidence: float = 0.0
    source_row_ids: list[str] = field(default_factory=list)
    source_block_ids: list[str] = field(default_factory=list)
    source_table_types: list[str] = field(default_factory=list)
    source_scores: list[float] = field(default_factory=list)


@dataclass(slots=True)
class DeadlinesAnswerBuildResult:
    can_answer: bool
    question_deadline_kind: str = "other"
    primary_item: Optional[DeadlineAnswerItem] = None
    alternative_items: list[DeadlineAnswerItem] = field(default_factory=list)
    dropped_rows_debug: list[dict[str, Any]] = field(default_factory=list)
    merged_items_debug: list[dict[str, Any]] = field(default_factory=list)
    ambiguity_reason: Optional[str] = None
    reason: Optional[str] = None

    @property
    def all_items(self) -> list[DeadlineAnswerItem]:
        items: list[DeadlineAnswerItem] = []
        if self.primary_item is not None:
            items.append(self.primary_item)
        items.extend(self.alternative_items)
        return items

    def debug_payload(self) -> dict[str, Any]:
        return {
            "can_answer": self.can_answer,
            "question_deadline_kind": self.question_deadline_kind,
            "primary_item": (
                {
                    "deadline_value": self.primary_item.deadline_value,
                    "scope_text": self.primary_item.scope_text,
                    "deadline_kind": self.primary_item.deadline_kind,
                    "kind_confidence": self.primary_item.kind_confidence,
                    "source_row_ids": self.primary_item.source_row_ids,
                    "source_block_ids": self.primary_item.source_block_ids,
                    "source_table_types": self.primary_item.source_table_types,
                    "source_scores": self.primary_item.source_scores,
                }
                if self.primary_item is not None
                else None
            ),
            "alternative_items_count": len(self.alternative_items),
            "items": [
                {
                    "deadline_value": item.deadline_value,
                    "scope_text": item.scope_text,
                    "deadline_kind": item.deadline_kind,
                    "kind_confidence": item.kind_confidence,
                    "source_row_ids": item.source_row_ids,
                    "source_block_ids": item.source_block_ids,
                    "source_table_types": item.source_table_types,
                    "source_scores": item.source_scores,
                }
                for item in self.all_items
            ],
            "merged_items": self.merged_items_debug,
            "dropped_rows": self.dropped_rows_debug,
            "ambiguity_reason": self.ambiguity_reason,
            "reason": self.reason,
        }


class TableDeadlinesAnswerBuilder:
    """
    Deterministic builder for deadline questions.

    Почему builder теперь работает не только с table_row:
    - в текущем проекте часть реальных сроков лежит в таблицах,
      но часть лежит в обычных paragraph/list_item блоках;
    - для deadline-question нам важен не тип источника сам по себе,
      а наличие явной temporal-formula и понятного procedural scope;
    - builder по-прежнему не ходит в БД и работает только с уже
      выбранными retrieval-кандидатами.
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

    _DEADLINE_KIND_MARKERS: dict[str, tuple[str, ...]] = {
        "decision": (
            "принятия решения",
            "принятие решения",
            "решение о предоставлении",
            "решение о назначении",
            "рассмотрения заявления",
            "рассмотрение заявления",
            "рассмотрения документов",
            "регистрации заявления",
            "назначении",
            "назначение",
        ),
        "notification": (
            "уведомления",
            "направляет уведомление",
            "направление уведомления",
            "направляет заявителю",
            "уведомить",
            "информирования",
            "информирование",
            "сообщения о решении",
            "извещение",
        ),
        "payment": (
            "выплаты",
            "выплата",
            "выплачивается",
            "перечисления",
            "перечисление",
            "зачисления",
            "зачисление",
            "осуществления выплаты",
            "доставки выплаты",
            "26-го числа",
            "26 числа",
        ),
    }

    _DEADLINE_KIND_LABELS: dict[str, str] = {
        "decision": "принятия решения",
        "notification": "уведомления",
        "payment": "выплаты",
        "other": "срока",
    }

    _BLOCK_DEADLINE_PATTERNS = [
        re.compile(r"в течение\s+\d+\s+(?:рабоч(?:их|его)|календарн(?:ых|ого))\s+дн(?:я|ей)", re.IGNORECASE),
        re.compile(r"в течение\s+\d+\s+дн(?:я|ей)", re.IGNORECASE),
        re.compile(r"не позднее\s+[^.;]{3,120}", re.IGNORECASE),
        re.compile(r"не более\s+\d+\s+(?:рабоч(?:их|его)|календарн(?:ых|ого))\s+дн(?:я|ей)", re.IGNORECASE),
    ]

    def build(
        self,
        *,
        candidates: list[Any],
        question_text: Optional[str] = None,
    ) -> DeadlinesAnswerBuildResult:
        raw_items: list[DeadlineAnswerItem] = []
        dropped_rows_debug: list[dict[str, Any]] = []

        normalized_question = self._normalize(question_text)
        question_deadline_kind = self._detect_question_deadline_kind(normalized_question)

        for candidate in candidates:
            source_type = getattr(candidate, "source_type", None)
            if source_type == "table_row":
                item = self._build_item_from_table_row(
                    candidate=candidate,
                    normalized_question=normalized_question,
                    question_deadline_kind=question_deadline_kind,
                    dropped_rows_debug=dropped_rows_debug,
                )
            elif source_type == "block":
                item = self._build_item_from_block(
                    candidate=candidate,
                    normalized_question=normalized_question,
                    question_deadline_kind=question_deadline_kind,
                    dropped_rows_debug=dropped_rows_debug,
                )
            else:
                continue

            if item is not None:
                raw_items.append(item)

        if not raw_items:
            return DeadlinesAnswerBuildResult(
                can_answer=False,
                question_deadline_kind=question_deadline_kind,
                reason="no_deadline_items",
                dropped_rows_debug=dropped_rows_debug,
            )

        merged_items, merged_items_debug = self._merge_similar_items(raw_items)
        if not merged_items:
            return DeadlinesAnswerBuildResult(
                can_answer=False,
                question_deadline_kind=question_deadline_kind,
                reason="no_merged_deadline_items",
                dropped_rows_debug=dropped_rows_debug,
                merged_items_debug=merged_items_debug,
            )

        merged_items.sort(
            key=lambda item: self._primary_sort_key(
                item=item,
                question_deadline_kind=question_deadline_kind,
            )
        )

        primary_item = merged_items[0]
        alternative_items = merged_items[1:]

        ambiguity_reason: Optional[str] = None
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
            merged_items_debug=merged_items_debug,
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
        label = self._DEADLINE_KIND_LABELS.get(primary.deadline_kind, self._DEADLINE_KIND_LABELS["other"])

        if not result.alternative_items:
            if primary.scope_text:
                return f"Срок {label} по найденным источникам: {primary.deadline_value} ({primary.scope_text})."
            return f"Срок {label} по найденным источникам: {primary.deadline_value}."

        lines: list[str] = ["По найденным источникам установлены следующие сроки:"]
        lines.append(self._render_bulleted_item(primary))
        for item in result.alternative_items:
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
            dropped_rows_debug.append({
                "row_id": row_id,
                "reason": "empty_deadline_value",
                "table_semantic_type": table_semantic_type,
            })
            return None

        if self._is_service_value(deadline_value):
            dropped_rows_debug.append({
                "row_id": row_id,
                "reason": "service_header_row",
                "deadline_value": deadline_value,
            })
            return None

        if not self._looks_like_deadline_value(deadline_value):
            dropped_rows_debug.append({
                "row_id": row_id,
                "reason": "not_deadline_like_value",
                "deadline_value": deadline_value,
            })
            return None

        if table_semantic_type and table_semantic_type.lower() not in {"deadlines", "deadline"}:
            dropped_rows_debug.append({
                "row_id": row_id,
                "reason": "not_deadlines_table",
                "table_semantic_type": table_semantic_type,
            })
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
        groups: dict[tuple[str, str, str], list[DeadlineAnswerItem]] = {}
        for item in items:
            key = (
                self._normalize(item.deadline_value),
                self._normalize(item.scope_text),
                item.deadline_kind,
            )
            groups.setdefault(key, []).append(item)

        merged_items: list[DeadlineAnswerItem] = []
        merged_items_debug: list[dict[str, Any]] = []

        for _, group_items in groups.items():
            best_item = sorted(
                group_items,
                key=lambda item: (
                    -self._best_score(item),
                    -item.kind_confidence,
                    -self._deadline_specificity_score(item.deadline_value),
                    len(item.scope_text or ""),
                ),
            )[0]

            row_ids: list[str] = []
            block_ids: list[str] = []
            table_types: list[str] = []
            scores: list[float] = []
            for item in group_items:
                for row_id in item.source_row_ids:
                    if row_id and row_id not in row_ids:
                        row_ids.append(row_id)
                for block_id in item.source_block_ids:
                    if block_id and block_id not in block_ids:
                        block_ids.append(block_id)
                for table_type in item.source_table_types:
                    if table_type and table_type not in table_types:
                        table_types.append(table_type)
                scores.extend(item.source_scores)

            merged_item = DeadlineAnswerItem(
                deadline_value=best_item.deadline_value,
                scope_text=best_item.scope_text,
                deadline_kind=best_item.deadline_kind,
                kind_confidence=max((item.kind_confidence for item in group_items), default=0.0),
                source_row_ids=row_ids,
                source_block_ids=block_ids,
                source_table_types=table_types,
                source_scores=sorted(scores, reverse=True),
            )
            merged_items.append(merged_item)

            if len(group_items) > 1:
                merged_items_debug.append({
                    "merged_deadline_value": merged_item.deadline_value,
                    "merged_scope_text": merged_item.scope_text,
                    "deadline_kind": merged_item.deadline_kind,
                    "source_row_ids": row_ids,
                    "source_block_ids": block_ids,
                    "source_items_count": len(group_items),
                })

        return merged_items, merged_items_debug

    def _primary_sort_key(
        self,
        *,
        item: DeadlineAnswerItem,
        question_deadline_kind: str,
    ) -> tuple[int, int, float, float, int, int]:
        kind_match_bucket = 1
        if question_deadline_kind == "other":
            kind_match_bucket = 0
        elif item.deadline_kind == question_deadline_kind:
            kind_match_bucket = 0

        source_bucket = 1
        if item.source_row_ids:
            source_bucket = 0
        elif item.source_block_ids:
            source_bucket = 1

        return (
            kind_match_bucket,
            source_bucket,
            -self._best_score(item),
            -item.kind_confidence,
            -self._deadline_specificity_score(item.deadline_value),
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
        if deadline_kind == "decision":
            return "принятие решения"
        if deadline_kind == "notification":
            return "уведомление о решении"
        if deadline_kind == "payment":
            return "выплата"
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

        scores: dict[str, float] = {"decision": 0.0, "notification": 0.0, "payment": 0.0}
        for kind, markers in self._DEADLINE_KIND_MARKERS.items():
            for marker in markers:
                if marker in norm:
                    scores[kind] += 1.0

        winner = max(scores, key=scores.get)
        best_score = scores[winner]
        if best_score <= 0:
            return ("other", 0.0)
        total = sum(scores.values()) or 1.0
        return (winner, round(best_score / total, 3))

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
            bonus += 0.40
        elif question_deadline_kind != "other" and item_deadline_kind != "other":
            bonus -= 0.18

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
        return score

    def _best_score(self, item: DeadlineAnswerItem) -> float:
        if not item.source_scores:
            return 0.0
        return max(item.source_scores)

    def _render_bulleted_item(self, item: DeadlineAnswerItem) -> str:
        label = self._DEADLINE_KIND_LABELS.get(item.deadline_kind, self._DEADLINE_KIND_LABELS["other"])
        if item.scope_text:
            return f"— {item.deadline_value} ({label}; {item.scope_text})"
        return f"— {item.deadline_value} ({label})"

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
        return any(marker in norm for marker in (
            "рабочих дней",
            "календарных дней",
            "в течение",
            "не позднее",
            "не более",
        ))

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
