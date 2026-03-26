from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(slots=True)
class DeadlineAnswerItem:
    deadline_value: str
    scope_text: Optional[str] = None
    deadline_kind: str = "other"
    kind_confidence: float = 0.0
    source_row_ids: list[str] = field(default_factory=list)
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
    Deterministic builder for deadline questions based on table rows.

    Principles:
    - works only with retrieval-selected candidates
    - does not query DB directly
    - prefers rows from extractor-classified deadlines tables
    - distinguishes deadline kinds generically:
      decision / notification / payment / other
    - stays conservative when multiple different deadlines are found
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
            "принять решение",
            "рассмотрения заявления",
            "рассмотрение заявления",
            "рассмотрения документов",
            "рассмотрение документов",
            "регистрации заявления",
            "регистрация заявления",
            "назначении",
            "назначение",
            "предоставления государственной услуги",
            "предоставления услуги",
        ),
        "notification": (
            "уведомления",
            "направления уведомления",
            "направление уведомления",
            "уведомить",
            "информирования",
            "информирование",
            "сообщения о решении",
            "сообщение о решении",
            "извещения",
            "извещение",
        ),
        "payment": (
            "выплаты",
            "выплата",
            "срок выплаты",
            "перечисления",
            "перечисление",
            "зачисления",
            "зачисление",
            "доставки выплаты",
            "осуществления выплаты",
            "первая выплата",
        ),
    }

    _DEADLINE_KIND_LABELS: dict[str, str] = {
        "decision": "принятия решения",
        "notification": "уведомления",
        "payment": "выплаты",
        "other": "срока",
    }

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
            if getattr(candidate, "source_type", None) != "table_row":
                continue

            row_id = str(getattr(candidate, "source_id", "") or "")
            metadata = getattr(candidate, "metadata_json", None) or {}
            table_semantic_type = self._clean(metadata.get("table_semantic_type"))
            score = self._extract_candidate_score(candidate)

            cells = metadata.get("cells_by_semantic_key") or metadata.get("cells_by_header_key") or {}
            if not isinstance(cells, dict):
                dropped_rows_debug.append(
                    {
                        "row_id": row_id,
                        "reason": "cells_not_dict",
                    }
                )
                continue

            deadline_value = self._extract_deadline_value(cells)
            if not deadline_value:
                dropped_rows_debug.append(
                    {
                        "row_id": row_id,
                        "reason": "empty_deadline_value",
                        "table_semantic_type": table_semantic_type,
                    }
                )
                continue

            if self._is_service_value(deadline_value):
                dropped_rows_debug.append(
                    {
                        "row_id": row_id,
                        "reason": "service_header_row",
                        "deadline_value": deadline_value,
                    }
                )
                continue

            if not self._looks_like_deadline_value(deadline_value):
                dropped_rows_debug.append(
                    {
                        "row_id": row_id,
                        "reason": "not_deadline_like_value",
                        "deadline_value": deadline_value,
                    }
                )
                continue

            if table_semantic_type and table_semantic_type.lower() not in {"deadlines", "deadline"}:
                # Для deadline-builder не берём чужие таблицы только по случайному совпадению слов.
                dropped_rows_debug.append(
                    {
                        "row_id": row_id,
                        "reason": "not_deadlines_table",
                        "table_semantic_type": table_semantic_type,
                    }
                )
                continue

            scope_text = self._extract_scope_text(cells)
            deadline_kind, kind_confidence = self._classify_deadline_kind(
                cells=cells,
                scope_text=scope_text,
            )
            question_bonus = self._question_scope_bonus(
                question_text_normalized=normalized_question,
                question_deadline_kind=question_deadline_kind,
                item_deadline_kind=deadline_kind,
                scope_text=scope_text,
            )
            priority_score = score + question_bonus

            raw_items.append(
                DeadlineAnswerItem(
                    deadline_value=deadline_value,
                    scope_text=scope_text,
                    deadline_kind=deadline_kind,
                    kind_confidence=kind_confidence,
                    source_row_ids=[row_id] if row_id else [],
                    source_table_types=[table_semantic_type] if table_semantic_type else [],
                    source_scores=[priority_score],
                )
            )

        if not raw_items:
            return DeadlinesAnswerBuildResult(
                can_answer=False,
                question_deadline_kind=question_deadline_kind,
                reason="no_deadline_rows",
                dropped_rows_debug=dropped_rows_debug,
            )

        merged_items, merged_items_debug = self._merge_similar_items(raw_items)

        if not merged_items:
            return DeadlinesAnswerBuildResult(
                can_answer=False,
                question_deadline_kind=question_deadline_kind,
                reason="no_merged_deadline_rows",
                dropped_rows_debug=dropped_rows_debug,
                merged_items_debug=merged_items_debug,
            )

        merged_items.sort(
            key=lambda item: (
                -self._deadline_kind_alignment_score(
                    question_deadline_kind=question_deadline_kind,
                    item_deadline_kind=item.deadline_kind,
                ),
                -self._best_score(item),
                -self._deadline_specificity_score(item.deadline_value),
                len(item.scope_text or ""),
                len(item.deadline_value),
            )
        )

        primary_item = merged_items[0]
        alternative_items = merged_items[1:]

        ambiguity_reason: Optional[str] = None
        if alternative_items:
            if any(
                self._normalize(item.deadline_value) != self._normalize(primary_item.deadline_value)
                for item in alternative_items
            ):
                ambiguity_reason = "multiple_distinct_deadlines"

        if (
            question_deadline_kind != "other"
            and primary_item.deadline_kind != question_deadline_kind
        ):
            ambiguity_reason = ambiguity_reason or "target_deadline_kind_not_confirmed"

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
        target_label = self._deadline_kind_label(result.question_deadline_kind)
        has_target_match = (
            result.question_deadline_kind != "other"
            and primary.deadline_kind == result.question_deadline_kind
        )

        if not result.alternative_items:
            intro = "Срок по найденным источникам"
            if has_target_match:
                intro = f"Срок {target_label} по найденным источникам"

            if primary.scope_text:
                text = f"{intro}: {primary.deadline_value} ({primary.scope_text})."
            else:
                text = f"{intro}: {primary.deadline_value}."

            if result.question_deadline_kind != "other" and not has_target_match:
                text += (
                    f" Не удалось однозначно подтвердить, что это именно срок {target_label}."
                )
            return text

        lines: list[str] = []
        lines.append("По найденным источникам установлены следующие сроки:")
        lines.append(self._render_bulleted_item(primary))

        for item in result.alternative_items:
            lines.append(self._render_bulleted_item(item))

        lines.append("")
        if result.question_deadline_kind != "other":
            if has_target_match:
                lines.append(
                    f"Для вопроса о сроке {target_label} ориентируйся прежде всего на соответствующую строку."
                )
            else:
                lines.append(
                    f"Не удалось однозначно подтвердить, какая из найденных строк относится именно к сроку {target_label}."
                )
        else:
            lines.append(
                "Конкретный срок зависит от того, о каком действии или этапе процедуры идёт речь."
            )
        return "\n".join(lines)

    def _merge_similar_items(
        self,
        items: list[DeadlineAnswerItem],
    ) -> tuple[list[DeadlineAnswerItem], list[dict[str, Any]]]:
        groups: dict[tuple[str, str, str], list[DeadlineAnswerItem]] = {}

        for item in items:
            key = (
                item.deadline_kind,
                self._normalize(item.deadline_value),
                self._normalize(item.scope_text),
            )
            groups.setdefault(key, []).append(item)

        merged_items: list[DeadlineAnswerItem] = []
        merged_items_debug: list[dict[str, Any]] = []

        for _, group_items in groups.items():
            best_item = sorted(
                group_items,
                key=lambda item: (
                    -self._best_score(item),
                    -self._deadline_specificity_score(item.deadline_value),
                    item.deadline_kind != "other",
                    len(item.scope_text or ""),
                    len(item.deadline_value),
                ),
            )[0]

            row_ids: list[str] = []
            table_types: list[str] = []
            scores: list[float] = []
            confidences: list[float] = []

            for item in group_items:
                for row_id in item.source_row_ids:
                    if row_id and row_id not in row_ids:
                        row_ids.append(row_id)
                for table_type in item.source_table_types:
                    if table_type and table_type not in table_types:
                        table_types.append(table_type)
                scores.extend(item.source_scores)
                confidences.append(item.kind_confidence)

            merged_item = DeadlineAnswerItem(
                deadline_value=best_item.deadline_value,
                scope_text=best_item.scope_text,
                deadline_kind=best_item.deadline_kind,
                kind_confidence=max(confidences) if confidences else best_item.kind_confidence,
                source_row_ids=row_ids,
                source_table_types=table_types,
                source_scores=sorted(scores, reverse=True),
            )
            merged_items.append(merged_item)

            if len(group_items) > 1:
                merged_items_debug.append(
                    {
                        "merged_deadline_value": merged_item.deadline_value,
                        "merged_scope_text": merged_item.scope_text,
                        "merged_deadline_kind": merged_item.deadline_kind,
                        "merged_kind_confidence": merged_item.kind_confidence,
                        "source_row_ids": row_ids,
                        "source_items_count": len(group_items),
                    }
                )

        return merged_items, merged_items_debug

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

    def _question_scope_bonus(
        self,
        *,
        question_text_normalized: str,
        question_deadline_kind: str,
        item_deadline_kind: str,
        scope_text: Optional[str],
    ) -> float:
        bonus = 0.0

        if not question_text_normalized:
            return bonus

        if question_deadline_kind != "other":
            alignment_score = self._deadline_kind_alignment_score(
                question_deadline_kind=question_deadline_kind,
                item_deadline_kind=item_deadline_kind,
            )
            if alignment_score >= 3:
                bonus += 0.35
            elif alignment_score == 1:
                bonus -= 0.04
            else:
                bonus -= 0.18

        if not scope_text:
            return bonus

        scope_norm = self._normalize(scope_text)
        if not scope_norm:
            return bonus

        strong_markers = [
            "принятия решения",
            "рассмотрения заявления",
            "предоставления услуги",
            "регистрации заявления",
            "направления уведомления",
            "уведомления",
            "перечисления",
            "выплаты",
        ]
        for marker in strong_markers:
            if marker in question_text_normalized and marker in scope_norm:
                bonus += 0.18

        matched_terms = 0
        for term in question_text_normalized.split():
            if len(term) < 4:
                continue
            if term in scope_norm:
                matched_terms += 1

        if matched_terms >= 1:
            bonus += min(0.12, matched_terms * 0.04)

        return bonus

    def _classify_deadline_kind(
        self,
        *,
        cells: dict[str, Any],
        scope_text: Optional[str],
    ) -> tuple[str, float]:
        scores = {
            "decision": 0.0,
            "notification": 0.0,
            "payment": 0.0,
        }

        weighted_cell_keys = {
            "step",
            "action",
            "result",
            "comment",
            "deadline_subject",
            "procedure_stage",
        }

        text_fragments: list[tuple[str, float]] = []
        if scope_text:
            text_fragments.append((self._normalize(scope_text), 1.0))

        for key, value in cells.items():
            if key in self._IGNORED_SCOPE_KEYS:
                continue
            cleaned = self._clean(value)
            if not cleaned:
                continue
            weight = 1.35 if str(key).strip().lower() in weighted_cell_keys else 1.0
            text_fragments.append((self._normalize(cleaned), weight))

        for text, weight in text_fragments:
            if not text:
                continue
            for kind, markers in self._DEADLINE_KIND_MARKERS.items():
                for marker in markers:
                    if marker in text:
                        scores[kind] += weight

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_kind, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        if best_score <= 0.0:
            return ("other", 0.0)

        if second_score > 0.0 and abs(best_score - second_score) < 0.6:
            return ("other", round(best_score, 3))

        return (best_kind, round(best_score, 3))

    def _detect_question_deadline_kind(self, question_text_normalized: str) -> str:
        text = self._normalize(question_text_normalized)
        if not text:
            return "other"

        scores = {
            "decision": 0,
            "notification": 0,
            "payment": 0,
        }
        for kind, markers in self._DEADLINE_KIND_MARKERS.items():
            for marker in markers:
                if marker in text:
                    scores[kind] += 1

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_kind, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0

        if best_score <= 0:
            return "other"
        if second_score == best_score and best_score > 0:
            return "other"
        return best_kind

    def _deadline_kind_alignment_score(
        self,
        *,
        question_deadline_kind: str,
        item_deadline_kind: str,
    ) -> int:
        if question_deadline_kind == "other":
            return 1 if item_deadline_kind != "other" else 0

        if item_deadline_kind == question_deadline_kind:
            return 3

        if item_deadline_kind == "other":
            return 1

        return 0

    def _deadline_kind_label(self, deadline_kind: str) -> str:
        return self._DEADLINE_KIND_LABELS.get(deadline_kind, self._DEADLINE_KIND_LABELS["other"])

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
        kind_prefix = ""
        if item.deadline_kind != "other":
            kind_prefix = f"{self._deadline_kind_label(item.deadline_kind)}: "

        if item.scope_text:
            return f"— {kind_prefix}{item.deadline_value} ({item.scope_text})"
        return f"— {kind_prefix}{item.deadline_value}"

    def _extract_candidate_score(self, candidate: Any) -> float:
        rerank_score = getattr(candidate, "rerank_score", None)
        if isinstance(rerank_score, (int, float)):
            return float(rerank_score)

        score = getattr(candidate, "score", None)
        if isinstance(score, (int, float)):
            return float(score)

        return 0.0

    def _looks_like_deadline_value(self, value: str) -> bool:
        text = self._normalize(value)
        if not text:
            return False

        markers = [
            "срок",
            "рабоч",
            "календар",
            "дней",
            "дня",
            "день",
            "суток",
            "месяц",
            "месяцев",
            "год",
            "лет",
            "не позднее",
            "не более",
            "в течение",
        ]
        if any(marker in text for marker in markers):
            return True

        return any(ch.isdigit() for ch in value)

    def _is_service_value(self, value: str) -> bool:
        text = self._normalize(value)
        if text in self._SERVICE_VALUES:
            return True

        if text.startswith("срок ") and len(text.split()) <= 4:
            return True

        return False

    def _pretty_label(self, key: Any) -> Optional[str]:
        if key is None:
            return None

        text = " ".join(str(key).strip().split())
        if not text:
            return None

        replacements = {
            "applicant_category_id": "Категория заявителя",
            "applicant_category_name": "Категория заявителя",
            "step": "Этап",
            "action": "Действие",
            "result": "Результат",
            "comment": "Примечание",
        }
        if text in replacements:
            return replacements[text]

        text = text.replace("_", " ")
        if not text:
            return None

        return text[:1].upper() + text[1:]

    def _normalize(self, value: Any) -> str:
        if value is None:
            return ""

        text = " ".join(str(value).strip().split())
        text = text.lower()

        for ch in [",", ".", ";", ":", "(", ")", "«", "»", "\"", "'"]:
            text = text.replace(ch, " ")

        return " ".join(text.split())

    def _clean(self, value: Any) -> Optional[str]:
        if value is None:
            return None

        text = " ".join(str(value).strip().split())
        return text or None
