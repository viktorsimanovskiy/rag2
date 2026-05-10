from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(slots=True)
class RejectionReasonItem:
    reason_text: str
    rejection_scope: str = "service_refusal"
    applicant_category_ids: list[str] = field(default_factory=list)
    source_row_ids: list[str] = field(default_factory=list)
    source_scores: list[float] = field(default_factory=list)
    row_order: Optional[int] = None


@dataclass(slots=True)
class RejectionReasonsBuildResult:
    can_answer: bool
    question_scope: str = "service_refusal"
    items: list[RejectionReasonItem] = field(default_factory=list)
    dropped_rows_debug: list[dict[str, Any]] = field(default_factory=list)
    merged_items_debug: list[dict[str, Any]] = field(default_factory=list)
    reason: Optional[str] = None

    def debug_payload(self) -> dict[str, Any]:
        return {
            "can_answer": self.can_answer,
            "question_scope": self.question_scope,
            "items_count": len(self.items),
            "items": [
                {
                    "reason_text": item.reason_text,
                    "rejection_scope": item.rejection_scope,
                    "applicant_category_ids": item.applicant_category_ids,
                    "source_row_ids": item.source_row_ids,
                    "source_scores": item.source_scores,
                    "row_order": item.row_order,
                }
                for item in self.items
            ],
            "merged_items": self.merged_items_debug,
            "dropped_rows": self.dropped_rows_debug,
            "reason": self.reason,
        }


class TableRejectionReasonsAnswerBuilder:
    """
    Deterministic builder for refusal / rejection questions.

    Первая версия специально узкая:
    - работаем только по retrieval-selected table_row;
    - берем только таблицы refusal_reasons / rejection_reasons;
    - для общего вопроса "почему могут отказать" по умолчанию целимся
      в отказ в предоставлении услуги, а не в отказ в приеме документов.
    """

    _SERVICE_VALUES = {
        "основания отказа",
        "основания для отказа",
        "основания для отказа в предоставлении государственной услуги",
        "основания для отказа в предоставлении",
        "основания для отказа в приеме",
        "основания для приостановления",
        "отказ в приеме",
        "приостановление",
        "основание отказа",
    }

    def build(
        self,
        *,
        candidates: list[Any],
        question_text: str,
    ) -> RejectionReasonsBuildResult:
        question_scope = self._detect_question_scope(question_text)
        raw_items: list[RejectionReasonItem] = []
        dropped_rows_debug: list[dict[str, Any]] = []

        for candidate in candidates:
            if getattr(candidate, "source_type", None) != "table_row":
                continue

            row_id = str(getattr(candidate, "source_id", "") or "")
            metadata = getattr(candidate, "metadata_json", None) or {}
            table_semantic_type = self._normalize_text(
                metadata.get("table_semantic_type")
            )

            if table_semantic_type not in {"refusal_reasons", "rejection_reasons"}:
                dropped_rows_debug.append(
                    {
                        "row_id": row_id,
                        "reason": "not_rejection_table",
                    }
                )
                continue

            cells = (
                metadata.get("cells_by_semantic_key")
                or metadata.get("cells_by_header_key")
                or {}
            )
            if not isinstance(cells, dict):
                dropped_rows_debug.append(
                    {
                        "row_id": row_id,
                        "reason": "cells_not_dict",
                    }
                )
                continue

            reason_text = self._clean(cells.get("refusal_reason"))
            if not reason_text:
                reason_text = self._clean(getattr(candidate, "snippet", None))

            if not reason_text:
                dropped_rows_debug.append(
                    {
                        "row_id": row_id,
                        "reason": "empty_refusal_reason",
                    }
                )
                continue

            if self._is_service_value(reason_text):
                dropped_rows_debug.append(
                    {
                        "row_id": row_id,
                        "reason": "service_header_row",
                        "reason_text": reason_text,
                    }
                )
                continue

            row_scope = self._normalize_text(metadata.get("row_scope"))
            if not row_scope or row_scope == "other":
                row_scope = self._classify_reason_scope(reason_text)

            if question_scope != "other" and row_scope not in {question_scope, "other"}:
                dropped_rows_debug.append(
                    {
                        "row_id": row_id,
                        "reason": "scope_mismatch",
                        "row_scope": row_scope,
                        "question_scope": question_scope,
                        "reason_text": reason_text,
                    }
                )
                continue

            applicant_category_id = self._clean(cells.get("applicant_category_id"))
            row_order = getattr(candidate, "row_order", None)
            score = float(
                getattr(candidate, "rerank_score", None)
                or getattr(candidate, "score", 0.0)
                or 0.0
            )

            raw_items.append(
                RejectionReasonItem(
                    reason_text=reason_text,
                    rejection_scope=row_scope if row_scope != "other" else question_scope,
                    applicant_category_ids=[applicant_category_id]
                    if applicant_category_id
                    else [],
                    source_row_ids=[row_id] if row_id else [],
                    source_scores=[score],
                    row_order=row_order,
                )
            )

        if not raw_items:
            return RejectionReasonsBuildResult(
                can_answer=False,
                question_scope=question_scope,
                dropped_rows_debug=dropped_rows_debug,
                reason="no_rejection_rows",
            )

        merged_items, merged_items_debug = self._merge_duplicate_items(raw_items)

        merged_items.sort(
            key=lambda item: (
                item.row_order if item.row_order is not None else 10**9,
                -max(item.source_scores) if item.source_scores else 0.0,
                item.reason_text.lower(),
            )
        )

        return RejectionReasonsBuildResult(
            can_answer=True,
            question_scope=question_scope,
            items=merged_items[:12],
            dropped_rows_debug=dropped_rows_debug,
            merged_items_debug=merged_items_debug,
        )

    def render_text(
        self,
        *,
        result: RejectionReasonsBuildResult,
    ) -> str:
        intro_map = {
            "service_refusal": "По найденным источникам основания отказа в предоставлении услуги следующие.",
            "intake_refusal": "По найденным источникам основания отказа в приеме документов следующие.",
            "suspension": "По найденным источникам основания для приостановления следующие.",
            "renewal_refusal": "По найденным источникам основания отказа в возобновлении следующие.",
            "other": "По найденным источникам причины отказа сформулированы следующим образом.",
        }

        lines = [intro_map.get(result.question_scope, intro_map["other"])]

        for item in result.items[:10]:
            lines.append(f"— {self._normalize_sentence(item.reason_text)}")

        return "\n".join(lines)

    def _merge_duplicate_items(
        self,
        raw_items: list[RejectionReasonItem],
    ) -> tuple[list[RejectionReasonItem], list[dict[str, Any]]]:
        merged: dict[str, RejectionReasonItem] = {}
        debug: list[dict[str, Any]] = []

        for item in raw_items:
            key = self._normalize_text(item.reason_text)
            existing = merged.get(key)

            if existing is None:
                merged[key] = item
                continue

            existing.source_row_ids.extend(
                row_id for row_id in item.source_row_ids if row_id not in existing.source_row_ids
            )
            existing.applicant_category_ids.extend(
                value
                for value in item.applicant_category_ids
                if value not in existing.applicant_category_ids
            )
            existing.source_scores.extend(item.source_scores)

            if existing.row_order is None and item.row_order is not None:
                existing.row_order = item.row_order

            debug.append(
                {
                    "merged_into": existing.reason_text,
                    "merged_reason_text": item.reason_text,
                    "source_row_ids": item.source_row_ids,
                }
            )

        return list(merged.values()), debug

    def _detect_question_scope(self, question_text: str) -> str:
        text = self._normalize_text(question_text)

        if not text:
            return "service_refusal"

        if "приостанов" in text:
            return "suspension"

        if "отказ" in text and "возобнов" in text:
            return "renewal_refusal"

        if (
            "отказа в приеме" in text
            or "отказ в приеме" in text
            or "приеме документов" in text
            or "прием документов" in text
            or "не принять документы" in text
            or "не примут документы" in text
            or "не принимают документы" in text
            or "не приняли документы" in text
            or "откажут принять документы" in text
            or "заявление не примут" in text
            or "не принять заявление" in text
            or "не примут заявление" in text
        ):
            return "intake_refusal"

        if "отказ" in text or "почему могут отказать" in text:
            return "service_refusal"

        return "other"

    def _classify_reason_scope(self, reason_text: str) -> str:
        text = self._normalize_text(reason_text)

        if "приостанов" in text:
            return "suspension"

        if "отказа в возобновлении" in text or "отказ в возобновлении" in text:
            return "renewal_refusal"

        if "отказа в приеме" in text or "отказ в приеме" in text:
            return "intake_refusal"

        if any(
            marker in text
            for marker in (
                "отказа в предоставлении",
                "отказ в предоставлении",
                "отказа в назначении",
                "отказ в назначении",
                "отказа в предоставлении государственной услуги",
            )
        ):
            return "service_refusal"

        return "other"

    def _is_service_value(self, value: str) -> bool:
        return self._normalize_text(value) in self._SERVICE_VALUES

    def _clean(self, value: Any) -> str:
        if value is None:
            return ""
        text = str(value).replace("\xa0", " ").strip()
        return " ".join(text.split())

    def _normalize_text(self, value: Any) -> str:
        return self._clean(value).lower()

    def _normalize_sentence(self, value: str) -> str:
        text = self._clean(value)
        if not text:
            return text
        if text[-1] not in ".!?":
            text += "."
        return text