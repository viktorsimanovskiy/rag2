from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(slots=True)
class DocumentsAnswerItem:
    document_name: str
    role: str
    submission_note: Optional[str] = None
    source_row_id: Optional[str] = None
    applicant_category_id: Optional[str] = None


@dataclass(slots=True)
class DocumentsAnswerBuildResult:
    can_answer: bool
    base_items: list[DocumentsAnswerItem] = field(default_factory=list)
    conditional_items: list[DocumentsAnswerItem] = field(default_factory=list)
    representative_items: list[DocumentsAnswerItem] = field(default_factory=list)
    category_specific_items: list[DocumentsAnswerItem] = field(default_factory=list)
    dropped_rows_debug: list[dict[str, Any]] = field(default_factory=list)
    reason: Optional[str] = None

    @property
    def all_items(self) -> list[DocumentsAnswerItem]:
        return [
            *self.base_items,
            *self.conditional_items,
            *self.representative_items,
            *self.category_specific_items,
        ]

    def debug_payload(self, *, submission_channel: Optional[str]) -> dict[str, Any]:
        return {
            "submission_channel": submission_channel,
            "base_items_count": len(self.base_items),
            "conditional_items_count": len(self.conditional_items),
            "representative_items_count": len(self.representative_items),
            "category_specific_items_count": len(self.category_specific_items),
            "input_row_ids": [
                item.source_row_id
                for item in self.all_items
                if item.source_row_id
            ],
            "dropped_rows": self.dropped_rows_debug,
            "reason": self.reason,
        }


class TableDocumentsAnswerBuilder:
    """
    Deterministic builder for document-list questions based on table rows.

    Principles:
    - works only with retrieval-selected candidates
    - does not query DB directly
    - classifies rows into answer roles
    - keeps channel-specific submission notes outside retrieval logic
    """

    def build(
        self,
        *,
        candidates: list[Any],
        submission_channel: Optional[str],
    ) -> DocumentsAnswerBuildResult:
        base_items: list[DocumentsAnswerItem] = []
        conditional_items: list[DocumentsAnswerItem] = []
        representative_items: list[DocumentsAnswerItem] = []
        category_specific_items: list[DocumentsAnswerItem] = []
        dropped_rows_debug: list[dict[str, Any]] = []

        seen_by_key: dict[str, DocumentsAnswerItem] = {}

        for candidate in candidates:
            if getattr(candidate, "source_type", None) != "table_row":
                continue

            row_id = str(getattr(candidate, "source_id", "") or "")
            metadata = getattr(candidate, "metadata_json", None) or {}
            if str(metadata.get("table_semantic_type") or "").strip().lower() != "documents":
                dropped_rows_debug.append(
                    {
                        "row_id": row_id,
                        "reason": "not_documents_table",
                    }
                )
                continue

            cells = metadata.get("cells_by_semantic_key") or metadata.get("cells_by_header_key") or {}
            if not isinstance(cells, dict):
                dropped_rows_debug.append(
                    {
                        "row_id": row_id,
                        "reason": "cells_not_dict",
                    }
                )
                continue

            document_name = self._clean(cells.get("document_name"))
            if not document_name:
                dropped_rows_debug.append(
                    {
                        "row_id": row_id,
                        "reason": "empty_document_name",
                    }
                )
                continue

            if self._is_service_value(document_name):
                dropped_rows_debug.append(
                    {
                        "row_id": row_id,
                        "reason": "service_header_row",
                        "document_name": document_name,
                    }
                )
                continue

            applicant_category_id = self._clean(cells.get("applicant_category_id"))
            submission_note = self._extract_submission_note(
                cells=cells,
                submission_channel=submission_channel,
            )

            role = self._classify_item_role(
                document_name=document_name,
                cells=cells,
                applicant_category_id=applicant_category_id,
            )

            canonical_key = self._canonical_document_key(document_name)

            if canonical_key in seen_by_key:
                existing = seen_by_key[canonical_key]
                if not existing.submission_note and submission_note:
                    existing.submission_note = submission_note
                if not existing.applicant_category_id and applicant_category_id:
                    existing.applicant_category_id = applicant_category_id
                continue

            item = DocumentsAnswerItem(
                document_name=document_name,
                role=role,
                submission_note=submission_note,
                source_row_id=row_id or None,
                applicant_category_id=applicant_category_id,
            )
            seen_by_key[canonical_key] = item

            if role == "base_required":
                base_items.append(item)
            elif role == "representative_only":
                representative_items.append(item)
            elif role == "category_specific":
                category_specific_items.append(item)
            else:
                conditional_items.append(item)

        if not (base_items or conditional_items or representative_items or category_specific_items):
            return DocumentsAnswerBuildResult(
                can_answer=False,
                reason="no_documents_rows",
                dropped_rows_debug=dropped_rows_debug,
            )

        return DocumentsAnswerBuildResult(
            can_answer=True,
            base_items=base_items,
            conditional_items=conditional_items,
            representative_items=representative_items,
            category_specific_items=category_specific_items,
            dropped_rows_debug=dropped_rows_debug,
            reason=None,
        )

    def render_text(
        self,
        *,
        result: DocumentsAnswerBuildResult,
        submission_channel: Optional[str],
    ) -> Optional[str]:
        if not result.can_answer:
            return None

        has_channel = bool(submission_channel)
        channel_label = self._channel_label(submission_channel) if submission_channel else None

        if has_channel:
            lines = [
                f"Для предоставления услуги при подаче {channel_label} обычно требуются следующие документы:"
            ]
        else:
            lines = [
                "Для предоставления услуги обычно требуются следующие документы:"
            ]

        counter = 1
        for item in result.base_items:
            lines.append(self._render_numbered_item(counter, item, has_channel=has_channel))
            counter += 1

        if result.conditional_items:
            lines.append("")
            lines.append("Дополнительно в отдельных случаях могут потребоваться:")
            for item in result.conditional_items:
                lines.append(self._render_bulleted_item(item, has_channel=has_channel))

        if result.representative_items:
            lines.append("")
            lines.append("Если документы подаёт представитель, также потребуется:")
            for item in result.representative_items:
                lines.append(self._render_bulleted_item(item, has_channel=has_channel))

        if result.category_specific_items:
            lines.append("")
            lines.append("Для отдельных категорий заявителей могут потребоваться дополнительные документы:")
            for item in result.category_specific_items:
                lines.append(self._render_bulleted_item(item, has_channel=has_channel))

        lines.append("")
        lines.append("Итоговый перечень зависит от конкретной жизненной ситуации заявителя и оснований обращения.")

        return "\n".join(lines)

    def _render_numbered_item(
        self,
        idx: int,
        item: DocumentsAnswerItem,
        *,
        has_channel: bool,
    ) -> str:
        if has_channel and item.submission_note:
            return f"{idx}. {item.document_name} — {item.submission_note}"
        return f"{idx}. {item.document_name}"

    def _render_bulleted_item(
        self,
        item: DocumentsAnswerItem,
        *,
        has_channel: bool,
    ) -> str:
        if has_channel and item.submission_note:
            return f"— {item.document_name} — {item.submission_note}"
        return f"— {item.document_name}"

    def _classify_item_role(
        self,
        *,
        document_name: str,
        cells: dict[str, Any],
        applicant_category_id: Optional[str],
    ) -> str:
        text = self._normalize(document_name)

        if any(marker in text for marker in [
            "представител",
            "доверенн",
            "полномоч",
            "удостоверяющ полномоч",
        ]):
            return "representative_only"

        if applicant_category_id:
            return "category_specific"

        if any(marker in text for marker in [
            "в случае",
            "при отсутствии",
            "подтверждающ",
            "решение суда",
            "регистрац",
            "проживани",
            "перемен",
            "смен",
            "смерт",
            "усынов",
            "опек",
            "попеч",
            "брак",
            "развод",
            "рождени",
        ]):
            return "conditional_required"

        if any(marker in text for marker in [
            "заявление",
            "паспорт",
            "документ, удостоверяющий личность",
            "иной документ, удостоверяющий личность",
        ]):
            return "base_required"

        # По умолчанию документ лучше считать условным, чем базовым.
        return "conditional_required"

    def _extract_submission_note(
        self,
        *,
        cells: dict[str, Any],
        submission_channel: Optional[str],
    ) -> Optional[str]:
        if submission_channel == "epgu":
            return self._clean(cells.get("epgu_submission"))
        if submission_channel == "regional_portal":
            return self._clean(
                cells.get("regional_portal_submission") or cells.get("epgu_submission")
            )
        if submission_channel == "in_person":
            return self._clean(cells.get("in_person_submission"))
        if submission_channel == "post":
            return self._clean(cells.get("post_submission"))
        if submission_channel == "mfc":
            return self._clean(cells.get("mfc_submission"))
        return None

    def _is_service_value(self, value: str) -> bool:
        text = self._normalize(value)
        markers = [
            "документы информация необходимые",
            "исчерпывающий перечень документов",
            "способ подачи в уполномоченное учреждение",
            "наименование документа",
            "наименование документов",
        ]
        return any(marker in text for marker in markers)

    def _canonical_document_key(self, value: str) -> str:
        text = self._normalize(value)
        replacements = {
            "документ удостоверяющий личность": "удостоверение личности",
            "иной документ удостоверяющий личность": "удостоверение личности",
            "документ удостоверяющий личность заявителя": "удостоверение личности",
        }
        return replacements.get(text, text)

    def _channel_label(self, submission_channel: Optional[str]) -> str:
        mapping = {
            "epgu": "через ЕПГУ",
            "regional_portal": "через РПГУ / краевой портал",
            "in_person": "лично",
            "post": "почтовым отправлением",
            "mfc": "через МФЦ",
        }
        return mapping.get(submission_channel or "", "указанным способом")

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