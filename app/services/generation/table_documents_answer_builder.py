from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(slots=True)
class DocumentsAnswerItem:
    document_name: str
    role: str
    applicability: str
    document_family: str
    submission_note: Optional[str] = None
    source_row_ids: list[str] = field(default_factory=list)
    applicant_category_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DocumentsAnswerBuildResult:
    can_answer: bool
    base_items: list[DocumentsAnswerItem] = field(default_factory=list)
    conditional_items: list[DocumentsAnswerItem] = field(default_factory=list)
    representative_items: list[DocumentsAnswerItem] = field(default_factory=list)
    category_specific_items: list[DocumentsAnswerItem] = field(default_factory=list)
    dropped_rows_debug: list[dict[str, Any]] = field(default_factory=list)
    merged_items_debug: list[dict[str, Any]] = field(default_factory=list)
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
                row_id
                for item in self.all_items
                for row_id in item.source_row_ids
            ],
            "items": [
                {
                    "document_name": item.document_name,
                    "role": item.role,
                    "applicability": item.applicability,
                    "document_family": item.document_family,
                    "submission_note": item.submission_note,
                    "source_row_ids": item.source_row_ids,
                    "applicant_category_ids": item.applicant_category_ids,
                }
                for item in self.all_items
            ],
            "merged_items": self.merged_items_debug,
            "dropped_rows": self.dropped_rows_debug,
            "reason": self.reason,
        }


class TableDocumentsAnswerBuilder:
    """
    Deterministic builder for document-list questions based on table rows.

    Principles:
    - works only with retrieval-selected candidates
    - does not query DB directly
    - separates document role from applicability
    - merges semantically close rows by generic document families
    """

    def build(
        self,
        *,
        candidates: list[Any],
        submission_channel: Optional[str],
    ) -> DocumentsAnswerBuildResult:
        raw_items: list[DocumentsAnswerItem] = []
        dropped_rows_debug: list[dict[str, Any]] = []

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
            document_family = self._infer_document_family(document_name)
            role = self._classify_document_role(
                document_name=document_name,
                document_family=document_family,
            )
            applicability = self._infer_applicability(
                document_name=document_name,
                applicant_category_id=applicant_category_id,
                document_family=document_family,
            )

            raw_items.append(
                DocumentsAnswerItem(
                    document_name=document_name,
                    role=role,
                    applicability=applicability,
                    document_family=document_family,
                    submission_note=submission_note,
                    source_row_ids=[row_id] if row_id else [],
                    applicant_category_ids=[applicant_category_id] if applicant_category_id else [],
                )
            )

        if not raw_items:
            return DocumentsAnswerBuildResult(
                can_answer=False,
                reason="no_documents_rows",
                dropped_rows_debug=dropped_rows_debug,
            )

        merged_items, merged_items_debug = self._merge_similar_items(raw_items)

        base_items: list[DocumentsAnswerItem] = []
        conditional_items: list[DocumentsAnswerItem] = []
        representative_items: list[DocumentsAnswerItem] = []
        category_specific_items: list[DocumentsAnswerItem] = []

        for item in merged_items:
            if item.role == "representative_only":
                representative_items.append(item)
                continue

            if item.applicability == "category_specific":
                category_specific_items.append(item)
                continue

            if item.applicability == "always":
                base_items.append(item)
            else:
                conditional_items.append(item)

        return DocumentsAnswerBuildResult(
            can_answer=True,
            base_items=base_items,
            conditional_items=conditional_items,
            representative_items=representative_items,
            category_specific_items=category_specific_items,
            dropped_rows_debug=dropped_rows_debug,
            merged_items_debug=merged_items_debug,
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
        lines.append("Итоговый перечень зависит от конкретной жизненной ситуации, основания обращения и категории заявителя.")

        return "\n".join(lines)
        
    def _normalize_display_name(
        self,
        *,
        document_name: str,
        document_family: str,
    ) -> str:
        text = self._clean(document_name) or ""

        # Универсальная нормализация identity-документов
        if document_family == "identity_document":
            return "Паспорт или иной документ, удостоверяющий личность"

        # Убираем служебные хвосты в скобках (представителя и т.п.)
        if "(" in text and ")" in text:
            # аккуратно убираем только последние скобки
            base = text.split("(")[0].strip()
            if base:
                return base

        return text

    def _merge_similar_items(
        self,
        items: list[DocumentsAnswerItem],
    ) -> tuple[list[DocumentsAnswerItem], list[dict[str, Any]]]:
        groups: dict[tuple[str, str, str], list[DocumentsAnswerItem]] = {}

        for item in items:
            key = (item.document_family, item.role, item.applicability)
            groups.setdefault(key, []).append(item)

        merged_items: list[DocumentsAnswerItem] = []
        merged_items_debug: list[dict[str, Any]] = []

        for (document_family, role, applicability), group_items in groups.items():
            merged_name = self._choose_merged_display_name(
                document_family=document_family,
                group_items=group_items,
            )
            merged_name = self._normalize_display_name(
                document_name=merged_name,
                document_family=document_family,
            )
            merged_submission_note = self._merge_submission_notes(group_items)

            row_ids: list[str] = []
            category_ids: list[str] = []
            for item in group_items:
                for row_id in item.source_row_ids:
                    if row_id and row_id not in row_ids:
                        row_ids.append(row_id)
                for category_id in item.applicant_category_ids:
                    if category_id and category_id not in category_ids:
                        category_ids.append(category_id)

            merged_items.append(
                DocumentsAnswerItem(
                    document_name=merged_name,
                    role=role,
                    applicability=applicability,
                    document_family=document_family,
                    submission_note=merged_submission_note,
                    source_row_ids=row_ids,
                    applicant_category_ids=category_ids,
                )
            )

            if len(group_items) > 1:
                merged_items_debug.append(
                    {
                        "document_family": document_family,
                        "role": role,
                        "applicability": applicability,
                        "source_document_names": [item.document_name for item in group_items],
                        "merged_document_name": merged_name,
                        "source_row_ids": row_ids,
                    }
                )

        merged_items.sort(
            key=lambda item: (
                self._role_order(item.role),
                self._applicability_order(item.applicability),
                item.document_name.lower(),
            )
        )

        return merged_items, merged_items_debug

    def _choose_merged_display_name(
        self,
        *,
        document_family: str,
        group_items: list[DocumentsAnswerItem],
    ) -> str:
        if document_family == "application_request":
            has_multiple_variants = len(group_items) > 1
            if has_multiple_variants:
                return "Заявление (в зависимости от основания обращения)"
            return group_items[0].document_name

        priority_names = sorted(
            (item.document_name for item in group_items),
            key=lambda value: (len(value), value.lower()),
        )
        return priority_names[0]

    def _merge_submission_notes(
        self,
        items: list[DocumentsAnswerItem],
    ) -> Optional[str]:
        notes: list[str] = []
        for item in items:
            note = self._clean(item.submission_note)
            if note and note not in notes:
                notes.append(note)

        if not notes:
            return None
        if len(notes) == 1:
            return notes[0]
        return "; ".join(notes)

    def _classify_document_role(
        self,
        *,
        document_name: str,
        document_family: str,
    ) -> str:
        text = self._normalize(document_name)

        # Сначала решаем по семейству документа, а не по случайным словам в тексте.
        if document_family == "authority_document":
            return "representative_only"

        if document_family in {
            "identity_document",
            "application_request",
            "residency_proof",
            "status_certificate",
            "court_decision",
            "employment_proof",
            "other",
        }:
            return "general_document"

        # Страховка на случай, если family не распознался, но текст явно про представителя.
        if any(marker in text for marker in [
            "полномоч",
            "доверенн",
            "представител",
        ]):
            return "representative_only"

        return "general_document"

    def _infer_applicability(
        self,
        *,
        document_name: str,
        applicant_category_id: Optional[str],
        document_family: str,
    ) -> str:
        text = self._normalize(document_name)

        # Явные условные маркеры всегда сильнее всего остального.
        if any(marker in text for marker in [
            "в случае",
            "при отсутствии",
            "при наличии",
            "при обращении",
            "подтверждающ",
            "решение суда",
            "регистрац",
            "проживани",
            "смен",
            "перемен",
            "смерт",
            "усынов",
            "опек",
            "попеч",
            "брак",
            "развод",
            "рождени",
        ]):
            return "conditional"

        # Базовые семейства не должны автоматически улетать в category_specific
        # только из-за applicant_category_id.
        if document_family in {
            "identity_document",
            "application_request",
        }:
            return "always"

        if applicant_category_id:
            return "category_specific"

        return "always"

    def _infer_document_family(self, document_name: str) -> str:
        text = self._normalize(document_name)

        if any(marker in text for marker in [
            "заявление",
            "запрос",
            "ходатайств",
        ]):
            return "application_request"

        if any(marker in text for marker in [
            "паспорт",
            "удостоверяющ личность",
            "иной документ удостоверяющий личность",
        ]):
            return "identity_document"

        if any(marker in text for marker in [
            "полномоч",
            "доверенн",
            "представител",
        ]):
            return "authority_document"

        if any(marker in text for marker in [
            "регистрац",
            "проживани",
            "место жительства",
            "место пребывания",
        ]):
            return "residency_proof"

        if any(marker in text for marker in [
            "решение суда",
        ]):
            return "court_decision"

        if any(marker in text for marker in [
            "справк",
            "удостоверени",
            "свидетельств",
            "подтверждающ статус",
            "категори",
        ]):
            return "status_certificate"

        if any(marker in text for marker in [
            "трудов",
            "работ",
            "служб",
            "занятост",
            "доход",
        ]):
            return "employment_proof"

        return "other"

    def _role_order(self, role: str) -> int:
        order = {
            "general_document": 0,
            "representative_only": 1,
        }
        return order.get(role, 99)

    def _applicability_order(self, applicability: str) -> int:
        order = {
            "always": 0,
            "conditional": 1,
            "category_specific": 2,
        }
        return order.get(applicability, 99)

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
        service_values = {
            "наименование документа",
            "наименование документов",
            "документы",
            "документы информация необходимые",
            "исчерпывающий перечень документов",
            "способ подачи в уполномоченное учреждение",
        }
        return text in service_values

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