from __future__ import annotations

import re
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

    row_order: Optional[int] = None
    table_number: Optional[str] = None
    requirement_group: str = "unknown"
    requirement_group_label: Optional[str] = None

    requested_channel_key: Optional[str] = None
    requested_channel_label: Optional[str] = None
    requested_channel_value: Optional[str] = None
    generic_submission_value: Optional[str] = None
    channel_values: dict[str, str] = field(default_factory=dict)

    row_number: Optional[str] = None
    parent_row_number: Optional[str] = None
    hierarchy_level: int = 0
    row_kind: str = "document"
    has_children: bool = False
    is_bez_statement: bool = False

    is_exact_row: bool = False


@dataclass(slots=True)
class DocumentsAnswerBuildResult:
    can_answer: bool
    base_items: list[DocumentsAnswerItem] = field(default_factory=list)
    conditional_items: list[DocumentsAnswerItem] = field(default_factory=list)
    representative_items: list[DocumentsAnswerItem] = field(default_factory=list)
    category_specific_items: list[DocumentsAnswerItem] = field(default_factory=list)
    full_list_items: list[DocumentsAnswerItem] = field(default_factory=list)
    dropped_rows_debug: list[dict[str, Any]] = field(default_factory=list)
    merged_items_debug: list[dict[str, Any]] = field(default_factory=list)
    reason: Optional[str] = None
    full_list_mode: bool = False

    @property
    def all_items(self) -> list[DocumentsAnswerItem]:
        if self.full_list_mode and self.full_list_items:
            return self.full_list_items
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
            "full_list_items_count": len(self.full_list_items),
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
                    "row_number": item.row_number,
                    "parent_row_number": item.parent_row_number,
                    "hierarchy_level": item.hierarchy_level,
                    "row_kind": item.row_kind,
                    "is_bez_statement": item.is_bez_statement,
                }
                for item in self.all_items
            ],
            "merged_items": self.merged_items_debug,
            "dropped_rows": self.dropped_rows_debug,
            "reason": self.reason,
            "full_list_mode": self.full_list_mode,
        }


class TableDocumentsAnswerBuilder:
    """
    Deterministic builder for document-list questions based on table rows.

    In short-list mode it keeps the older behaviour: it uses retrieval-selected
    rows and merges similar documents into a compact answer.

    In full-list mode it preserves the table structure from the regulation:
    - original row numbers are shown instead of generated numbering;
    - groups/subgroups are kept as headings;
    - channels are shown only for concrete document rows;
    - беззаявительные blocks are separated from the applicant's document package.
    """

    _ROW_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)*$")

    def build(
        self,
        *,
        candidates: list[Any],
        submission_channel: Optional[str],
        full_list_mode: bool = False,
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

            cells = self._collect_document_cells(metadata)
            if not cells:
                dropped_rows_debug.append(
                    {
                        "row_id": row_id,
                        "reason": "cells_not_dict_or_empty",
                    }
                )
                continue

            row_order = getattr(candidate, "row_order", None) or metadata.get("row_order")

            if full_list_mode:
                exact_item = self._build_exact_row_item(
                    row_id=row_id,
                    row_order=row_order,
                    metadata=metadata,
                    cells=cells,
                    submission_channel=submission_channel,
                )
                if exact_item is None:
                    dropped_rows_debug.append(
                        {
                            "row_id": row_id,
                            "reason": "not_exact_documents_row",
                        }
                    )
                    continue
                raw_items.append(exact_item)
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

            if self._is_noise_or_truncated_document_name(document_name):
                dropped_rows_debug.append(
                    {
                        "row_id": row_id,
                        "reason": "noise_or_truncated_document_name",
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
                full_list_mode=full_list_mode,
            )

        if full_list_mode:
            full_list_items = self._prepare_full_list_items(raw_items)
            if not full_list_items:
                return DocumentsAnswerBuildResult(
                    can_answer=False,
                    reason="no_exact_documents_rows_after_structure_filter",
                    dropped_rows_debug=dropped_rows_debug,
                    full_list_mode=True,
                )
            return DocumentsAnswerBuildResult(
                can_answer=True,
                full_list_items=full_list_items,
                dropped_rows_debug=dropped_rows_debug,
                merged_items_debug=[],
                reason=None,
                full_list_mode=True,
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
            full_list_mode=full_list_mode,
        )

    def render_text(
        self,
        *,
        result: DocumentsAnswerBuildResult,
        submission_channel: Optional[str],
        include_submission_form_details: bool = False,
        compact_full_list: bool = False,
    ) -> Optional[str]:
        if not result.can_answer:
            return None

        if result.full_list_mode:
            if compact_full_list:
                return self._render_compact_full_list_text(
                    result=result,
                    submission_channel=submission_channel,
                    include_submission_form_details=include_submission_form_details,
                )
            return self._render_full_list_text(
                result=result,
                submission_channel=submission_channel,
                include_submission_form_details=include_submission_form_details,
            )

        has_channel = bool(submission_channel and include_submission_form_details)
        channel_label = self._channel_label(submission_channel) if has_channel else None

        lines: list[str] = []

        if result.base_items:
            if has_channel:
                lines.append(
                    f"Для предоставления услуги при подаче {channel_label} обычно требуются следующие документы:"
                )
            else:
                lines.append("Для предоставления услуги обычно требуются следующие документы:")

            counter = 1
            for item in result.base_items:
                lines.append(self._render_numbered_item(counter, item, has_channel=has_channel))
                counter += 1
        else:
            lines.append(
                "По найденным строкам перечня нет отдельного документа, "
                "который требуется всем заявителям безусловно."
            )

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
            visible_items = result.category_specific_items[:5]
            for item in visible_items:
                lines.append(self._render_bulleted_item(item, has_channel=has_channel))
            hidden_count = len(result.category_specific_items) - len(visible_items)
            if hidden_count > 0:
                lines.append(f"— ещё {hidden_count} позиций зависят от конкретной категории заявителя.")

        lines.append("")
        lines.append("Итоговый перечень зависит от конкретной жизненной ситуации, основания обращения и категории заявителя.")

        return "\n".join(lines)

    def render_representative_text(
        self,
        *,
        result: DocumentsAnswerBuildResult,
        submission_channel: Optional[str],
        include_submission_form_details: bool = False,
    ) -> Optional[str]:
        if not result.can_answer:
            return None

        has_channel = bool(submission_channel and include_submission_form_details)
        representative_items = result.representative_items

        lines: list[str] = [
            "В найденном перечне документов предусмотрена подача через представителя.",
        ]

        if representative_items:
            lines.append("Если документы подаёт представитель, нужен документ, подтверждающий его полномочия:")
            for item in representative_items[:3]:
                lines.append(self._render_bulleted_item(item, has_channel=has_channel))
        else:
            lines.append(
                "В выбранных строках перечня не найден отдельный документ о полномочиях представителя. "
                "Нужно проверить конкретный пункт регламента по этой услуге."
            )

        lines.append(
            "Обычно основной комплект документов заявителя при этом сохраняется; "
            "отдельно добавляется подтверждение полномочий представителя."
        )
        return "\n".join(lines)

    def _normalize_display_name(
        self,
        *,
        document_name: str,
        document_family: str,
    ) -> str:
        text = self._clean(document_name) or ""

        if document_family == "identity_document":
            return "Паспорт или иной документ, удостоверяющий личность"

        if "(" in text and ")" in text:
            base = text.split("(")[0].strip()
            if base:
                return base

        return text

    def _resolve_requested_channel_key(
        self,
        submission_channel: Optional[str],
    ) -> Optional[str]:
        mapping = {
            "epgu": "epgu_submission",
            "regional_portal": "regional_portal_submission",
            "in_person": "in_person_submission",
            "post": "post_submission",
            "mfc": "mfc_submission",
        }
        return mapping.get(submission_channel or "")

    def _build_exact_row_item(
        self,
        *,
        row_id: str,
        row_order: Optional[int],
        metadata: dict[str, Any],
        cells: dict[str, Any],
        submission_channel: Optional[str],
    ) -> Optional[DocumentsAnswerItem]:
        document_name = self._clean(cells.get("document_name"))
        row_number = self._extract_row_number(cells=cells, metadata=metadata)

        # Section/header rows often have no semantic document_name, or repeat a section title
        # across all cells. We keep real group rows with a row number, but drop table headers
        # and editorial notes from user-facing output.
        if not document_name:
            return None
        if self._is_amendment_note(row_number or document_name, document_name):
            return None
        if self._is_service_value(document_name) and not self._is_number_like(row_number):
            return None
        if self._is_noise_or_truncated_document_name(document_name):
            return None

        applicant_category_id = self._clean(cells.get("applicant_category_id"))
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

        requested_channel_key = self._resolve_requested_channel_key(submission_channel)
        channel_values = self._extract_channel_values(cells)
        requested_channel_value = self._extract_submission_note(
            cells=cells,
            submission_channel=submission_channel,
        )
        generic_submission_value = self._extract_generic_submission_note(
            cells=cells,
            requested_channel_key=requested_channel_key,
        )
        requested_channel_label = (
            self._channel_label(submission_channel) if submission_channel else None
        )

        return DocumentsAnswerItem(
            document_name=document_name,
            role=role,
            applicability=applicability,
            document_family=document_family,
            submission_note=requested_channel_value,
            source_row_ids=[row_id] if row_id else [],
            applicant_category_ids=[applicant_category_id] if applicant_category_id else [],
            row_order=row_order,
            table_number=self._clean(metadata.get("table_number")),
            requirement_group=self._clean(metadata.get("requirement_group")) or "unknown",
            requirement_group_label=self._clean(metadata.get("requirement_group_label")),
            requested_channel_key=requested_channel_key,
            requested_channel_label=requested_channel_label,
            requested_channel_value=requested_channel_value,
            generic_submission_value=generic_submission_value,
            channel_values=channel_values,
            row_number=row_number,
            is_exact_row=True,
        )

    def _prepare_full_list_items(
        self,
        items: list[DocumentsAnswerItem],
    ) -> list[DocumentsAnswerItem]:
        prepared = sorted(
            items,
            key=lambda item: (
                item.row_order if item.row_order is not None else 10**9,
                item.row_number or "",
            ),
        )

        numbered_items = [item for item in prepared if self._is_number_like(item.row_number)]
        numbers = {item.row_number for item in numbered_items if item.row_number}

        for item in prepared:
            if self._is_number_like(item.row_number):
                item.hierarchy_level = len(str(item.row_number).split("."))
                item.parent_row_number = self._parent_row_number(item.row_number)
                item.has_children = any(
                    other_number != item.row_number and str(other_number).startswith(f"{item.row_number}.")
                    for other_number in numbers
                )
            else:
                item.hierarchy_level = 0
                item.parent_row_number = None
                item.has_children = False

        bez_roots: set[str] = set()
        for item in prepared:
            if item.row_number and self._looks_like_bez_statement_row(item):
                bez_roots.add(item.row_number)

        for item in prepared:
            item.is_bez_statement = any(
                item.row_number == root or str(item.row_number or "").startswith(f"{root}.")
                for root in bez_roots
            )
            item.row_kind = self._classify_full_list_row(item)

        return [
            item
            for item in prepared
            if item.row_kind not in {"amendment_note", "section_header", "skip"}
        ]

    def _classify_full_list_row(self, item: DocumentsAnswerItem) -> str:
        if self._is_amendment_note(item.row_number or "", item.document_name):
            return "amendment_note"

        if not self._is_number_like(item.row_number):
            if self._is_section_heading_text(item.document_name):
                return "section_header"
            return "skip"

        if item.is_bez_statement:
            if (item.has_children or self._is_group_like_document_name(item.document_name)) and not self._has_any_submission_value(item):
                return "bez_statement_group"
            return "bez_statement_document"

        if item.has_children and not self._has_any_submission_value(item):
            return "group"

        if self._is_group_like_document_name(item.document_name) and not self._has_any_submission_value(item):
            return "group"

        return "document"

    def _render_compact_full_list_text(
        self,
        *,
        result: DocumentsAnswerBuildResult,
        submission_channel: Optional[str],
        include_submission_form_details: bool = False,
    ) -> str:
        """Render a short human document package.

        This is the default answer for ordinary questions like
        "список документов" / "какие документы нужны". It must not reproduce
        the whole regulation table. Full row-by-row output is available only
        through explicit "полный перечень" questions; form/original/copy
        details are available only through explicit form questions.
        """
        items = result.full_list_items or result.all_items

        required_items = [
            item for item in items
            if item.requirement_group == "required" and not item.is_bez_statement
        ]
        optional_items = [
            item for item in items
            if item.requirement_group == "optional" and not item.is_bez_statement
        ]
        other_items = [
            item for item in items
            if item.requirement_group not in {"required", "optional"} and not item.is_bez_statement
        ]
        bez_items = [item for item in items if item.is_bez_statement]

        lines: list[str] = ["Основной комплект документов:"]
        if submission_channel:
            if include_submission_form_details:
                lines.append(
                    f"Канал подачи: {self._channel_label(submission_channel)}. "
                    "Показываю форму подачи только для основных документов."
                )
            else:
                lines.append(
                    f"Канал подачи: {self._channel_label(submission_channel)}. "
                    "Ниже только список документов; оригиналы, копии и электронные образы — по отдельному вопросу."
                )

        required_lines, required_hidden = self._render_minimal_compact_group(
            required_items,
            submission_channel=submission_channel,
            include_submission_form_details=include_submission_form_details,
        )
        if required_lines:
            lines.append("")
            lines.extend(required_lines)

        optional_summary = self._render_optional_compact_summary(optional_items)
        if optional_summary:
            lines.append("")
            lines.append(optional_summary)

        # In the compact user answer we intentionally do not list miscellaneous
        # non-required rows.  They often duplicate optional/initiative rows and
        # make the answer look like a regulation dump.  Full details remain
        # available via explicit "полный перечень документов" questions.

        if bez_items:
            lines.append("")
            lines.append(
                "Отдельно: в регламенте есть беззаявительный порядок. "
                "Это не документы, которые заявитель подаёт через МФЦ/ЕПГУ; "
                "сведения представляет уполномоченный орган."
            )

        if required_hidden > 0:
            lines.append("")
            lines.append(
                "Детальные подпункты не раскрыты. Если нужен список строго по строкам регламента, "
                "спросите: «полный перечень документов …»."
            )

        return "\n".join(lines).strip()

    def _render_minimal_compact_group(
        self,
        items: list[DocumentsAnswerItem],
        *,
        limit_top_level: int = 8,
        child_summary_limit: int = 4,
        submission_channel: Optional[str] = None,
        include_submission_form_details: bool = False,
    ) -> tuple[list[str], int]:
        visible_lines: list[str] = []
        hidden_count = 0

        item_numbers = {item.row_number for item in items if item.row_number}
        by_parent: dict[str, list[DocumentsAnswerItem]] = {}
        for item in items:
            if item.parent_row_number:
                by_parent.setdefault(item.parent_row_number, []).append(item)

        root_items = [
            item for item in items
            if not item.parent_row_number or item.parent_row_number not in item_numbers
        ]
        root_items = sorted(
            root_items,
            key=lambda item: (
                item.row_order if item.row_order is not None else 10**9,
                self._row_number_sort_key(item.row_number),
            ),
        )

        emitted = 0
        emitted_signatures: set[str] = set()

        for item in root_items:
            if not self._should_show_in_compact_list(item):
                hidden_count += 1
                continue

            signature = self._compact_signature(item.document_name)
            if signature and signature in emitted_signatures:
                hidden_count += 1
                continue

            if emitted >= limit_top_level:
                hidden_count += 1
                continue

            children = sorted(
                [
                    child for child in by_parent.get(item.row_number or "", [])
                    if self._should_show_in_compact_list(child)
                ],
                key=lambda child: (
                    child.row_order if child.row_order is not None else 10**9,
                    self._row_number_sort_key(child.row_number),
                ),
            )

            rendered = self._render_minimal_compact_item(
                item=item,
                children=children,
                child_summary_limit=child_summary_limit,
                submission_channel=submission_channel,
                include_submission_form_details=include_submission_form_details,
            )
            if rendered:
                visible_lines.append(rendered)
                emitted += 1
                if signature:
                    emitted_signatures.add(signature)

            shown_child_lines = 0
            for child in children:
                # В компактном ответе раскрываем только ближайшие подпункты.
                # Глубокие подпункты вроде 5.1.1, 5.1.2 оставляем для полного перечня.
                if (child.hierarchy_level or 1) > (item.hierarchy_level or 1) + 1:
                    hidden_count += 1
                    continue
                if shown_child_lines >= child_summary_limit:
                    hidden_count += 1
                    continue
                child_line = self._render_minimal_child_line(
                    child,
                    submission_channel=submission_channel,
                    include_submission_form_details=include_submission_form_details,
                )
                if child_line:
                    visible_lines.append(child_line)
                    shown_child_lines += 1

        return visible_lines, hidden_count

    def _render_minimal_compact_item(
        self,
        *,
        item: DocumentsAnswerItem,
        children: list[DocumentsAnswerItem],
        child_summary_limit: int,
        submission_channel: Optional[str] = None,
        include_submission_form_details: bool = False,
    ) -> Optional[str]:
        number = item.row_number or ""
        is_group = item.row_kind in {"group", "bez_statement_group"}
        text = self._humanize_compact_document_name(
            self._strip_leading_letter_marker(item.document_name),
            is_group=is_group,
        )
        if not text:
            return None

        # В компактном режиме не склеиваем группу и её подпункты в одну длинную строку.
        # Иначе пункт 5 ТЖС превращается в нечитаемую простыню.
        if is_group:
            text = text.rstrip(" .;:") + ":"
        else:
            text = text.rstrip(" .;")

        line = f"• {number}. {text}" if number else f"• {text}"
        if include_submission_form_details and not is_group:
            channel_note = self._render_compact_channel_note(
                item=item,
                submission_channel=submission_channel,
            )
            if channel_note:
                line += f" — {channel_note}"
        return line

    def _render_minimal_child_line(
        self,
        item: DocumentsAnswerItem,
        *,
        submission_channel: Optional[str] = None,
        include_submission_form_details: bool = False,
    ) -> Optional[str]:
        text = self._humanize_compact_document_name(
            self._strip_leading_letter_marker(item.document_name),
            is_group=item.row_kind in {"group", "bez_statement_group"},
        )
        if not text:
            return None
        text = text.rstrip(" .;:")
        number = item.row_number or ""
        prefix = "  —"
        line = f"{prefix} {number}. {text}" if number else f"{prefix} {text}"
        if include_submission_form_details and item.row_kind not in {"group", "bez_statement_group"}:
            channel_note = self._render_compact_channel_note(
                item=item,
                submission_channel=submission_channel,
            )
            if channel_note:
                line += f" — {channel_note}"
        return line

    def _humanize_child_summary(self, value: str) -> str:
        text = self._replace_long_legal_phrases(self._strip_leading_letter_marker(value))
        text = self._shorten_for_compact_list(text, max_len=90)
        return text.rstrip(" ,;.")

    def _render_optional_compact_summary(
        self,
        items: list[DocumentsAnswerItem],
        *,
        prefix: str = "По собственной инициативе",
    ) -> Optional[str]:
        if not items:
            return None

        # Do not enumerate initiative/optional rows in the compact answer.
        # In many regulations these rows are long interagency data items, not
        # documents the applicant must collect.  A short note is more useful
        # for a chat answer.
        return (
            f"{prefix}: можно приложить дополнительные сведения, если они есть. "
            "Остальное уполномоченный орган может запросить самостоятельно."
        )

    def _render_human_compact_group(
        self,
        items: list[DocumentsAnswerItem],
        *,
        limit_top_level: int = 12,
        limit_children_per_group: int = 6,
        suppress_duplicate_texts: Optional[set[str]] = None,
    ) -> tuple[list[str], int]:
        suppress_duplicate_texts = suppress_duplicate_texts or set()
        visible_lines: list[str] = []
        hidden_count = 0
        emitted_top = 0
        emitted_signatures: set[str] = set()

        by_parent: dict[str, list[DocumentsAnswerItem]] = {}
        for item in items:
            if item.parent_row_number:
                by_parent.setdefault(item.parent_row_number, []).append(item)

        root_items = [
            item for item in items
            if not item.parent_row_number or item.parent_row_number not in {x.row_number for x in items if x.row_number}
        ]
        root_items = sorted(
            root_items,
            key=lambda item: (
                item.row_order if item.row_order is not None else 10**9,
                self._row_number_sort_key(item.row_number),
            ),
        )

        for item in root_items:
            if not self._should_show_in_compact_list(item):
                hidden_count += 1
                continue

            signature = self._compact_signature(item.document_name)
            if signature and (signature in suppress_duplicate_texts or signature in emitted_signatures):
                hidden_count += 1
                continue

            if emitted_top >= limit_top_level:
                hidden_count += 1
                continue

            emitted_top += 1
            if signature:
                emitted_signatures.add(signature)

            rendered = self._render_human_compact_item(item, as_child=False)
            if rendered:
                visible_lines.append(rendered)

            children = [
                child for child in by_parent.get(item.row_number or "", [])
                if self._should_show_in_compact_list(child)
            ]
            children = sorted(
                children,
                key=lambda child: (
                    child.row_order if child.row_order is not None else 10**9,
                    self._row_number_sort_key(child.row_number),
                ),
            )

            # For compact answers we show only one nesting level below a group.
            # Deeper rows usually enumerate examples inside a category and make
            # the answer unreadable in chat.
            shown_children = 0
            for child in children:
                if (child.hierarchy_level or 1) > (item.hierarchy_level or 1) + 1:
                    hidden_count += 1
                    continue
                if shown_children >= limit_children_per_group:
                    hidden_count += 1
                    continue

                child_signature = self._compact_signature(child.document_name)
                if child_signature and (child_signature in suppress_duplicate_texts or child_signature in emitted_signatures):
                    hidden_count += 1
                    continue

                rendered_child = self._render_human_compact_item(child, as_child=True)
                if rendered_child:
                    visible_lines.append(rendered_child)
                    shown_children += 1
                    if child_signature:
                        emitted_signatures.add(child_signature)

            if len(children) > shown_children:
                # Do not add a noisy "ещё N" after every group; one final note is enough.
                pass

        return visible_lines, hidden_count

    def _render_human_compact_item(
        self,
        item: DocumentsAnswerItem,
        *,
        as_child: bool,
    ) -> Optional[str]:
        if item.row_kind in {"amendment_note", "section_header", "skip"}:
            return None

        number = item.row_number or ""
        text = self._humanize_compact_document_name(
            self._strip_leading_letter_marker(item.document_name),
            is_group=item.row_kind in {"group", "bez_statement_group"},
        )
        if not text:
            return None

        prefix = "  •" if as_child else "•"
        if number:
            line = f"{prefix} {number}. {text}"
        else:
            line = f"{prefix} {text}"

        if item.row_kind in {"group", "bez_statement_group"} and not line.rstrip().endswith(":"):
            line = line.rstrip(".;") + ":"
        return line

    def _should_show_in_compact_list(self, item: DocumentsAnswerItem) -> bool:
        if item.row_kind in {"amendment_note", "section_header", "skip"}:
            return False
        if item.row_kind in {"bez_statement_group", "bez_statement_document"}:
            return False
        if self._is_correction_or_appeal_document(item.document_name):
            return False
        if self._is_empty_or_service_like_group(item.document_name):
            return False
        # Do not show deep examples as separate lines in the compact answer.
        if (item.hierarchy_level or 1) >= 3:
            return False
        return True

    def _humanize_compact_document_name(self, value: str, *, is_group: bool) -> str:
        text = self._clean(value) or ""
        text = self._strip_legal_tail_for_compact(text)
        text = self._replace_long_legal_phrases(text)
        text = self._shorten_for_compact_list(text, max_len=150 if is_group else 190)
        text = text.rstrip(" ,;.")
        return text

    def _strip_legal_tail_for_compact(self, value: str) -> str:
        text = self._clean(value) or ""
        # Keep the useful condition, remove heavy legal wording after it.
        replacements = [
            (" - представляется в случае,", " — если"),
            (" — представляется в случае,", " — если"),
            (" (в случае представления документов представителем)", " — если обращается представитель"),
        ]
        for old, new in replacements:
            text = text.replace(old, new)
        return text

    def _replace_long_legal_phrases(self, value: str) -> str:
        text = self._clean(value) or ""
        lower = text.lower().replace("ё", "е")

        if "документы" in lower and "подтверждающие доход" in lower:
            return "документы о доходах заявителя и членов семьи"
        if "медицинское заключение" in lower or "справка медицинской организации" in lower:
            return "медицинское заключение или справка — если помощь нужна по медицинским основаниям"
        if "акт" in lower and "осмотра имущества" in lower:
            return "справка или акт осмотра имущества — если имущество пострадало от пожара, затопления или ЧС"
        if "справка об освобождении" in lower:
            return "справка об освобождении из мест лишения свободы"
        if "трудовая книжка" in lower or "сведения о трудовой деятельности" in lower:
            return "трудовая книжка или сведения о трудовой деятельности — если ТЖС связана с потерей работы"
        if "неспособность к самообслуживанию" in lower:
            return "документы о неспособности к самообслуживанию, одиночестве, отсутствии жилья или жестоком обращении"
        if "страхового свидетельства" in lower or "снилс" in lower:
            return "СНИЛС или иной документ о регистрации в системе индивидуального учёта"
        if "свидетельства о регистрации по месту пребывания" in lower or "свидетельство о регистрации по месту пребывания" in lower:
            return "свидетельство о регистрации по месту пребывания — если есть временная регистрация"
        if "решения суда об установлении факта проживания" in lower or "решение суда об установлении факта проживания" in lower:
            return "решение суда о факте проживания — если нет регистрации в Красноярском крае"
        if "полномоч" in lower and "представител" in lower:
            return "документ о полномочиях представителя — если обращается представитель"
        if "заявление по форме" in lower:
            return "заявление"
        if "согласие на обработку персональных данных" in lower:
            return "согласие на обработку персональных данных"
        return text.replace("если если", "если")

    def _is_correction_or_appeal_document(self, value: str) -> bool:
        text = self._normalize(value)
        return (
            "исправлен" in text and ("опечат" in text or "ошиб" in text)
        ) or "обжалован" in text

    def _is_empty_or_service_like_group(self, value: str) -> bool:
        text = self._normalize(value)
        if not text:
            return True
        if text in {"документы", "сведения", "документы сведения"}:
            return True
        return False

    def _collect_compact_signatures(self, items: list[DocumentsAnswerItem]) -> set[str]:
        result: set[str] = set()
        for item in items:
            signature = self._compact_signature(item.document_name)
            if signature:
                result.add(signature)
        return result

    def _compact_signature(self, value: str) -> str:
        text = self._normalize(value)
        if not text:
            return ""
        # Normalize broad groups so required/optional duplicate groups collapse.
        if "трудн" in text and "жизненн" in text and "ситуац" in text:
            return "documents_confirming_tjs"
        if "доход" in text:
            return "income_documents"
        if "паспорт" in text or "удостоверяющий личность" in text:
            return "identity_document"
        if "полномоч" in text and "представител" in text:
            return "representative_power"
        return text[:140]

    def _row_number_sort_key(self, value: Optional[str]) -> tuple[int, ...]:
        if not value:
            return (10**6,)
        parts: list[int] = []
        for part in str(value).split("."):
            if part.isdigit():
                parts.append(int(part))
            else:
                parts.append(10**6)
        return tuple(parts) or (10**6,)

    def _render_compact_document_item(
        self,
        item: DocumentsAnswerItem,
    ) -> Optional[str]:
        # Kept for compatibility with older rendering paths.  New compact full-list
        # rendering uses _render_human_compact_item.
        if item.row_kind in {"amendment_note", "section_header", "skip"}:
            return None

        number = item.row_number or "—"
        indent_level = max(0, min((item.hierarchy_level or 1) - 1, 3))
        indent = "  " * indent_level
        text = self._shorten_for_compact_list(
            self._strip_leading_letter_marker(item.document_name),
            max_len=180 if item.row_kind in {"group", "bez_statement_group"} else 220,
        )

        if item.row_kind in {"group", "bez_statement_group"}:
            return f"{indent}{number}. {text.rstrip(':')}:"
        return f"{indent}{number}. {text}"

    def _shorten_for_compact_list(self, value: str, *, max_len: int) -> str:
        text = self._clean(value) or ""
        if len(text) <= max_len:
            return text

        # Prefer cutting before long legal explanations in parentheses/clauses.
        cut_candidates = [
            text.find(" ("),
            text.find(" в том числе"),
            text.find(" за исключением"),
            text.find(" утвержден"),
            text.find(" выданн"),
            text.find(" в соответствии"),
        ]
        cut_candidates = [pos for pos in cut_candidates if 60 <= pos <= max_len]
        if cut_candidates:
            return text[:min(cut_candidates)].rstrip(" ,;:") + "…"

        cut = text.rfind(" ", 0, max_len)
        if cut < 80:
            cut = max_len
        return text[:cut].rstrip(" ,;:") + "…"

    def _render_full_list_text(
        self,
        *,
        result: DocumentsAnswerBuildResult,
        submission_channel: Optional[str],
        include_submission_form_details: bool = False,
    ) -> str:
        items = result.full_list_items or result.all_items
        channel_label = self._channel_label(submission_channel) if (submission_channel and include_submission_form_details) else None

        required_items = [
            item for item in items
            if item.requirement_group == "required" and not item.is_bez_statement
        ]
        optional_items = [
            item for item in items
            if item.requirement_group == "optional" and not item.is_bez_statement
        ]
        other_items = [
            item for item in items
            if item.requirement_group not in {"required", "optional"} and not item.is_bez_statement
        ]
        bez_items = [item for item in items if item.is_bez_statement]

        lines: list[str] = [
            "Ниже приведён перечень документов с сохранением структуры регламента.",
        ]
        if channel_label:
            lines.append(
                f"Канал подачи: {channel_label}. Способ подачи показываю только у конкретных документов, "
                "если он заполнен в таблице; у строк-групп способ подачи не указывается."
            )

        def append_group(title: str, group_items: list[DocumentsAnswerItem]) -> None:
            if not group_items:
                return
            lines.append("")
            lines.append(title)
            for item in group_items:
                rendered = self._render_full_list_item(
                    item=item,
                    submission_channel=submission_channel,
                    include_submission_form_details=include_submission_form_details,
                )
                if rendered:
                    lines.append(rendered)

        append_group("Документы, которые заявитель представляет самостоятельно:", required_items)
        append_group("Документы и сведения, которые можно представить по собственной инициативе:", optional_items)

        if other_items:
            append_group("Прочие связанные позиции из таблицы:", other_items)

        if bez_items:
            lines.append("")
            lines.append("Отдельно о беззаявительном порядке:")
            if channel_label:
                lines.append(
                    "Эти пункты не являются документами, которые заявитель подаёт выбранным способом. "
                    "Они относятся к беззаявительному предоставлению помощи, когда сведения представляет уполномоченный орган."
                )
            for item in bez_items:
                rendered = self._render_full_list_item(
                    item=item,
                    submission_channel=None,
                    include_submission_form_details=False,
                )
                if rendered:
                    lines.append(rendered)

        document_count = len([item for item in items if item.row_kind in {"document", "bez_statement_document"}])
        group_count = len([item for item in items if item.row_kind in {"group", "bez_statement_group"}])
        if document_count or group_count:
            lines.append("")
            lines.append(
                f"Итого в таблице: {document_count} конкретных позиций документов/сведений "
                f"и {group_count} строк-групп."
            )

        return "\n".join(lines).strip()

    def _render_full_list_item(
        self,
        *,
        item: DocumentsAnswerItem,
        submission_channel: Optional[str],
        include_submission_form_details: bool,
    ) -> Optional[str]:
        if item.row_kind in {"amendment_note", "section_header", "skip"}:
            return None

        number = item.row_number or "—"
        indent_level = max(0, min((item.hierarchy_level or 1) - 1, 4))
        indent = "  " * indent_level
        text = self._strip_leading_letter_marker(item.document_name)

        if item.row_kind in {"group", "bez_statement_group"}:
            return f"{indent}{number}. {text.rstrip(':')}:"

        line = f"{indent}{number}. {text}"
        channel_note = None
        if include_submission_form_details:
            channel_note = self._render_channel_note(
                item=item,
                submission_channel=submission_channel,
            )
        if channel_note:
            line += f" — {channel_note}"
        return line

    def _render_compact_channel_note(
        self,
        *,
        item: DocumentsAnswerItem,
        submission_channel: Optional[str],
    ) -> Optional[str]:
        note = self._render_channel_note(
            item=item,
            submission_channel=submission_channel,
        )
        if not note:
            return None

        note = self._clean(note) or ""
        note = note.replace(
            "Копия документа сличается с подлинником, после чего подлинник возвращается заявителю (представителю)",
            "копия сверяется с подлинником, подлинник возвращается",
        )
        note = note.replace(
            "Копия документа сличается с подлинником, после чего подлинник возвращается представителю",
            "копия сверяется с подлинником, подлинник возвращается",
        )
        return self._shorten_for_compact_list(note, max_len=120)

    def _render_channel_note(
        self,
        *,
        item: DocumentsAnswerItem,
        submission_channel: Optional[str],
    ) -> Optional[str]:
        if not submission_channel:
            return None
        if item.row_kind in {"group", "bez_statement_group"}:
            return None

        label = self._channel_label(submission_channel)
        value = self._clean(item.requested_channel_value)
        if value and not self._is_dash_value(value):
            return f"{label}: {value}"

        # Some 6-column regulations have one generic column "Способ подачи документов",
        # not separate columns for МФЦ/ЕПГУ/почта. In that case show the generic value,
        # but do not pretend that it is a dedicated МФЦ column.
        generic = self._clean(item.generic_submission_value)
        if generic and not self._is_dash_value(generic):
            return f"способ подачи по таблице: {generic}"

        if value and self._is_dash_value(value):
            return f"{label}: в таблице указано «-»"

        return None

    def _collect_document_cells(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Collect document-table cells from all metadata mappings.

        Older extractor versions keep channel columns under generic repeated
        header keys such as ``способ_подачи_в_уполномоченное_учреждение_3``
        instead of semantic ``mfc_submission``.  The builder must not lose
        these values: normal document-list answers ignore form details, but
        explicit questions about originals/copies/scans need them.
        """
        result: dict[str, Any] = {}

        sources = [
            # Prefer display/raw values. ``cells_by_header_normalized`` can contain
            # normalized cell text with underscores, which is useful for search but
            # not suitable for answers shown to people.
            metadata.get("cells_by_semantic_key"),
            metadata.get("cells_by_header_key"),
            metadata.get("cells_by_header"),
            metadata.get("cells_by_header_normalized"),
        ]

        for source in sources:
            if not isinstance(source, dict):
                continue
            for raw_key, value in source.items():
                if value is None:
                    continue
                key = self._normalize_cell_key(raw_key)
                if not key:
                    continue
                # Do not overwrite a meaningful semantic value with a noisier
                # header-based duplicate.
                if key in result and self._clean(result.get(key)):
                    continue
                result[key] = value

        # Fallback for row number from cells_text/column zero.
        if not self._clean(result.get("row_number")):
            cells_text = metadata.get("cells_text") or []
            if isinstance(cells_text, list) and cells_text:
                result["row_number"] = cells_text[0]

        return result

    def _normalize_cell_key(self, value: Any) -> Optional[str]:
        key = str(value or "").strip()
        if not key:
            return None

        key_l = key.lower().replace("ё", "е")
        key_l = key_l.replace("[", " ").replace("]", " ")
        key_l = re.sub(r"[^a-zа-я0-9_]+", "_", key_l)
        key_l = re.sub(r"_+", "_", key_l).strip("_")

        direct = {
            "document_name": "document_name",
            "row_number": "row_number",
            "applicant_category_id": "applicant_category_id",
            "epgu_submission": "epgu_submission",
            "regional_portal_submission": "regional_portal_submission",
            "in_person_submission": "in_person_submission",
            "post_submission": "post_submission",
            "mfc_submission": "mfc_submission",
            "submission_method": "generic_submission",
            "submission_channel": "generic_submission",
        }
        if key_l in direct:
            return direct[key_l]

        if key_l in {"n", "n_п_п", "номер", "п_п", "№", "№_п_п"}:
            return "row_number"

        if "наименование" in key_l and "документ" in key_l:
            return "document_name"
        if "перечень" in key_l and "документ" in key_l:
            return "document_name"

        if "идентификатор" in key_l and "заявител" in key_l:
            return "applicant_category_id"

        # Tables with one common submission column.
        if "способы_подачи" in key_l or "способ_подачи_документ" in key_l:
            return "generic_submission"

        # 8-column tables often have four visually separate columns with the
        # same header "Способ подачи в уполномоченное учреждение".  In metadata
        # they become repeated keys with suffixes _2/_3/_4.
        if "способ_подачи_в_уполномоч" in key_l or "способ_подачи_в_министер" in key_l:
            if key_l.endswith("_2"):
                return "epgu_submission_positional"
            if key_l.endswith("_3"):
                return "mfc_submission_positional"
            if key_l.endswith("_4"):
                return "post_submission_positional"
            return "in_person_submission_positional"

        return key_l

    def _extract_channel_values(self, cells: dict[str, Any]) -> dict[str, str]:
        key_groups = {
            "epgu_submission": ["epgu_submission", "epgu_submission_positional"],
            "regional_portal_submission": ["regional_portal_submission", "epgu_submission_positional"],
            "in_person_submission": ["in_person_submission", "in_person_submission_positional"],
            "post_submission": ["post_submission", "post_submission_positional"],
            "mfc_submission": ["mfc_submission", "mfc_submission_positional"],
            "generic_submission": ["generic_submission", "submission_method", "submission_channel"],
        }
        result: dict[str, str] = {}
        for output_key, source_keys in key_groups.items():
            value = self._first_clean(cells, source_keys)
            if value:
                result[output_key] = value
        return result

    def _extract_generic_submission_note(
        self,
        *,
        cells: dict[str, Any],
        requested_channel_key: Optional[str],
    ) -> Optional[str]:
        # If a concrete channel column exists, generic fallback is not needed.
        if requested_channel_key and self._clean(cells.get(requested_channel_key)):
            return None

        candidates = [
            self._clean(cells.get("generic_submission")),
            self._clean(cells.get("submission_method")),
            self._clean(cells.get("submission_channel")),
        ]
        # Some 6-column tables use one common submission column, while older
        # semantic mapping can put that value into in_person_submission. Use it
        # only when no separate channel columns are present.
        if not any(self._clean(cells.get(key)) for key in [
            "epgu_submission",
            "epgu_submission_positional",
            "regional_portal_submission",
            "post_submission",
            "post_submission_positional",
            "mfc_submission",
            "mfc_submission_positional",
        ]):
            candidates.append(self._clean(cells.get("in_person_submission")))
            candidates.append(self._clean(cells.get("in_person_submission_positional")))

        for value in candidates:
            if value:
                return value
        return None

    def _extract_row_number(
        self,
        *,
        cells: dict[str, Any],
        metadata: dict[str, Any],
    ) -> Optional[str]:
        candidates: list[Any] = [
            cells.get("row_number"),
            metadata.get("row_number"),
        ]

        for source_key in ["cells_by_header_key", "cells_by_header", "cells_by_header_normalized"]:
            source = metadata.get(source_key) or {}
            if not isinstance(source, dict):
                continue
            for key, value in source.items():
                normalized_key = self._normalize(key)
                if normalized_key in {"n п п", "n", "№", "№ п п", "номер", "п п"}:
                    candidates.append(value)
                if normalized_key.replace(" ", "_") in {"n_п_п", "№_п_п"}:
                    candidates.append(value)

        cells_text = metadata.get("cells_text") or []
        if isinstance(cells_text, list) and cells_text:
            candidates.append(cells_text[0])

        for value in candidates:
            text = self._clean(value)
            if not text:
                continue
            # Strip common paragraph suffixes but keep hierarchical numbering.
            first_token = text.split()[0].strip()
            first_token = first_token.rstrip(".)")
            if self._is_number_like(first_token):
                return first_token
            if self._is_number_like(text):
                return text

        return None

    def _is_number_like(self, value: Optional[str]) -> bool:
        if not value:
            return False
        return bool(self._ROW_NUMBER_RE.fullmatch(str(value).strip()))

    def _parent_row_number(self, value: Optional[str]) -> Optional[str]:
        if not self._is_number_like(value):
            return None
        parts = str(value).split(".")
        if len(parts) <= 1:
            return None
        return ".".join(parts[:-1])

    def _has_any_submission_value(self, item: DocumentsAnswerItem) -> bool:
        values = [
            item.requested_channel_value,
            item.generic_submission_value,
            *item.channel_values.values(),
        ]
        return any(self._clean(value) and not self._is_dash_value(str(value)) for value in values)

    def _is_dash_value(self, value: str) -> bool:
        text = self._clean(value) or ""
        return text in {"-", "—", "–"}

    def _is_amendment_note(self, row_number: str, document_name: str) -> bool:
        text = self._normalize(f"{row_number} {document_name}")
        return any(marker in text for marker in [
            "в ред приказ",
            "в редакции приказ",
            "введен приказом",
            "введена приказом",
            "исключен приказом",
            "исключена приказом",
            "утратил силу",
            "утратила силу",
        ])

    def _is_section_heading_text(self, value: str) -> bool:
        text = self._normalize(value)
        return (
            "документы информация необходимые" in text
            or "документы необходимые для предоставления" in text
            or "представляемые заявителем самостоятельно" in text
            or "по собственной инициативе" in text
        )

    def _is_group_like_document_name(self, value: str) -> bool:
        text = self._normalize(value)
        if "беззаявительн" in text:
            return True
        return any(marker in text for marker in [
            "документы свидетельствующие",
            "документы подтверждающие",
            "документы копии документов сведения",
            "документы сведения подтверждающие",
            "документы о доходах",
            "сведения подтверждающие",
            "для предоставления в беззаявительном порядке",
        ])

    def _looks_like_bez_statement_row(self, item: DocumentsAnswerItem) -> bool:
        text = self._normalize(item.document_name)
        return any(marker in text for marker in [
            "беззаявительн",
            "без заявления",
            "без подачи заявления",
        ])

    def _strip_leading_letter_marker(self, value: str) -> str:
        text = self._clean(value) or ""
        # Keep legal wording but remove simple letter markers that are not part of the document name.
        return re.sub(r"^(?:[а-я]\)|[а-я]\.)\s+", "", text, flags=re.IGNORECASE).strip() or text

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
            "pension_proof",
            "other",
        }:
            return "general_document"

        # Страховка на случай, если family не распознался, но текст явно про представителя.
        if any(marker in text for marker in [
            "доверенн",
            "полномоч",
            "документ удостоверяющий личность представителя",
            "личность представителя",
            "представителя заявителя",
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

        # Представительские документы выделяем только по самому документу.
        # Служебные хвосты вида "заявителю (представителю)" в способах подачи
        # не должны превращать обычный документ в доверенность.
        if any(marker in text for marker in [
            "доверенн",
            "полномоч",
            "документ удостоверяющий личность представителя",
            "личность представителя",
            "представителя заявителя",
        ]):
            return "authority_document"

        if any(marker in text for marker in [
            "регистрац",
            "проживани",
            "место жительства",
            "место пребывания",
            "медицинском наблюдени",
            "медицинское наблюдение",
        ]):
            return "residency_proof"

        if any(marker in text for marker in [
            "решение суда",
        ]):
            return "court_decision"

        if any(marker in text for marker in [
            "трудов",
            "работ",
            "служб",
            "занятост",
            "стаж",
        ]):
            return "employment_proof"

        if any(marker in text for marker in [
            "пенсион",
            "пенсия",
        ]):
            return "pension_proof"

        if any(marker in text for marker in [
            "справк",
            "удостоверени",
            "свидетельств",
            "подтверждающ статус",
            "категори",
        ]):
            return "status_certificate"

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
            return self._first_clean(cells, ["epgu_submission", "epgu_submission_positional"])
        if submission_channel == "regional_portal":
            return self._first_clean(cells, [
                "regional_portal_submission",
                "epgu_submission",
                "epgu_submission_positional",
            ])
        if submission_channel == "in_person":
            return self._first_clean(cells, ["in_person_submission", "in_person_submission_positional"])
        if submission_channel == "post":
            return self._first_clean(cells, ["post_submission", "post_submission_positional"])
        if submission_channel == "mfc":
            return self._first_clean(cells, ["mfc_submission", "mfc_submission_positional"])
        return None

    def _first_clean(self, cells: dict[str, Any], keys: list[str]) -> Optional[str]:
        for key in keys:
            value = self._clean(cells.get(key))
            if value:
                return value
        return None

    def _is_noise_or_truncated_document_name(self, value: str) -> bool:
        text = self._normalize(value)
        if not text:
            return True

        # Частый дефект таблиц: ячейка оборвалась на союзе и в ответ попадало
        # "сведения о прохождении заявителем и" / "сведения о нахождении заявителя и".
        if text.endswith((" и", " или", " либо", " а также")):
            if text.startswith(("сведения о", "документы о", "информация о")):
                return True

        if len(text.split()) <= 5 and text.startswith("сведения о") and text.endswith("заявителя и"):
            return True

        return False

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