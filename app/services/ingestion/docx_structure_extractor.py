from __future__ import annotations

import logging
import re
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional
from uuid import uuid4

from docx import Document as DocxDocument
from docx.document import Document as _Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph

from app.services.ingestion.document_ingestion_pipeline import (
    ExtractionInput,
    ExtractionResult,
)

from app.config.measure_registry import (
    detect_primary_measure_code,
    get_measure_definition,
)

logger = logging.getLogger(__name__)


class DocxStructureExtractorError(Exception):
    """Base DOCX structure extraction error."""


class DocxStructureExtractor:
    """
    Concrete DOCX extractor for normative/legal documents.

    Responsibilities:
    - read DOCX files with paragraph/table order preserved
    - extract structural text blocks
    - extract tables as table-level objects
    - extract table rows as row-level retrieval units
    - build retrieval-friendly summaries for table rows
    - provide conservative document-level metadata

    Notes:
    - blank template tables without meaningful rows are skipped
    - form-like tables are preserved but marked with table_type='form_fields'
    - title detection prefers real table captions/headings over nearby field labels
    """

    _REVISION_DATE_RE = re.compile(
        r"(?:от\s+)?(?P<date>\d{2}\.\d{2}\.\d{4})",
        flags=re.IGNORECASE,
    )
    _NUMBER_RE = re.compile(
        r"(?:№|N)\s*(?P<number>[0-9A-Za-zА-Яа-я\-/]+)",
        flags=re.IGNORECASE,
    )
    _APPENDIX_RE = re.compile(
        r"^\s*(?:приложение)\s*№?\s*(?P<num>[0-9A-Za-zА-Яа-я\-/]+)?",
        flags=re.IGNORECASE,
    )
    _SECTION_RE = re.compile(
        r"^\s*(?P<section>(?:[IVXLCM]+|\d+(?:\.\d+)*)\.?)\s+(?P<title>.+)$",
        flags=re.IGNORECASE,
    )
    _CLAUSE_RE = re.compile(
        r"^\s*(?P<clause>\d+(?:\.\d+)*)[\)\.]?\s+(?P<text>.+)$",
        flags=re.IGNORECASE,
    )

    _GOOD_TITLE_KEYWORDS = (
        "таблица",
        "перечень",
        "документы",
        "документ",
        "основан",
        "отказ",
        "идентификатор",
        "категори",
        "срок",
        "условных обозначений",
        "сокращени",
        "результат",
        "исчерпывающий перечень",
    )

    _STRONG_TITLE_STARTS = (
        "ИСЧЕРПЫВАЮЩИЙ ПЕРЕЧЕНЬ",
        "ИДЕНТИФИКАТОРЫ КАТЕГОРИЙ",
        "ПЕРЕЧЕНЬ НЕОБХОДИМЫХ",
        "ПЕРЕЧЕНЬ ДОКУМЕНТОВ",
        "ПЕРЕЧЕНЬ ОСНОВАНИЙ",
        "ФОРМА",
        "СОСТАВ ЗАПРОСА",
        "ЗАЯВЛЕНИЕ",
        "СОГЛАСИЕ",
        "РАСПИСКА",
        "ПЕРЕЧЕНЬ УСЛОВНЫХ ОБОЗНАЧЕНИЙ",
    )

    _NOISE_TABLE_MARKERS = (
        "СПИСОК ИЗМЕНЯЮЩИХ ДОКУМЕНТОВ",
        "КОНСУЛЬТАНТПЛЮС",
    )

    _FORM_TITLE_MARKERS = (
        "СОСТАВ ЗАПРОСА",
        "СОГЛАСИЕ НА ОБРАБОТКУ ПЕРСОНАЛЬНЫХ ДАННЫХ",
        "К ЗАЯВЛЕНИЮ ПРИЛАГАЮ",
        "ПРИЛАГАЮ СЛЕДУЮЩИЕ ДОКУМЕНТЫ",
        "ПРИЛАГАЮТСЯ СЛЕДУЮЩИЕ ДОКУМЕНТЫ",
        "ПЕРЕЧЕНЬ ПРИЛАГАЕМЫХ ДОКУМЕНТОВ",
        "СВЕДЕНИЯ О ДОКУМЕНТЕ",
        "СВЕДЕНИЯ О ДОКУМЕНТАХ",
        "СВЕДЕНИЯ О МЕДИЦИНСКИХ ДОКУМЕНТАХ",
        "ДОКУМЕНТ, УДОСТОВЕРЯЮЩИЙ ЛИЧНОСТЬ",
        "ДОКУМЕНТА, УДОСТОВЕРЯЮЩЕГО ЛИЧНОСТЬ",
        "ПОДТВЕРЖДАЮЩЕГО ПОЛНОМОЧИЯ ПРЕДСТАВИТЕЛЯ",
        "СВЕДЕНИЯ О ЗАКОННОМ ПРЕДСТАВИТЕЛЕ",
        "СВЕДЕНИЯ О ПРЕДСТАВИТЕЛЕ",
        "ЗАЯВЛЕНИЕ",
        "ФОРМА",
        "СОГЛАСИЕ",
        "РАСПИСКА",
    )

    _FORM_CONTEXT_MARKERS = (
        "ПОДПИСЬ ЗАЯВИТЕЛЯ",
        "ФАМИЛИЯ, ИМЯ, ОТЧЕСТВО",
        "ДАТА ПОДПИСЬ",
        "ДОСТОВЕРНОСТЬ И ПОЛНОТУ СВЕДЕНИЙ",
        "ИТОГО ПРИЛОЖЕНИЯ",
        "ИТОГО: ПРИЛОЖЕНИЯ",
        "ПРИНЯЛ ДОКУМЕНТЫ",
        "КОЛИЧЕСТВО (ШТ.)",
        "КОЛ-ВО (ШТ.)",
        "КОЛ-ВО (ЛИСТОВ)",
    )

    _CATEGORY_MARKERS = (
        "ИДЕНТИФИКАТОРЫ КАТЕГОРИЙ",
        "КАТЕГОРИЙ (ПРИЗНАКОВ) ЗАЯВИТЕЛЕЙ",
        "ПРИЗНАКОВ ЗАЯВИТЕЛЕЙ",
        "НАИМЕНОВАНИЕ ПРИЗНАКА ЗАЯВИТЕЛЯ",
        "НАИМЕНОВАНИЕ ОТДЕЛЬНЫХ ПРИЗНАКОВ ЗАЯВИТЕЛЕЙ",
    )

    _DOCUMENTS_MARKERS = (
        "ИСЧЕРПЫВАЮЩИЙ ПЕРЕЧЕНЬ ДОКУМЕНТОВ",
        "ПЕРЕЧЕНЬ ДОКУМЕНТОВ",
        "НАИМЕНОВАНИЕ ДОКУМЕНТА",
        "ДОКУМЕНТЫ, НЕОБХОДИМЫЕ ДЛЯ ПРЕДОСТАВЛЕНИЯ",
        "ДОКУМЕНТОВ, НЕОБХОДИМЫХ ДЛЯ ПРЕДОСТАВЛЕНИЯ",
    )

    _REFUSAL_MARKERS = (
        "ИСЧЕРПЫВАЮЩИЙ ПЕРЕЧЕНЬ ОСНОВАНИЙ",
        "ОСНОВАНИЙ ДЛЯ ОТКАЗА",
        "ОТКАЗА В ПРИЕМЕ",
        "ОТКАЗ В ПРИЕМЕ",
        "ПРИОСТАНОВЛЕНИЯ ПРЕДОСТАВЛЕНИЯ",
        "ОТКАЗА В ПРЕДОСТАВЛЕНИИ",
        "ПЕРЕЧЕНЬ ОСНОВАНИЙ",
    )

    _AMOUNT_MARKERS = (
        "РАЗМЕР",
        "СУММА",
        "РУБ",
        "ВЫПЛАТ",
    )

    _BAD_TITLE_PATTERNS = (
        re.compile(r"^\([^)]{0,200}\)$", flags=re.IGNORECASE),
        re.compile(r"^\s*\d+\.?\s*$", flags=re.IGNORECASE),
        re.compile(r"^\s*\d+\.\s*к\s+заявлению\s+прилага", flags=re.IGNORECASE),
        re.compile(r"к\s+заявлению\s+прилага", flags=re.IGNORECASE),
        re.compile(r"перечень\s+прилагаемых\s+документов", flags=re.IGNORECASE),
        re.compile(r"сведения\s+о\s+.*документ", flags=re.IGNORECASE),
        re.compile(r"подтверждающего\s+полномочия\s+представителя", flags=re.IGNORECASE),
        re.compile(r"водоснабжение|водоотведение", flags=re.IGNORECASE),
        re.compile(r"привед[ее]н[аы]?\s+в\s+таблице", flags=re.IGNORECASE),
        re.compile(r"^\s*приложение\s+N?\s*\d+", flags=re.IGNORECASE),
        re.compile(r"^\s*почтовый адрес", flags=re.IGNORECASE),
        re.compile(r"^\s*телефон", flags=re.IGNORECASE),
        re.compile(r"^\s*адрес электронной почты", flags=re.IGNORECASE),
        re.compile(r"^\s*кем выдан", flags=re.IGNORECASE),
        re.compile(r"^\s*серия,\s*номер", flags=re.IGNORECASE),
        re.compile(r"^\s*дата выдачи", flags=re.IGNORECASE),
        re.compile(r"^\s*срок действия полномочий", flags=re.IGNORECASE),
    )
    
    _TEXTUAL_DATE_RE = re.compile(
        r"(?P<day>\d{1,2})\s+"
        r"(?P<month>января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)"
        r"\s+(?P<year>\d{4})\s*г?\.?",
        flags=re.IGNORECASE,
    )

    _RUS_MONTHS = {
        "января": 1,
        "февраля": 2,
        "марта": 3,
        "апреля": 4,
        "мая": 5,
        "июня": 6,
        "июля": 7,
        "августа": 8,
        "сентября": 9,
        "октября": 10,
        "ноября": 11,
        "декабря": 12,
    }

    _TITLE_STOP_MARKERS = (
        "в соответствии с",
        "i. общие положения",
        "i. предмет",
        "предмет регулирования",
        "круг заявителей",
        "список изменяющих документов",
    )

    _TITLE_SKIP_EXACT = {
        "министерство социальной политики",
        "министерство социальной политики красноярского края",
        "красноярского края",
        "приказ",
        "приложение",
    }

    def __init__(
        self,
        *,
        max_markdown_preview_rows: int = 8,
        keep_last_paragraph_context: int = 14,
    ) -> None:
        self.max_markdown_preview_rows = max_markdown_preview_rows
        self.keep_last_paragraph_context = keep_last_paragraph_context

    async def extract(
        self,
        payload: ExtractionInput,
    ) -> ExtractionResult:
        self._validate_input(payload)

        file_path = Path(payload.file_path)
        doc = DocxDocument(str(file_path))

        blocks: list[dict[str, Any]] = []
        tables: list[dict[str, Any]] = []
        table_rows: list[dict[str, Any]] = []

        paragraph_context: deque[dict[str, Any]] = deque(
            maxlen=self.keep_last_paragraph_context
        )

        block_order = 0
        table_counter = 0
        meaningful_paragraph_count = 0
        skipped_blank_tables_count = 0

        current_heading_text: Optional[str] = None
        heading_path: list[str] = []

        for item in self._iter_block_items(doc):
            if isinstance(item, Paragraph):
                block = self._build_block_from_paragraph(
                    paragraph=item,
                    block_order=block_order + 1,
                    current_heading_text=current_heading_text,
                    heading_path=heading_path,
                )
                if block is None:
                    continue

                block_order += 1
                blocks.append(block)

                if block.get("block_type") == "heading":
                    heading_text = self._clean_text(block.get("content_clean") or "")
                    if heading_text:
                        current_heading_text = heading_text
                        heading_path = self._update_heading_path(
                            heading_path=heading_path,
                            heading_text=heading_text,
                            section_number=block.get("section_number"),
                        )

                if self._is_meaningful_text(block.get("content_clean")):
                    meaningful_paragraph_count += 1
                    paragraph_context.append(block)

            elif isinstance(item, Table):
                table_counter += 1
                table_id = f"docx_tbl_{table_counter}_{uuid4().hex[:8]}"

                table_title = self._detect_table_title(
                    paragraph_context=list(paragraph_context),
                    fallback_number=table_counter,
                )

                row_payloads = self._build_table_row_payloads(
                    table=item,
                    table_id=table_id,
                    table_number=str(table_counter),
                    table_title=table_title,
                    paragraph_context=list(paragraph_context),
                )

                if not row_payloads:
                    skipped_blank_tables_count += 1
                    logger.info(
                        "Skipping blank DOCX table without meaningful rows",
                        extra={
                            "file_path": payload.file_path,
                            "table_number": table_counter,
                            "table_title": table_title,
                        },
                    )
                    continue

                table_payload = self._build_table_payload(
                    table=item,
                    table_id=table_id,
                    table_number=str(table_counter),
                    table_title=table_title,
                    paragraph_context=list(paragraph_context),
                    row_payloads=row_payloads,
                )
                tables.append(table_payload)
                table_rows.extend(row_payloads)

        document_title = self._detect_document_title(
            original_filename=payload.original_filename,
            blocks=blocks,
        )

        document_number, document_date = self._extract_document_date_and_number(
            original_filename=payload.original_filename,
            blocks=blocks,
        )

        service_name_full = self._detect_service_name_full(
            blocks=blocks,
        )

        primary_measure_code = detect_primary_measure_code(
            service_name_full
            or document_title
            or payload.normalized_text[:4000]
        )

        measure_definition = get_measure_definition(primary_measure_code)
        self._attach_measure_code_to_rows(
            table_rows=table_rows,
            primary_measure_code=primary_measure_code,
        )
        service_name_short = None
        if measure_definition is not None:
            service_name_short = (
                measure_definition.aliases[0]
                if measure_definition.aliases
                else measure_definition.canonical_name
            )

        revision_date = self._detect_revision_date(
            original_filename=payload.original_filename,
            blocks=blocks,
            normalized_text=payload.normalized_text,
            document_date=document_date,
        )

        doc_uid_base = self._detect_doc_uid_base(
            original_filename=payload.original_filename,
            document_title=document_title,
            normalized_text=payload.normalized_text,
            service_name_full=service_name_full,
            document_number=document_number,
            document_date=document_date,
        )

        extraction_payload_json = {
            "extractor": "docx_structure_extractor",
            "declared_table_count": table_counter,
            "skipped_blank_tables_count": skipped_blank_tables_count,
            "blocks_count": len(blocks),
            "tables_count": len(tables),
            "table_rows_count": len(table_rows),
            "meaningful_paragraph_count": meaningful_paragraph_count,
            "source_format": "docx",
            "document_number": document_number,
            "document_date": (
                document_date.isoformat()
                if document_date is not None
                else None
            ),
            "revision_date": (
                revision_date.isoformat()
                if revision_date is not None
                else None
            ),
            "service_name_full": service_name_full,
            "service_name_short": service_name_short,
            "primary_measure_code": primary_measure_code,
        }

        logger.info(
            "DOCX structure extracted",
            extra={
                "file_path": payload.file_path,
                "blocks_count": len(blocks),
                "tables_count": len(tables),
                "table_rows_count": len(table_rows),
                "doc_uid_base": doc_uid_base,
                "primary_measure_code": primary_measure_code,
            },
        )

        return ExtractionResult(
            document_title=document_title,
            doc_uid_base=doc_uid_base,
            revision_date=revision_date,
            document_number=document_number,
            document_date=document_date,
            service_name_full=service_name_full,
            service_name_short=service_name_short,
            primary_measure_code=primary_measure_code,
            blocks=blocks,
            tables=tables,
            table_rows=table_rows,
            extraction_payload_json=extraction_payload_json,
        )

    def _validate_input(self, payload: ExtractionInput) -> None:
        if payload is None:
            raise DocxStructureExtractorError("ExtractionInput must not be None.")

        if not payload.file_path or not str(payload.file_path).strip():
            raise DocxStructureExtractorError("file_path is required.")

        path = Path(payload.file_path)
        if not path.exists():
            raise DocxStructureExtractorError(f"DOCX file not found: {payload.file_path}")

        if path.suffix.lower() != ".docx":
            raise DocxStructureExtractorError(
                f"Unsupported extension for DOCX extractor: {path.suffix}"
            )

    def _iter_block_items(
        self,
        parent: _Document | _Cell,
    ) -> Iterator[Paragraph | Table]:
        """
        Yield Paragraph and Table objects in document order.
        """
        parent_element = parent.element.body if isinstance(parent, _Document) else parent._tc

        for child in parent_element.iterchildren():
            if isinstance(child, CT_P):
                yield Paragraph(child, parent)
            elif isinstance(child, CT_Tbl):
                yield Table(child, parent)

    def _build_block_from_paragraph(
        self,
        *,
        paragraph: Paragraph,
        block_order: int,
        current_heading_text: Optional[str],
        heading_path: list[str],
    ) -> Optional[dict[str, Any]]:
        raw_text = self._clean_text(paragraph.text)
        if not raw_text:
            return None

        style_name = self._safe_style_name(paragraph)
        block_type = self._detect_block_type(raw_text, style_name)

        section_number = None
        clause_number = None
        appendix_number = None

        appendix_match = self._APPENDIX_RE.match(raw_text)
        if appendix_match:
            appendix_number = appendix_match.group("num")

        section_match = self._SECTION_RE.match(raw_text)
        if section_match and block_type == "heading":
            section_number = self._clean_text(section_match.group("section"))

        clause_match = self._CLAUSE_RE.match(raw_text)
        if clause_match:
            clause_number = self._clean_text(clause_match.group("clause"))

        return {
            "block_order": block_order,
            "block_type": block_type,
            "content_raw": paragraph.text,
            "content_clean": raw_text,
            "chapter": None,
            "section_number": section_number,
            "clause_number": clause_number,
            "appendix_number": appendix_number,
            "table_number": None,
            "citation_json": {
                "source_type": "docx_paragraph",
                "block_order": block_order,
            },
            "metadata_json": {
                "style_name": style_name,
                "is_heading_style": self._is_heading_style(style_name),
                "is_list_like": self._looks_like_list_item(raw_text),
                "current_heading_text": current_heading_text,
                "heading_path": list(heading_path),
                "block_semantic_hints": self._infer_block_semantic_hints(
                    text=raw_text,
                    current_heading_text=current_heading_text,
                    heading_path=heading_path,
                ),
            },
        }

    def _detect_block_type(
        self,
        text: str,
        style_name: str,
    ) -> str:
        if self._is_heading_style(style_name):
            return "heading"
        if self._looks_like_list_item(text):
            return "list_item"
        if text.lower().startswith("таблица"):
            return "table_caption"
        return "paragraph"

    def _safe_style_name(self, paragraph: Paragraph) -> str:
        try:
            style = paragraph.style
            if style is None:
                return ""
            return str(style.name or "").strip()
        except Exception:
            return ""

    def _is_heading_style(self, style_name: str) -> bool:
        style_name_normalized = style_name.strip().lower()
        return "heading" in style_name_normalized or "заголов" in style_name_normalized

    def _looks_like_list_item(self, text: str) -> bool:
        return bool(
            re.match(
                r"^\s*(?:[-–—•*]|\d+[\.\)]|[а-яa-z]\))\s+",
                text,
                flags=re.IGNORECASE,
            )
        )

    def _build_table_payload(
        self,
        *,
        table: Table,
        table_id: str,
        table_number: str,
        table_title: str,
        paragraph_context: list[dict[str, Any]],
        row_payloads: list[dict[str, Any]],
    ) -> dict[str, Any]:
        headers, header_keys = self._extract_headers(table)
        appendix_number = self._detect_appendix_number_from_context(paragraph_context)
        normative_table_number = self._detect_normative_table_number_from_context(paragraph_context)
        table_type = self._detect_table_type(
            table_title=table_title,
            headers=headers,
            row_payloads=row_payloads,
            normative_table_number=normative_table_number,
        )
        effective_table_title = self._repair_table_title(
            table_title=table_title,
            table_type=table_type,
            headers=headers,
            row_payloads=row_payloads,
            normative_table_number=normative_table_number,
        )

        metadata_json = {
            "docx_table_index": int(table_number),
            "normative_table_number": normative_table_number,
            "table_semantic_type": table_type,
            "preceding_paragraphs": [
                x.get("content_clean")
                for x in paragraph_context
                if self._is_meaningful_text(x.get("content_clean"))
            ],
            "header_columns_count": len(headers),
        }
        if effective_table_title != table_title:
            metadata_json["detected_table_title_raw"] = table_title
            metadata_json["table_title_repaired"] = True

        return {
            "table_id": table_id,
            "table_number": table_number,
            "appendix_number": appendix_number,
            "table_type": table_type,
            "table_title": effective_table_title,
            "summary": self._build_table_summary(
                table_title=effective_table_title,
                headers=headers,
                rows_count=len(row_payloads),
                table_type=table_type,
            ),
            "header_schema_json": {
                "columns": [
                    {
                        "index": idx + 1,
                        "name": headers[idx],
                        "key": header_keys[idx],
                    }
                    for idx in range(len(headers))
                ],
                "raw_headers": headers,
                "normalized_keys": header_keys,
            },
            "rows_count": len(row_payloads),
            "markdown_preview": self._render_markdown_preview(
                headers=headers,
                header_keys=header_keys,
                rows=row_payloads,
            ),
            "citation_json": {
                "source_type": "docx_table",
                "table_number": table_number,
                "normative_table_number": normative_table_number,
                "table_title": effective_table_title,
            },
            "metadata_json": metadata_json,
        }

    def _build_table_row_payloads(
        self,
        *,
        table: Table,
        table_id: str,
        table_number: str,
        table_title: str,
        paragraph_context: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        headers, header_keys = self._extract_headers(table)
        raw_rows = self._extract_raw_rows(table, header_keys)
        row_payloads: list[dict[str, Any]] = []

        current_requirement_group = "unknown"
        current_requirement_group_label: Optional[str] = None

        preview_rows: list[dict[str, Any]] = []
        for idx, row_json in enumerate(raw_rows, start=1):
            normalized_row_json = {
                self._normalize_column_key(k): self._normalize_value(v)
                for k, v in row_json.items()
            }
            if self._is_effectively_empty_row(normalized_row_json):
                continue

            if self._is_structural_numbering_row(
                row_json=row_json,
                normalized_row_json=normalized_row_json,
            ):
                continue

            if self._is_service_section_row(
                row_json=row_json,
                normalized_row_json=normalized_row_json,
            ):
                service_section = self._classify_service_section_row(
                    row_json=row_json,
                    normalized_row_json=normalized_row_json,
                )
                if service_section is not None:
                    current_requirement_group = service_section["section_kind"]
                    current_requirement_group_label = service_section["section_label"]
                continue

            row_summary = self._build_row_summary(
                table_title=table_title,
                headers=headers,
                row_json=row_json,
            )

            preview_rows.append(
                {
                    "table_id": table_id,
                    "row_order": idx,
                    "row_json": row_json,
                    "normalized_row_json": normalized_row_json,
                    "row_summary": row_summary,
                    "row_context": {
                        "requirement_group": current_requirement_group,
                        "requirement_group_label": current_requirement_group_label,
                    },
                }
            )

        if not preview_rows:
            return []

        appendix_number = self._detect_appendix_number_from_context(paragraph_context)
        normative_table_number = self._detect_normative_table_number_from_context(paragraph_context)
        table_type = self._detect_table_type(
            table_title=table_title,
            headers=headers,
            row_payloads=preview_rows,
            normative_table_number=normative_table_number,
        )
        effective_table_title = self._repair_table_title(
            table_title=table_title,
            table_type=table_type,
            headers=headers,
            row_payloads=preview_rows,
            normative_table_number=normative_table_number,
        )

        current_refusal_scope: Optional[str] = None

        for row in preview_rows:
            row_json = row["row_json"]
            normalized_row_json = row["normalized_row_json"]
            effective_row_summary = self._build_row_summary(
                table_title=effective_table_title,
                headers=headers,
                row_json=row_json,
            )

            cells_by_header = self._build_cells_by_header(
                headers=headers,
                header_keys=header_keys,
                row_json=row_json,
            )
            cells_by_header_normalized = self._build_cells_by_header_normalized(
                headers=headers,
                header_keys=header_keys,
                normalized_row_json=normalized_row_json,
                table_type=table_type,
            )
            cells_by_semantic_key = self._build_cells_by_semantic_key(
                headers=headers,
                header_keys=header_keys,
                row_json=row_json,
                table_type=table_type,
            )

            section_text = self._build_refusal_section_candidate_text(
                cells_by_semantic_key=cells_by_semantic_key,
                cells_by_header=cells_by_header,
            )

            explicit_section_scope = self._detect_refusal_section_scope(
                table_type=table_type,
                section_text=section_text,
            )

            if table_type == "refusal_reasons" and self._is_refusal_section_heading_row(
                explicit_section_scope=explicit_section_scope,
                cells_by_header=cells_by_header,
            ):
                current_refusal_scope = explicit_section_scope
                continue

            explicit_row_scope = self._detect_refusal_row_scope(
                table_type=table_type,
                cells_by_semantic_key=cells_by_semantic_key,
                cells_by_header=cells_by_header,
                cells_by_header_normalized=cells_by_header_normalized,
            )

            row_scope_source: Optional[str] = None

            if table_type == "refusal_reasons":
                if current_refusal_scope is not None:
                    row_scope = current_refusal_scope
                    row_scope_source = "section_context"
                elif explicit_row_scope in {"renewal_refusal", "suspension", "intake_refusal", "service_refusal"}:
                    row_scope = explicit_row_scope
                    row_scope_source = "row_text"
                elif explicit_row_scope is not None:
                    row_scope = explicit_row_scope
                    row_scope_source = "row_text_fallback"
                else:
                    row_scope = "service_refusal"
                    row_scope_source = "default_service_fallback"
            else:
                row_scope = None
                row_scope_source = None

            metadata_json = {
                "docx_table_index": int(table_number),
                "normative_table_number": normative_table_number,
                "table_number": table_number,
                "table_title": effective_table_title,
                "appendix_number": appendix_number,
                "table_semantic_type": table_type,
                "row_kind": "data_row",
                "row_scope": row_scope,
                "row_scope_source": row_scope_source,
                "requirement_group": row.get("row_context", {}).get("requirement_group", "unknown"),
                "requirement_group_label": row.get("row_context", {}).get("requirement_group_label"),
                "table_section_context": {
                    "requirement_group": row.get("row_context", {}).get("requirement_group", "unknown"),
                    "requirement_group_label": row.get("row_context", {}).get("requirement_group_label"),
                },
                "column_headers": headers,
                "header_keys": header_keys,
                "cells_text": [v for v in row_json.values() if self._clean_text(v)],
                "cells_by_header": cells_by_header,
                "cells_by_header_key": {
                    key: self._clean_text(str(row_json.get(key, "")))
                    for key in header_keys
                    if self._clean_text(str(row_json.get(key, "")))
                },
                "cells_by_header_normalized": cells_by_header_normalized,
                "cells_by_semantic_key": cells_by_semantic_key,
            }
            if effective_table_title != table_title:
                metadata_json["detected_table_title_raw"] = table_title
                metadata_json["table_title_repaired"] = True

            row_payloads.append(
                {
                    "table_id": table_id,
                    "row_order": row["row_order"],
                    "row_json": row_json,
                    "normalized_row_json": normalized_row_json,
                    "row_summary": effective_row_summary,
                    "citation_json": {
                        "source_type": "docx_table_row",
                        "table_number": table_number,
                        "normative_table_number": normative_table_number,
                        "table_title": effective_table_title,
                        "row_order": row["row_order"],
                    },
                    "metadata_json": metadata_json,
                }
            )

        return row_payloads
        
    def _attach_measure_code_to_rows(
        self,
        *,
        table_rows: list[dict[str, Any]],
        primary_measure_code: Optional[str],
    ) -> None:
        if not primary_measure_code:
            return

        for row in table_rows:
            metadata = row.get("metadata_json")
            if not isinstance(metadata, dict):
                metadata = {}
                row["metadata_json"] = metadata

            metadata.setdefault("measure_code", primary_measure_code)

    def _detect_refusal_row_scope(
        self,
        *,
        table_type: str,
        cells_by_semantic_key: dict[str, str] | None,
        cells_by_header: dict[str, str] | None,
        cells_by_header_normalized: dict[str, str] | None,
    ) -> Optional[str]:
        if table_type != "refusal_reasons":
            return None

        cells_by_semantic_key = cells_by_semantic_key or {}
        cells_by_header = cells_by_header or {}
        cells_by_header_normalized = cells_by_header_normalized or {}

        # Важно: не используем table_title и row_summary для row-level scope.
        # В объединённых таблицах отказов общий заголовок содержит сразу
        # "отказ в приёме / приостановление / отказ в предоставлении".
        # Если учитывать его в каждой строке, все строки ошибочно становятся intake_refusal.
        local_parts = [
            cells_by_semantic_key.get("refusal_reason", ""),
            *cells_by_header.values(),
            *cells_by_header_normalized.values(),
        ]
        haystack = self._normalize_search_text(" ".join(x for x in local_parts if x))

        if not haystack:
            return None

        intake_markers = (
            "отказ в приеме запроса",
            "отказа в приеме запроса",
            "отказ в приеме заявления",
            "отказа в приеме заявления",
            "отказ в приеме документов",
            "отказа в приеме документов",
            "отказ в приеме к рассмотрению",
            "отказа в приеме к рассмотрению",
            "отказ в принятии документов",
            "отказа в принятии документов",
        )
        if any(marker in haystack for marker in intake_markers):
            return "intake_refusal"

        if "приостанов" in haystack:
            return "suspension"

        renewal_markers = (
            "отказ в возобновлении",
            "отказа в возобновлении",
            "об отказе в возобновлении",
            "возобновлении едв",
            "возобновлении выплаты",
        )
        if any(marker in haystack for marker in renewal_markers):
            return "renewal_refusal"

        service_markers = (
            "отказ в предоставлении",
            "отказа в предоставлении",
            "об отказе в предоставлении",
            "отказ в назначении",
            "отказа в назначении",
            "об отказе в назначении",
            "отказ в предоставлении государственной услуги",
            "отказа в предоставлении государственной услуги",
        )
        if any(marker in haystack for marker in service_markers):
            return "service_refusal"

        return None
        
    def _detect_refusal_section_scope(
        self,
        *,
        table_type: str,
        section_text: str,
    ) -> Optional[str]:
        if table_type != "refusal_reasons":
            return None

        haystack = self._normalize_search_text(section_text)
        if not haystack:
            return None

        intake_markers = (
            "исчерпывающий перечень оснований для отказа в приеме",
            "отказа в приеме заявления и документов",
            "отказ в приеме заявления и документов",
            "отказа в приеме к рассмотрению",
            "отказ в приеме к рассмотрению",
        )
        if any(marker in haystack for marker in intake_markers):
            return "intake_refusal"

        if "исчерпывающий перечень оснований для приостановления" in haystack:
            return "suspension"

        renewal_markers = (
            "исчерпывающий перечень оснований для отказа в возобновлении",
            "отказ в возобновлении",
            "отказа в возобновлении",
            "об отказе в возобновлении",
        )
        if any(marker in haystack for marker in renewal_markers):
            return "renewal_refusal"

        service_markers = (
            "исчерпывающий перечень оснований для отказа в предоставлении",
            "отказ в предоставлении",
            "отказа в предоставлении",
            "об отказе в предоставлении",
            "отказ в назначении",
            "отказа в назначении",
        )
        if any(marker in haystack for marker in service_markers):
            return "service_refusal"

        return None
        
    def _build_refusal_section_candidate_text(
        self,
        *,
        cells_by_semantic_key: dict[str, str] | None,
        cells_by_header: dict[str, str] | None,
    ) -> str:
        cells_by_semantic_key = cells_by_semantic_key or {}
        cells_by_header = cells_by_header or {}

        # Для section-scope берём только содержимое текущей строки.
        # Общий table_title намеренно исключён: он часто перечисляет все виды отказов
        # и ломает классификацию каждой отдельной строки.
        parts = [
            cells_by_semantic_key.get("refusal_reason", ""),
            *cells_by_header.values(),
        ]
        return self._normalize_search_text(" ".join(x for x in parts if x))

    def _is_refusal_section_heading_row(
        self,
        *,
        explicit_section_scope: Optional[str],
        cells_by_header: dict[str, str] | None,
    ) -> bool:
        if explicit_section_scope not in {
            "renewal_refusal",
            "suspension",
            "intake_refusal",
            "service_refusal",
        }:
            return False

        cells_by_header = cells_by_header or {}
        values = [self._clean_text(str(value)) for value in cells_by_header.values()]
        values = [value for value in values if value]
        if not values:
            return False

        combined = self._normalize_search_text(" ".join(values))
        if "исчерпывающий перечень оснований" not in combined:
            return False

        # В DOCX merged cells часто приходят как несколько одинаковых значений.
        # Секционный заголовок обычно занимает всю строку, а не является причиной отказа.
        unique_values = {self._normalize_search_text(value) for value in values if value}
        has_row_number = any(re.fullmatch(r"\d+(?:\.\d+)?", value.strip()) for value in values)
        if len(unique_values) <= 2 and not has_row_number:
            return True

        first_value = self._normalize_search_text(values[0])
        if not re.fullmatch(r"\d+(?:\.\d+)?", values[0].strip()):
            if first_value.startswith("исчерпывающий перечень оснований"):
                return True

        return False


    def _normalize_search_text(
        self,
        value: Any,
    ) -> str:
        text = self._clean_text(value)
        if not text:
            return ""
        return text.lower().replace("ё", "е")

    def _extract_headers(
        self,
        table: Table,
    ) -> tuple[list[str], list[str]]:
        if not table.rows:
            return ([], [])

        header_cells = [self._clean_text(cell.text) for cell in table.rows[0].cells]
        headers = [
            value if value else f"Колонка {idx + 1}"
            for idx, value in enumerate(header_cells)
        ]

        # If the header row is obviously broken/overly long, prefer generic columns.
        if self._looks_like_broken_header(headers):
            headers = [f"Колонка {idx + 1}" for idx in range(len(header_cells))]

        header_keys = self._make_unique_keys(
            [self._normalize_column_key(x) for x in headers]
        )
        return (headers, header_keys)
        
    def _extract_raw_rows(
        self,
        table: Table,
        header_keys: list[str],
    ) -> list[dict[str, Any]]:
        if not table.rows:
            return []

        rows: list[dict[str, Any]] = []

        for row in table.rows[1:]:
            values = [self._clean_text(cell.text) for cell in row.cells]
            if not any(values):
                continue

            row_json: dict[str, Any] = {}
            max_len = max(len(header_keys), len(values))

            for idx in range(max_len):
                key = (
                    header_keys[idx]
                    if idx < len(header_keys)
                    else f"column_{idx + 1}"
                )
                value = values[idx] if idx < len(values) else ""
                row_json[key] = value

            rows.append(row_json)

        return rows
        
    def _is_structural_numbering_row(
        self,
        *,
        row_json: dict[str, Any],
        normalized_row_json: dict[str, Any],
    ) -> bool:
        """
        Detect rows like:
        1 | 2 | 3 | 4 | 5 | 6 ...
        which appear under header captions in appendix tables.
        They are not real data rows and must not become retrieval units.
        """
        values = [
            self._clean_text(str(v))
            for v in row_json.values()
            if self._clean_text(str(v))
        ]
        if not values:
            return False

        normalized_values = [
            self._normalize_value(v)
            for v in values
            if self._normalize_value(v)
        ]

        simple_number_tokens = 0
        for value in normalized_values:
            if re.fullmatch(r"[0-9]+", value):
                simple_number_tokens += 1
                continue
            if re.fullmatch(r"[0-9]+\.[0-9]+", value):
                simple_number_tokens += 1
                continue
            if re.fullmatch(r"[ivxlcm]+", value, flags=re.IGNORECASE):
                simple_number_tokens += 1
                continue

        # Typical numbering row: almost all non-empty cells are just numeric labels.
        if len(normalized_values) >= 3 and simple_number_tokens >= max(3, len(normalized_values) - 1):
            return True

        return False
        
    def _is_service_section_row(
        self,
        *,
        row_json: dict[str, Any],
        normalized_row_json: dict[str, Any],
    ) -> bool:
        """
        Detect non-answer-bearing service rows inside tables:
        section labels, group separators, repeated explanatory headers.
        """
        values = [
            self._clean_text(str(v))
            for v in row_json.values()
            if self._clean_text(str(v))
        ]
        if not values:
            return False

        joined = " ".join(values).lower()

        service_markers = [
            "документы (информация), необходимые",
            "документы информация необходимые",
            "способы подачи запроса",
            "документы, необходимые для предоставления",
            "исчерпывающий перечень документов",
        ]

        # Strong marker-based exclusion
        if any(marker in joined for marker in service_markers):
            # But avoid excluding real row if it also clearly contains an actual document entry
            if "наименование документа" in joined:
                return False
            if "паспорт" in joined or "заявление" in joined or "документ, подтверждающий" in joined:
                return False
            return True

        # Single-cell section rows are usually service separators.
        non_empty_values = [v for v in values if v]
        if len(non_empty_values) == 1:
            only_value = non_empty_values[0].lower()
            if len(only_value) > 30 and (
                "документ" in only_value
                or "информация" in only_value
                or "необходим" in only_value
                or "предоставлен" in only_value
            ):
                return True

        return False

    def _classify_service_section_row(
        self,
        *,
        row_json: dict[str, Any],
        normalized_row_json: dict[str, Any],
    ) -> Optional[dict[str, str]]:
        """
        Пытается определить, какую именно смысловую группу открывает
        service-строка внутри таблицы.

        На текущем этапе нам нужны две основные группы:
        - required: документы / сведения, которые заявитель или представитель
          должен представить самостоятельно;
        - optional: документы / сведения, которые заявитель или представитель
          вправе представить по собственной инициативе.

        Важно:
        - сама service-строка не становится retrievable row;
        - но её смысл должен быть перенесён в metadata следующих обычных строк.
        """
        raw_values = [
            self._clean_text(str(v))
            for v in row_json.values()
            if self._clean_text(str(v))
        ]
        if not raw_values:
            return None

        # Убираем дубли одинаковых фрагментов, которые часто появляются
        # из-за merged cells / повторяющихся ячеек таблицы.
        unique_values: list[str] = []
        seen: set[str] = set()
        for value in raw_values:
            normalized_value = " ".join(value.lower().split())
            if normalized_value in seen:
                continue
            seen.add(normalized_value)
            unique_values.append(value)

        if not unique_values:
            return None

        section_label = self._clean_text(" ".join(unique_values))

        compact = section_label.lower()
        compact = compact.replace("ё", "е")
        compact = compact.replace("(", " ").replace(")", " ")
        compact = compact.replace(":", " ").replace(";", " ")
        compact = " ".join(compact.split())

        # Сначала ловим optional, потому что это самый явный маркер
        # и он не должен случайно пересечься с "самостоятельно".
        optional_markers = (
            "по собственной инициативе",
            "вправе представить по собственной инициативе",
            "вправе представить самостоятельно по собственной инициативе",
            "документы, представляемые по собственной инициативе",
            "документы и информация, которые заявитель вправе представить по собственной инициативе",
            "документы и информация, которые заявитель или представитель вправе представить по собственной инициативе",
        )
        if any(marker in compact for marker in optional_markers):
            return {
                "section_kind": "optional",
                "section_label": section_label,
            }

        # Для required делаем более широкое распознавание.
        required_markers = (
            "должен представить самостоятельно",
            "должны представить самостоятельно",
            "заявитель должен представить самостоятельно",
            "заявитель или представитель должен представить самостоятельно",
            "заявителем самостоятельно",
            "заявителем или представителем самостоятельно",
            "представляемые заявителем самостоятельно",
            "представляемые заявителем или представителем самостоятельно",
            "документы, представляемые заявителем самостоятельно",
            "документы, представляемые заявителем или представителем самостоятельно",
            "документы и информация, которые заявитель должен представить самостоятельно",
            "документы и информация, которые заявитель или представитель должен представить самостоятельно",
        )
        if any(marker in compact for marker in required_markers):
            return {
                "section_kind": "required",
                "section_label": section_label,
            }

        # Дополнительный fallback:
        # если это service-строка про "самостоятельно" и при этом в ней нет
        # маркеров "по собственной инициативе", то почти наверняка это required.
        if "самостоятельно" in compact and "по собственной инициативе" not in compact:
            return {
                "section_kind": "required",
                "section_label": section_label,
            }

        return None
        
    def _build_cells_by_header(
        self,
        *,
        headers: list[str],
        header_keys: list[str],
        row_json: dict[str, Any],
    ) -> dict[str, str]:
        result: dict[str, str] = {}

        for idx, header in enumerate(headers):
            header_key = header_keys[idx] if idx < len(header_keys) else self._normalize_column_key(header)
            value = self._clean_text(str(row_json.get(header_key, "")))
            if not value:
                continue

            label = header if header else f"Колонка {idx + 1}"
            if label in result:
                label = f"{label} [{header_key}]"

            result[label] = value

        return result
        
    def _build_cells_by_header_normalized(
        self,
        *,
        headers: list[str],
        header_keys: list[str],
        normalized_row_json: dict[str, Any],
        table_type: str,
    ) -> dict[str, str]:
        result: dict[str, str] = {}

        for idx, header in enumerate(headers):
            header_key = header_keys[idx] if idx < len(header_keys) else self._normalize_column_key(header)
            value = self._clean_text(str(normalized_row_json.get(header_key, "")))
            if not value:
                continue

            semantic_key = self._map_header_to_semantic_key(
                header=header,
                normalized_key=header_key,
                table_type=table_type,
            )
            result[semantic_key] = value

        return result
        
    def _build_cells_by_semantic_key(
        self,
        *,
        headers: list[str],
        header_keys: list[str],
        row_json: dict[str, Any],
        table_type: str,
    ) -> dict[str, str]:
        result: dict[str, str] = {}

        for idx, header in enumerate(headers):
            header_key = (
                header_keys[idx]
                if idx < len(header_keys)
                else self._normalize_column_key(header)
            )
            value = self._clean_text(str(row_json.get(header_key, "")))
            if not value:
                continue

            semantic_key = self._map_header_to_semantic_key(
                header=header,
                normalized_key=header_key,
                table_type=table_type,
            )
            result[semantic_key] = value

        if table_type == "deadlines":
            if "deadline_value" not in result:
                for raw_value in row_json.values():
                    clean_value = self._clean_text(str(raw_value))
                    if not clean_value:
                        continue
                    if self._looks_like_deadline_value(clean_value):
                        result["deadline_value"] = clean_value
                        break

            if "deadline_scope" not in result:
                scope_candidates: list[str] = []
                deadline_value = result.get("deadline_value")

                for raw_value in row_json.values():
                    clean_value = self._clean_text(str(raw_value))
                    if not clean_value:
                        continue
                    if deadline_value and clean_value == deadline_value:
                        continue
                    if self._looks_like_deadline_value(clean_value):
                        continue
                    if len(clean_value) < 3:
                        continue
                    scope_candidates.append(clean_value)

                if scope_candidates:
                    result["deadline_scope"] = max(scope_candidates, key=len)

        return result

    def _map_header_to_semantic_key(
        self,
        *,
        header: str,
        normalized_key: str,
        table_type: str,
    ) -> str:
        header_text = self._clean_text(header).lower()
        norm = self._normalize_column_key(header)

        if table_type == "documents":
            if "наименование документа" in header_text:
                return "document_name"

            if "епгу" in header_text or "единого портала" in header_text:
                return "epgu_submission"

            if "краевого портала" in header_text or "рпгу" in header_text:
                return "regional_portal_submission"

            if "лично" in header_text or "личной подаче" in header_text:
                return "in_person_submission"

            if "почтов" in header_text or "по почте" in header_text:
                return "post_submission"

            if "мфц" in header_text:
                return "mfc_submission"

            if "идентификатор" in header_text and "заявител" in header_text:
                return "applicant_category_id"

            if header_text in {"n п/п", "№ п/п", "n", "№"}:
                return "row_number"

            if normalized_key.endswith("_2"):
                return "epgu_submission"
            if normalized_key.endswith("_3"):
                return "post_submission"
            if normalized_key.endswith("_4"):
                return "mfc_submission"
            if "способ_подачи" in normalized_key:
                return "in_person_submission"

        if table_type == "identifiers":
            if "идентификатор" in header_text:
                return "applicant_category_id"
            if "категор" in header_text or "заявител" in header_text:
                return "applicant_category_name"

        if table_type == "refusal_reasons":
            if "основан" in header_text:
                return "refusal_reason"
            if "идентификатор" in header_text:
                return "applicant_category_id"

        if table_type == "deadlines":
            if (
                "срок" in header_text
                or "рабочих дней" in header_text
                or "календарных дней" in header_text
                or "не позднее" in header_text
            ):
                return "deadline_value"

            if any(
                marker in header_text
                for marker in (
                    "административная процедура",
                    "процедура",
                    "действие",
                    "этап",
                    "операция",
                    "наименование",
                    "результат",
                    "основание",
                    "событие",
                )
            ):
                return "deadline_scope"

        return norm or normalized_key or "column"

    def _normalize_detection_text(self, value: Any) -> str:
        text = self._clean_text(value)
        if not text:
            return ""
        text = text.replace("\n", " ").replace("Ё", "Е")
        text = re.sub(r"\s+", " ", text)
        return text.strip().upper()

    def _contains_detection_marker(self, text: str, marker: str) -> bool:
        if not text or not marker:
            return False

        marker = marker.upper().replace("Ё", "Е")
        # Короткие слова нельзя искать простым substring-поиском:
        # иначе "ФОРМА" срабатывает внутри "ИНФОРМАЦИИ".
        if marker in {"ФОРМА", "СОГЛАСИЕ", "РАСПИСКА", "ЗАЯВЛЕНИЕ"}:
            pattern = rf"(^|[^0-9A-ZА-Я]){re.escape(marker)}([^0-9A-ZА-Я]|$)"
            return bool(re.search(pattern, text))

        return marker in text

    def _split_title_from_semantic_start(
        self,
        paragraphs: list[str],
    ) -> tuple[list[str], Optional[str]]:
        """
        Return paragraph tail starting from the last strong semantic title start.

        This avoids titles like:
        "... свои права и исполнять свои обязанности ИСЧЕРПЫВАЮЩИЙ ПЕРЕЧЕНЬ ..."
        """
        last_idx: Optional[int] = None
        last_marker: Optional[str] = None

        for idx, paragraph in enumerate(paragraphs):
            norm = self._normalize_detection_text(paragraph)
            for marker in self._STRONG_TITLE_STARTS:
                if norm.startswith(marker):
                    last_idx = idx
                    last_marker = marker
                    break

        if last_idx is not None:
            return paragraphs[last_idx:], last_marker

        return paragraphs, None

    def _detect_normative_table_number_from_context(
        self,
        paragraph_context: list[dict[str, Any]],
    ) -> Optional[int]:
        """
        Detect the normative table number from nearby text: "Таблица 1/2/3".

        This is intentionally separate from docx_table_index/table_number:
        - docx_table_index is the physical Word table order;
        - normative_table_number is the number printed in the regulation.
        """
        for item in reversed(paragraph_context):
            text = self._clean_text(str(item.get("content_clean") or ""))
            match = re.match(r"^\s*Таблица\s+(\d+)\b", text, flags=re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    return None
        return None

    def _detect_table_title(
        self,
        *,
        paragraph_context: list[dict[str, Any]],
        fallback_number: int,
    ) -> str:
        nearby = [
            self._clean_text(item.get("content_clean") or "")
            for item in paragraph_context
            if self._clean_text(item.get("content_clean") or "")
        ]
        nearby = nearby[-14:]

        if not nearby:
            return f"Таблица {fallback_number}"

        marker_idx: Optional[int] = None
        for idx in range(len(nearby) - 1, -1, -1):
            if re.match(r"^\s*Таблица\s+\d+\b", nearby[idx], flags=re.IGNORECASE):
                marker_idx = idx
                break

        if marker_idx is not None:
            candidate_parts = nearby[marker_idx + 1 :] or [nearby[marker_idx]]
            source = "explicit_table_marker"
        else:
            candidate_parts = nearby[:]
            source = "fallback_preceding_context"

        candidate_parts, semantic_marker = self._split_title_from_semantic_start(candidate_parts)
        if semantic_marker:
            source = "semantic_heading_start"

        title = self._normalize_title(" ".join(candidate_parts))
        title_norm = self._normalize_detection_text(title)

        # If a semantic heading appears inside a long fallback fragment, cut everything before it.
        if source == "fallback_preceding_context":
            for marker in self._STRONG_TITLE_STARTS:
                pos = title_norm.find(marker)
                if pos > 0:
                    title = self._normalize_title(title[pos:])
                    title_norm = self._normalize_detection_text(title)
                    break

        # Avoid dragging a whole service-name tail into table title.
        if source.startswith("fallback") and len(title) > 260:
            title = self._normalize_title(" ".join(candidate_parts[-2:]))

        if title and not self._is_bad_title_candidate(title):
            return title

        best_keyword_candidate: Optional[str] = None
        best_plain_candidate: Optional[str] = None

        for candidate in reversed(nearby):
            if not candidate or len(candidate) > 300:
                continue
            if self._is_bad_title_candidate(candidate):
                continue

            normalized = candidate.lower()
            if any(keyword in normalized for keyword in self._GOOD_TITLE_KEYWORDS):
                best_keyword_candidate = candidate
                break

            if best_plain_candidate is None and self._looks_like_caption_or_heading(candidate):
                best_plain_candidate = candidate

        if best_keyword_candidate:
            return self._normalize_title(best_keyword_candidate)

        if best_plain_candidate:
            return self._normalize_title(best_plain_candidate)

        return f"Таблица {fallback_number}"

    def _looks_like_caption_or_heading(self, text: str) -> bool:
        cleaned = self._clean_text(text)
        lowered = cleaned.lower()

        if not cleaned or self._is_bad_title_candidate(cleaned):
            return False

        if any(keyword in lowered for keyword in self._GOOD_TITLE_KEYWORDS):
            return True

        if cleaned[:1].islower():
            return False

        return len(cleaned) <= 120

    def _is_bad_title_candidate(self, text: str) -> bool:
        cleaned = self._clean_text(text)
        lowered = cleaned.lower().strip()

        if len(lowered) < 5:
            return True

        if cleaned[:1].islower():
            return True

        for pattern in self._BAD_TITLE_PATTERNS:
            if pattern.search(cleaned):
                return True

        weak_exact = {
            "государственной услуги",
            "услуг и или",
            "или отказа",
            "предоставлении государственной услуги",
            "формате)",
        }
        if lowered in weak_exact:
            return True

        return False

    def _normalize_title(self, text: str) -> str:
        title = self._clean_text(text).replace("\n", " ")
        title = re.sub(r"\s+", " ", title)
        title = re.sub(r"^[\-\–\—\:\;\,]+", "", title).strip()
        return title or "Таблица"

    def _repair_table_title(
        self,
        *,
        table_title: str,
        table_type: str,
        headers: list[str],
        row_payloads: list[dict[str, Any]],
        normative_table_number: Optional[int],
    ) -> str:
        title_n = self._normalize_detection_text(table_title)
        header_n = self._normalize_detection_text(" ".join(headers))
        row_n = self._normalize_detection_text(
            " ".join((row.get("row_summary") or "") for row in row_payloads[:6])
        )
        table_n = " ".join([title_n, header_n, row_n])

        title_is_weak = (
            not table_title
            or table_title.startswith("Таблица ")
            or self._is_bad_title_candidate(table_title)
        )

        if table_type == "identifiers" and (
            title_is_weak
            or not any(marker in title_n for marker in ("ИДЕНТИФИКАТОРЫ КАТЕГОРИЙ", "ПРИЗНАКОВ ЗАЯВИТЕЛЕЙ"))
        ):
            return "ИДЕНТИФИКАТОРЫ КАТЕГОРИЙ (ПРИЗНАКОВ) ЗАЯВИТЕЛЕЙ"

        if table_type == "documents" and (
            title_is_weak
            or not any(marker in title_n for marker in ("ИСЧЕРПЫВАЮЩИЙ ПЕРЕЧЕНЬ", "ПЕРЕЧЕНЬ НЕОБХОДИМЫХ", "ПЕРЕЧЕНЬ ДОКУМЕНТОВ"))
        ):
            if "ПЕРЕЧЕНЬ НЕОБХОДИМЫХ" in table_n:
                return (
                    "ПЕРЕЧЕНЬ НЕОБХОДИМЫХ ДЛЯ ПРЕДОСТАВЛЕНИЯ ГОСУДАРСТВЕННОЙ "
                    "УСЛУГИ ДОКУМЕНТОВ И (ИЛИ) ИНФОРМАЦИИ"
                )
            return (
                "ИСЧЕРПЫВАЮЩИЙ ПЕРЕЧЕНЬ ДОКУМЕНТОВ, НЕОБХОДИМЫХ ДЛЯ "
                "ПРЕДОСТАВЛЕНИЯ ГОСУДАРСТВЕННОЙ УСЛУГИ"
            )

        if table_type == "refusal_reasons" and (
            title_is_weak
            or ("ПРИВЕД" in title_n and "ТАБЛИЦ" in title_n)
            or not any(marker in title_n for marker in ("ИСЧЕРПЫВАЮЩИЙ ПЕРЕЧЕНЬ ОСНОВАНИЙ", "ПЕРЕЧЕНЬ ОСНОВАНИЙ"))
        ):
            return (
                "ИСЧЕРПЫВАЮЩИЙ ПЕРЕЧЕНЬ ОСНОВАНИЙ ДЛЯ ОТКАЗА В ПРИЕМЕ ЗАПРОСА "
                "И ДОКУМЕНТОВ, НЕОБХОДИМЫХ ДЛЯ ПРЕДОСТАВЛЕНИЯ ГОСУДАРСТВЕННОЙ "
                "УСЛУГИ, ОСНОВАНИЙ ДЛЯ ПРИОСТАНОВЛЕНИЯ ПРЕДОСТАВЛЕНИЯ "
                "ГОСУДАРСТВЕННОЙ УСЛУГИ ИЛИ ОТКАЗА В ПРЕДОСТАВЛЕНИИ "
                "ГОСУДАРСТВЕННОЙ УСЛУГИ"
            )

        return table_title

    def _detect_table_type(
        self,
        *,
        table_title: str,
        headers: list[str],
        row_payloads: list[dict[str, Any]],
        normative_table_number: Optional[int] = None,
    ) -> str:
        title_n = self._normalize_detection_text(table_title)
        header_n = self._normalize_detection_text(" ".join(headers))
        row_summaries = " ".join(
            (row.get("row_summary") or "")
            for row in row_payloads[:12]
        )
        row_values = " ".join(
            " ".join(
                self._clean_text(str(value))
                for value in (row.get("row_json") or {}).values()
                if self._clean_text(str(value))
            )
            for row in row_payloads[:12]
        )
        text_n = self._normalize_detection_text(f"{row_summaries} {row_values}")
        table_only_n = " ".join([title_n, header_n, text_n])

        rows_count = len(row_payloads)
        cols_count = max((len((row.get("row_json") or {})) for row in row_payloads), default=len(headers))
        total_cells = 0
        empty_cells = 0
        for row in row_payloads[:20]:
            row_json = row.get("row_json") or {}
            for value in row_json.values():
                total_cells += 1
                if not self._clean_text(str(value)):
                    empty_cells += 1

        scores: dict[str, float] = {
            "consultant_noise": 0.0,
            "form_fields": 0.0,
            "identifiers": 0.0,
            "documents": 0.0,
            "refusal_reasons": 0.0,
            "deadlines": 0.0,
            "amounts": 0.0,
            "generic": 0.1,
        }

        # Служебный шум. Смотрим только саму таблицу и её заголовок, не весь соседний контекст.
        for marker in self._NOISE_TABLE_MARKERS:
            if marker in table_only_n:
                scores["consultant_noise"] += 2.5
        if rows_count <= 2 and "СПИСОК ИЗМЕНЯЮЩИХ ДОКУМЕНТОВ" in table_only_n:
            scores["consultant_noise"] += 2.5

        # Формы / шаблоны заявлений.
        for marker in self._FORM_TITLE_MARKERS:
            if self._contains_detection_marker(title_n, marker):
                scores["form_fields"] += 2.0
            elif self._contains_detection_marker(header_n, marker):
                scores["form_fields"] += 1.2
            elif self._contains_detection_marker(text_n, marker):
                scores["form_fields"] += 0.8

        for marker in self._FORM_CONTEXT_MARKERS:
            if self._contains_detection_marker(table_only_n, marker):
                scores["form_fields"] += 0.5

        if total_cells >= 12 and empty_cells / max(total_cells, 1) > 0.35:
            scores["form_fields"] += 0.6
        if "ПОДПИСЬ" in table_only_n and "ДАТА" in table_only_n and "ФАМИЛ" in table_only_n:
            scores["form_fields"] += 1.2
        if (
            "НАИМЕНОВАНИЕ ДОКУМЕНТА" in header_n
            and ("КОЛ-ВО" in header_n or "КОЛИЧЕСТВО" in header_n or "КОЛ-ВО" in text_n or "КОЛИЧЕСТВО" in text_n)
            and not any(marker in header_n for marker in ("СПОСОБ ПОДАЧИ", "СПОСОБЫ ПОДАЧИ", "ЕДИНЫЙ ПОРТАЛ", "ЕПГУ", "МФЦ"))
        ):
            scores["form_fields"] += 2.2
        if (
            normative_table_number is None
            and "НАИМЕНОВАНИЕ ДОКУМЕНТА" in header_n
            and not any(marker in header_n for marker in ("СПОСОБ ПОДАЧИ", "СПОСОБЫ ПОДАЧИ", "ЕДИНЫЙ ПОРТАЛ", "ЕПГУ", "МФЦ"))
            and (rows_count <= 6 or cols_count <= 4)
        ):
            scores["form_fields"] += 1.8
        if re.match(r"^\d+\.", title_n) and ("СВЕДЕНИЯ" in title_n or "НАИМЕНОВАНИЕ ДОКУМЕНТА" in header_n):
            scores["form_fields"] += 1.2

        # Категории заявителей.
        for marker in self._CATEGORY_MARKERS:
            if marker in title_n:
                scores["identifiers"] += 2.0
            elif marker in header_n:
                scores["identifiers"] += 1.2
            elif marker in text_n:
                scores["identifiers"] += 0.6
        if "ПЕРЕЧЕНЬ РЕЗУЛЬТАТОВ" in header_n and ("А1" in text_n or "Б1" in text_n):
            scores["identifiers"] += 0.8
        if normative_table_number == 1 and ("ПРИЗНАК" in header_n or "ЗАЯВИТЕЛ" in header_n):
            scores["identifiers"] += 1.2

        # Таблица документов.
        strong_documents_title = any(
            marker in title_n
            for marker in (
                "ИСЧЕРПЫВАЮЩИЙ ПЕРЕЧЕНЬ ДОКУМЕНТОВ",
                "ПЕРЕЧЕНЬ НЕОБХОДИМЫХ ДЛЯ ПРЕДОСТАВЛЕНИЯ",
                "ПЕРЕЧЕНЬ ДОКУМЕНТОВ",
                "ДОКУМЕНТОВ, НЕОБХОДИМЫХ ДЛЯ ПРЕДОСТАВЛЕНИЯ",
            )
        )
        header_or_text_n = f"{header_n} {text_n}"
        documents_header = (
            "НАИМЕНОВАНИЕ ДОКУМЕНТА" in header_or_text_n
            or "ПЕРЕЧЕНЬ НЕОБХОДИМЫХ" in header_or_text_n
            or "ДОКУМЕНТОВ И (ИЛИ) ИНФОРМАЦИИ" in header_or_text_n
        )
        submission_header = any(
            marker in header_or_text_n
            for marker in (
                "СПОСОБ ПОДАЧИ",
                "СПОСОБЫ ПОДАЧИ",
                "ЕДИНЫЙ ПОРТАЛ",
                "ЕПГУ",
                "РПГУ",
                "МФЦ",
                "ПОЧТОВЫМ ОТПРАВЛЕНИЕМ",
                "ЛИЧНО",
            )
        )
        category_columns = "ИДЕНТИФИКАТОРЫ КАТЕГОРИЙ" in header_or_text_n or "ПРИЗНАКОВ ЗАЯВИТЕЛЕЙ" in header_or_text_n
        if (
            not documents_header
            and strong_documents_title
            and cols_count >= 5
            and rows_count >= 3
            and (submission_header or "ЛИЧНО" in header_or_text_n or "ПС" in header_or_text_n or "ЭД" in header_or_text_n)
        ):
            # Some appendix tables have merged header rows, so _extract_headers() falls back
            # to generic columns. The normative title + wide table + submission markers are
            # still a strong documents-table signal.
            documents_header = True

        for marker in self._DOCUMENTS_MARKERS:
            if marker in title_n:
                scores["documents"] += 2.0
            elif marker in header_n:
                scores["documents"] += 1.0
        if submission_header:
            scores["documents"] += 1.1
        if category_columns and documents_header:
            scores["documents"] += 1.0
        if normative_table_number == 2 and (strong_documents_title or documents_header):
            scores["documents"] += 1.5
        if strong_documents_title and documents_header and (submission_header or category_columns):
            scores["documents"] += 2.0

        # Основания отказа / приостановления.
        strong_refusal_title = (
            "ИСЧЕРПЫВАЮЩИЙ ПЕРЕЧЕНЬ ОСНОВАНИЙ" in title_n
            or "ПЕРЕЧЕНЬ ОСНОВАНИЙ" in title_n
        )
        weak_refusal_reference = "ПРИВЕД" in title_n and "ТАБЛИЦ" in title_n
        for marker in self._REFUSAL_MARKERS:
            if marker in title_n:
                scores["refusal_reasons"] += 2.0 if strong_refusal_title and not weak_refusal_reference else 0.25
            elif marker in header_n:
                scores["refusal_reasons"] += 1.1
            elif marker in text_n:
                scores["refusal_reasons"] += 0.6
        if "ПЕРЕЧЕНЬ ОСНОВАНИЙ" in header_n:
            scores["refusal_reasons"] += 1.1
        if normative_table_number == 3 and (
            "ОТКАЗ" in table_only_n or "ПРИОСТАНОВ" in table_only_n or "ОСНОВАН" in table_only_n
        ):
            scores["refusal_reasons"] += 1.4

        # Сроки.
        deadline_title_markers = [
            "срок предоставления",
            "максимальный срок",
            "срок регистрации",
            "срок исправления",
            "срок ожидания",
            "срок принятия решения",
            "срок направления",
            "срок выполнения",
        ]
        haystack_lower = f"{table_title} {' '.join(headers)}".lower()
        deadline_title_score = sum(
            1 for marker in deadline_title_markers if marker in haystack_lower
        )
        deadline_row_like_count = sum(
            1
            for row in row_payloads[:12]
            if self._row_looks_like_deadline_payload(row)
        )
        if deadline_title_score >= 1 and deadline_row_like_count >= 1:
            scores["deadlines"] += 1.5
        if deadline_title_score >= 2:
            scores["deadlines"] += 1.0
        if deadline_row_like_count >= 2:
            scores["deadlines"] += 1.0

        # Размеры.
        if any(marker in title_n for marker in self._AMOUNT_MARKERS) and ("РУБ" in table_only_n or "РАЗМЕР" in table_only_n):
            scores["amounts"] += 1.2

        strong_real_documents_table = (
            scores["documents"] >= 3.0
            and documents_header
            and (strong_documents_title or normative_table_number == 2)
            and (submission_header or category_columns or cols_count >= 6)
            and scores["consultant_noise"] < 2.0
        )
        form_over_documents = (
            scores["form_fields"] >= 2.0
            and not strong_real_documents_table
            and scores["form_fields"] >= scores["documents"] - 0.2
        )
        has_refusal_structure = (
            normative_table_number == 3
            or "ПЕРЕЧЕНЬ ОСНОВАНИЙ" in header_n
            or "ПЕРЕЧЕНЬ ОСНОВАНИЙ" in text_n
            or "ОСНОВАНИЙ ДЛЯ ОТКАЗА" in header_n
            or "ОСНОВАНИЙ ДЛЯ ОТКАЗА" in text_n
        )
        strong_form_like = (
            scores["form_fields"] >= 2.0
            and not strong_real_documents_table
            and not has_refusal_structure
        )
        weak_link_to_another_table = (
            "ПРИВЕД" in title_n
            and "ТАБЛИЦ" in title_n
            and normative_table_number is None
            and rows_count <= 2
        )

        if scores["consultant_noise"] >= 2.0:
            return "consultant_noise"
        if weak_link_to_another_table:
            return "generic"
        if strong_form_like:
            return "form_fields"
        if scores["refusal_reasons"] >= max(scores["documents"], scores["identifiers"], 1.5):
            return "refusal_reasons"
        if strong_real_documents_table or scores["documents"] >= max(scores["identifiers"], scores["form_fields"] + 0.5, 1.5):
            return "documents"
        if form_over_documents:
            return "form_fields"
        if scores["identifiers"] >= 1.5:
            return "identifiers"
        if scores["deadlines"] >= 1.5:
            return "deadlines"
        if scores["amounts"] >= 1.2:
            return "amounts"

        return "generic"
        
    def _looks_like_form_headers(
        self,
        headers: list[str],
    ) -> bool:
        norm_headers = [self._normalize_search_text(h) for h in headers if h]
        if not norm_headers:
            return False

        form_markers = (
            "фамилия",
            "имя",
            "отчество",
            "дата рождения",
            "адрес",
            "виды дохода",
            "документ",
            "подпись",
            "дата",
            "номер",
        )

        hits = 0
        for header in norm_headers:
            if any(marker in header for marker in form_markers):
                hits += 1

        return hits >= 2

    def _looks_like_form_table(
        self,
        *,
        table_title: str,
        headers: list[str],
        row_payloads: list[dict[str, Any]],
    ) -> bool:
        table_type = self._detect_table_type(
            table_title=table_title,
            headers=headers,
            row_payloads=row_payloads,
            normative_table_number=None,
        )
        return table_type == "form_fields"
        
    def _row_looks_like_deadline_payload(
        self,
        row: dict[str, Any],
    ) -> bool:
        row_json = row.get("row_json") or {}
        values = [
            self._clean_text(str(v))
            for v in row_json.values()
            if self._clean_text(str(v))
        ]
        if not values:
            return False

        deadline_values = [v for v in values if self._looks_like_deadline_value(v)]
        if not deadline_values:
            return False

        non_deadline_values = [
            v.lower()
            for v in values
            if v not in deadline_values
        ]

        scope_markers = (
            "решени",
            "предоставлен",
            "уведом",
            "выплат",
            "регистрац",
            "рассмотрен",
            "исправлен",
            "ожидани",
            "приостанов",
            "направлен",
            "подписани",
        )

        if any(
            any(marker in value for marker in scope_markers)
            for value in non_deadline_values
        ):
            return True

        return len(values) <= 3

    def _looks_like_deadline_value(
        self,
        text: str,
    ) -> bool:
        normalized = self._clean_text(text).lower().replace("ё", "е")
        if not normalized:
            return False

        patterns = (
            r"\b\d+\s+(?:рабоч(?:их|его)?|календарн(?:ых|ого)?)\s+дн",
            r"\b\d+\s+дн",
            r"в течение\s+\d+\s+(?:рабоч(?:их|его)?|календарн(?:ых|ого)?)\s+дн",
            r"не более\s+\d+\s+(?:рабоч(?:их|его)?|календарн(?:ых|ого)?)\s+дн",
            r"не позднее\s+\d{1,2}(?:-го)?\s+числа",
            r"в день регистрации",
            r"в день поступления",
            r"в день принятия решения",
            r"ежемесячно",
        )

        return any(re.search(pattern, normalized) for pattern in patterns)

    def _build_table_summary(
        self,
        *,
        table_title: str,
        headers: list[str],
        rows_count: int,
        table_type: str,
    ) -> str:
        if headers:
            return (
                f"{table_title}. "
                f"Тип таблицы: {table_type}. "
                f"Колонки: {', '.join(headers)}. "
                f"Количество строк: {rows_count}."
            )
        return (
            f"{table_title}. "
            f"Тип таблицы: {table_type}. "
            f"Количество строк: {rows_count}."
        )

    def _build_row_summary(
        self,
        *,
        table_title: str,
        headers: list[str],
        row_json: dict[str, Any],
    ) -> str:
        parts: list[str] = []

        if table_title:
            parts.append(f"Таблица: {table_title}.")

        if headers:
            pretty_headers = [h for h in headers if not self._is_noise_header(h)]
            if pretty_headers:
                parts.append(f"Колонки таблицы: {', '.join(pretty_headers)}.")

        for key, value in row_json.items():
            clean_value = self._clean_text(str(value))
            if not clean_value:
                continue
            if self._is_noise_key(key):
                continue
            parts.append(f"{self._pretty_label(key)}: {clean_value}.")

        return " ".join(parts).strip()

    def _render_markdown_preview(
        self,
        *,
        headers: list[str],
        header_keys: list[str],
        rows: list[dict[str, Any]],
    ) -> Optional[str]:
        if not headers:
            return None

        preview_rows = rows[: self.max_markdown_preview_rows]
        if not preview_rows:
            return None

        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]

        for row in preview_rows:
            row_json = row.get("row_json") or {}
            values = []
            for idx, header in enumerate(headers):
                key = header_keys[idx] if idx < len(header_keys) else self._normalize_column_key(header)
                value = self._clean_text(str(row_json.get(key, "")))
                values.append(value.replace("|", "\\|"))
            lines.append("| " + " | ".join(values) + " |")

        return "\n".join(lines)

    def _detect_document_title(
        self,
        *,
        original_filename: str,
        blocks: list[dict[str, Any]],
    ) -> str:
        texts = [
            self._clean_text(str(block.get("content_clean") or ""))
            for block in blocks[:80]
            if self._is_meaningful_text(block.get("content_clean"))
        ]

        if not texts:
            return Path(original_filename).stem

        # 1. Самый надёжный случай:
        # если в приложении явно найден заголовок самого административного регламента,
        # берём именно его, а не wrapper-title приказа.
        appendix_regulation_title = self._extract_regulation_title_from_appendix(texts)
        if appendix_regulation_title:
            return appendix_regulation_title

        # 2. Иначе пробуем собрать title из верхней части приказа,
        # но уже без шума, даты, подписи и служебных строк.
        start_idx = 0
        for idx, text in enumerate(texts[:15]):
            if text.lower() == "приказ":
                start_idx = idx + 1
                break

        title_lines: list[str] = []

        for text in texts[start_idx : start_idx + 18]:
            lowered = text.lower()

            if self._looks_like_signature_line(text):
                if title_lines:
                    break
                continue

            if any(marker in lowered for marker in self._TITLE_STOP_MARKERS):
                if title_lines:
                    break
                continue

            if self._is_noise_document_title_line(text):
                if title_lines:
                    break
                continue

            if not title_lines:
                if self._looks_like_document_title_line(text):
                    title_lines.append(text)
                continue

            if self._looks_like_document_title_line(text):
                title_lines.append(text)
                continue

            break

        if title_lines:
            return self._normalize_title(" ".join(title_lines))

        # 3. Осторожный fallback: первый содержательный block,
        # который не шум, не дата, не подпись.
        for text in texts[:30]:
            if self._is_noise_document_title_line(text):
                continue
            if self._looks_like_signature_line(text):
                continue
            if len(text) < 5:
                continue
            return text

        return Path(original_filename).stem
        
    def _extract_regulation_title_from_appendix(
        self,
        texts: list[str],
    ) -> Optional[str]:
        appendix_seen = False
        collecting = False
        collected: list[str] = []

        for text in texts[:80]:
            lowered = text.lower()

            if lowered == "приложение" or lowered.startswith("приложение"):
                appendix_seen = True
                if collected:
                    break
                continue

            if not appendix_seen:
                continue

            if self._looks_like_signature_line(text):
                if collected:
                    break
                continue

            if self._is_noise_document_title_line(text):
                if collected:
                    continue
                continue

            if "административный регламент" in lowered:
                collecting = True
                collected.append(text)
                continue

            if not collecting:
                continue

            if any(marker in lowered for marker in self._TITLE_STOP_MARKERS):
                break

            if self._looks_like_document_title_line(text):
                collected.append(text)
                continue

            break

        if not collected:
            return None

        return self._normalize_title(" ".join(collected))
        
    def _looks_like_signature_line(
        self,
        text: str,
    ) -> bool:
        clean_text = self._clean_text(text)
        lowered = clean_text.lower()

        if lowered in {
            "министр",
            "первый заместитель министра",
            "заместитель министра",
        }:
            return True

        # И.Л.ПАСТУХОВА / О.Н.ЧЕРНЫШЕВА / Д.В. БОГДАНОВ и т.п.
        if re.fullmatch(r"[А-ЯЁ]\.[А-ЯЁ]\.[А-ЯЁ\-]+", clean_text):
            return True

        # Иногда после OCR/нормализации бывают пробелы: И. Л. ПАСТУХОВА
        if re.fullmatch(r"[А-ЯЁ]\.\s*[А-ЯЁ]\.\s*[А-ЯЁ\-]+", clean_text):
            return True

        return False

    def _collect_leading_texts(
        self,
        blocks: list[dict[str, Any]],
        *,
        limit: int = 120,
    ) -> list[str]:
        return [
            self._clean_text(str(block.get("content_clean") or ""))
            for block in blocks[:limit]
            if self._is_meaningful_text(block.get("content_clean"))
        ]

    def _extract_document_number(
        self,
        text: str,
    ) -> Optional[str]:
        clean_text = self._clean_text(text)
        if not clean_text:
            return None

        match = self._NUMBER_RE.search(clean_text)
        if match is None:
            return None

        number = self._clean_text(match.group("number"))
        number = number.replace(" ", "")
        return number or None

    def _extract_document_date_and_number(
        self,
        *,
        original_filename: str,
        blocks: list[dict[str, Any]],
    ) -> tuple[Optional[str], Optional[datetime]]:
        texts = [original_filename, *self._collect_leading_texts(blocks, limit=80)]

        for text in texts[:25]:
            clean_text = self._clean_text(text)
            if not clean_text:
                continue

            dates = self._extract_candidate_dates(clean_text)
            number = self._extract_document_number(clean_text)

            lowered = clean_text.lower()
            looks_like_header = (
                ("приказ" in lowered)
                or lowered.startswith("от ")
                or ("№" in clean_text)
                or (" N " in clean_text)
                or (" N" in clean_text)
            )

            if looks_like_header and dates and number:
                return number, dates[0]

        for text in texts[:25]:
            clean_text = self._clean_text(text)
            if not clean_text:
                continue

            number = self._extract_document_number(clean_text)
            if number:
                dates = self._extract_candidate_dates(clean_text)
                return number, (dates[0] if dates else None)

        return None, None

    def _strip_clause_prefix(
        self,
        text: str,
    ) -> str:
        clean_text = self._clean_text(text)
        match = self._CLAUSE_RE.match(clean_text)
        if match:
            return self._clean_text(match.group("text"))
        return clean_text

    def _looks_like_service_name_candidate(
        self,
        text: str,
    ) -> bool:
        clean_text = self._clean_text(text)
        lowered = clean_text.lower()

        if len(clean_text) < 15 or len(clean_text) > 500:
            return False

        strong_markers = (
            "предоставление",
            "назначение",
            "ежемесячной денежной выплаты",
            "субсид",
            "социального контракта",
            "адресной материальной помощи",
            "санаторно-курорт",
            "бесплатных путевок",
            "государственной социальной помощи",
        )
        return any(marker in lowered for marker in strong_markers)

    def _detect_service_name_full(
        self,
        *,
        blocks: list[dict[str, Any]],
    ) -> Optional[str]:
        anchor_seen = False

        for block in blocks[:260]:
            text = self._clean_text(str(block.get("content_clean") or ""))
            if not text:
                continue

            lowered = text.lower()
            block_type = self._clean_text(block.get("block_type") or "").lower()

            if "наименование государственной услуги" in lowered:
                anchor_seen = True
                continue

            if not anchor_seen:
                continue

            if block_type == "heading":
                break

            candidate = self._strip_clause_prefix(text).rstrip(" .;:")

            if block.get("clause_number"):
                if self._looks_like_service_name_candidate(candidate):
                    return candidate

            if self._looks_like_service_name_candidate(candidate):
                return candidate

        return None

    def _detect_revision_date(
        self,
        *,
        original_filename: str,
        blocks: list[dict[str, Any]],
        normalized_text: str,
        document_date: Optional[datetime] = None,
    ) -> Optional[datetime]:
        texts = self._collect_leading_texts(blocks, limit=120)

        for text in texts[:30]:
            lowered = text.lower()
            if "в ред." in lowered or "в ред " in lowered:
                text_dates = self._extract_candidate_dates(text)
                if text_dates:
                    return text_dates[0]

        for idx, text in enumerate(texts[:40]):
            lowered = text.lower()
            if "список изменяющих документов" not in lowered:
                continue

            for nearby in texts[idx + 1 : idx + 5]:
                nearby_lowered = nearby.lower()
                if "в ред." in nearby_lowered or "в ред " in nearby_lowered:
                    nearby_dates = self._extract_candidate_dates(nearby)
                    if nearby_dates:
                        return nearby_dates[0]

        if document_date is not None:
            return document_date

        for text in texts[:12]:
            if self._looks_like_order_date_line(text):
                text_dates = self._extract_candidate_dates(text)
                if text_dates:
                    return text_dates[0]

        filename_dates = self._extract_candidate_dates(original_filename)
        if filename_dates:
            return filename_dates[0]

        prefix_dates = self._extract_candidate_dates(
            self._clean_text(normalized_text[:1200])
        )
        if prefix_dates:
            return prefix_dates[0]

        return None
        
    def _extract_candidate_dates(
        self,
        text: str,
    ) -> list[datetime]:
        clean_text = self._clean_text(text)
        if not clean_text:
            return []

        matches: list[tuple[int, datetime]] = []

        for match in self._TEXTUAL_DATE_RE.finditer(clean_text):
            day = int(match.group("day"))
            month_name = match.group("month").lower()
            year = int(match.group("year"))
            month = self._RUS_MONTHS.get(month_name)
            if month is None:
                continue
            try:
                parsed = datetime(year, month, day, tzinfo=timezone.utc)
                matches.append((match.start(), parsed))
            except ValueError:
                continue

        for match in self._REVISION_DATE_RE.finditer(clean_text):
            try:
                parsed = datetime.strptime(match.group("date"), "%d.%m.%Y")
                matches.append((match.start(), parsed.replace(tzinfo=timezone.utc)))
            except ValueError:
                continue

        matches.sort(key=lambda item: item[0])

        result: list[datetime] = []
        seen: set[str] = set()

        for _, parsed in matches:
            key = parsed.strftime("%Y-%m-%d")
            if key in seen:
                continue
            seen.add(key)
            result.append(parsed)

        return result

    def _looks_like_order_date_line(
        self,
        text: str,
    ) -> bool:
        clean_text = self._clean_text(text)
        lowered = clean_text.lower()

        if not self._extract_candidate_dates(clean_text):
            return False

        if lowered.startswith("от ") and ("№" in clean_text or re.search(r"\bN\b", clean_text)):
            return True

        if "приказ" in lowered and self._extract_candidate_dates(clean_text):
            return True

        return False

    def _looks_like_document_title_line(
        self,
        text: str,
    ) -> bool:
        clean_text = self._clean_text(text)
        if len(clean_text) < 5 or len(clean_text) > 250:
            return False

        lowered = clean_text.lower()

        strong_markers = (
            "об утверждении",
            "о внесении изменения",
            "о внесении изменений",
            "административного регламента предоставления",
            "государственной услуги",
            "предоставлению",
            "назначению",
        )
        if any(marker in lowered for marker in strong_markers):
            return True

        letters = [ch for ch in clean_text if ch.isalpha()]
        if not letters:
            return False

        upper_ratio = sum(1 for ch in letters if ch == ch.upper()) / len(letters)
        return upper_ratio >= 0.75 and len(clean_text) >= 12

    def _is_noise_document_title_line(
        self,
        text: str,
    ) -> bool:
        clean_text = self._clean_text(text)
        lowered = clean_text.lower()

        if lowered in self._TITLE_SKIP_EXACT:
            return True

        if self._looks_like_signature_line(clean_text):
            return True

        if lowered.startswith("документ предоставлен"):
            return True
        if lowered.startswith("дата сохранения"):
            return True
        if lowered.startswith("www."):
            return True
        if lowered.startswith("к приказу"):
            return True
        if lowered.startswith("министерство социальной политики"):
            return True
        if lowered == "красноярского края":
            return True

        if lowered.startswith("от ") and self._extract_candidate_dates(clean_text):
            return True

        if self._extract_candidate_dates(clean_text) and (
            "№" in clean_text or re.search(r"\bN\b", clean_text)
        ):
            return True

        return False

    def _detect_doc_uid_base(
        self,
        *,
        original_filename: str,
        document_title: str,
        normalized_text: str,
        service_name_full: Optional[str] = None,
        document_number: Optional[str] = None,
        document_date: Optional[datetime] = None,
    ) -> Optional[str]:
        identity_basis = (
            service_name_full
            or document_title
            or Path(original_filename).stem
            or ""
        )
        identity_token = self._normalize_token(identity_basis)[:160]
        if not identity_token:
            return None

        suffix_parts: list[str] = []

        if document_number:
            normalized_number = self._normalize_token(document_number)
            if normalized_number:
                suffix_parts.append(normalized_number)

        if not suffix_parts and document_date is not None:
            suffix_parts.append(document_date.strftime("%Y_%m_%d"))

        if not suffix_parts:
            return identity_token

        return f"{identity_token}__{'__'.join(suffix_parts)}"

    def _detect_appendix_number_from_context(
        self,
        paragraph_context: list[dict[str, Any]],
    ) -> Optional[str]:
        for item in reversed(paragraph_context):
            text = self._clean_text(str(item.get("content_clean") or ""))
            match = self._APPENDIX_RE.match(text)
            if match:
                return self._clean_text(match.group("num") or "")
        return None

    def _clean_text(self, value: Any) -> str:
        if value is None:
            return ""
        text = str(value).replace("\xa0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\s*\n\s*", "\n", text)
        return text.strip()

    def _normalize_column_key(self, value: str) -> str:
        normalized = self._normalize_token(value)
        return normalized or "column"

    def _normalize_value(self, value: Any) -> str:
        return self._normalize_token(str(value))

    def _normalize_token(self, value: str) -> str:
        cleaned = self._clean_text(value).lower()
        cleaned = cleaned.replace("№", "n")
        cleaned = re.sub(r"[^0-9a-zа-я]+", "_", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"_+", "_", cleaned)
        return cleaned.strip("_")

    def _make_unique_keys(self, keys: list[str]) -> list[str]:
        result: list[str] = []
        seen: dict[str, int] = {}

        for idx, key in enumerate(keys, start=1):
            base = key or f"column_{idx}"
            if base not in seen:
                seen[base] = 1
                result.append(base)
                continue

            seen[base] += 1
            result.append(f"{base}_{seen[base]}")

        return result

    def _pretty_label(self, key: str) -> str:
        pretty = str(key).replace("_", " ").strip()
        if not pretty:
            return "Поле"
        return pretty[:1].upper() + pretty[1:]

    def _is_meaningful_text(self, value: Any) -> bool:
        text = self._clean_text(value)
        return bool(text and len(text) >= 2)

    def _is_effectively_empty_row(self, row_json: dict[str, Any]) -> bool:
        meaningful_values = [
            self._clean_text(str(v))
            for v in row_json.values()
            if self._clean_text(str(v))
        ]
        return len(meaningful_values) == 0

    def _looks_like_broken_header(self, headers: list[str]) -> bool:
        if not headers:
            return False

        long_headers = [h for h in headers if len(self._clean_text(h)) > 80]
        if len(long_headers) >= 1:
            return True

        repeated_admin = sum(
            1 for h in headers if "административный регламент" in h.lower()
        )
        if repeated_admin >= 2:
            return True

        return False

    def _is_noise_header(self, header: str) -> bool:
        normalized = self._clean_text(header).lower()
        if normalized in {"-", "column", "колонка", "колонка 1", "колонка 2", "колонка 3", "колонка 4", "колонка 5"}:
            return True
        return False

    def _is_noise_key(self, key: str) -> bool:
        normalized = self._clean_text(key).lower()
        if normalized in {"column", "column_1", "column_2", "column_3", "column_4", "column_5"}:
            return True
        return False
        
    def _update_heading_path(
        self,
        *,
        heading_path: list[str],
        heading_text: str,
        section_number: Optional[str],
    ) -> list[str]:
        clean_heading = self._clean_text(heading_text)
        if not clean_heading:
            return list(heading_path)

        if section_number and re.fullmatch(r"[IVXLCM]+\.?", section_number, flags=re.IGNORECASE):
            return [clean_heading]

        new_path = list(heading_path)
        if not new_path or new_path[-1] != clean_heading:
            new_path.append(clean_heading)
        return new_path[-4:]

    def _infer_block_semantic_hints(
        self,
        *,
        text: str,
        current_heading_text: Optional[str],
        heading_path: list[str],
    ) -> dict[str, Any]:
        haystack_parts = [self._clean_text(text)]
        if current_heading_text:
            haystack_parts.append(self._clean_text(current_heading_text))
        haystack_parts.extend(self._clean_text(x) for x in heading_path if self._clean_text(x))

        haystack = " ".join(haystack_parts).lower()

        hints = {
            "is_deadline_related": False,
            "deadline_kind_hint": None,
        }

        if any(
            marker in haystack
            for marker in (
                "срок предоставления государственной услуги",
                "срок предоставления",
                "срок регистрации",
                "срок исправления",
                "в течение 2 рабочих дней",
                "не позднее 26-го числа",
                "не позднее 26 числа",
                "уведомляется",
                "направляется заявителю",
                "выплачивается",
                "решение принимается",
            )
        ):
            hints["is_deadline_related"] = True

        if any(marker in haystack for marker in ("решение принимается", "принятия решения", "срок предоставления")):
            hints["deadline_kind_hint"] = "decision"
        elif any(marker in haystack for marker in ("уведомляется", "уведомление", "направляется заявителю")):
            hints["deadline_kind_hint"] = "notification"
        elif any(marker in haystack for marker in ("выплачивается", "выплата", "не позднее 26-го числа", "не позднее 26 числа")):
            hints["deadline_kind_hint"] = "payment"

        return hints