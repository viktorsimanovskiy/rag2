from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from docx import Document as DocxDocument
from docx.document import Document as _Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph


CORE_TABLE_TYPES = {"identifiers", "documents", "refusal_reasons"}
OFFICIAL_START_MARKER = "МИНИСТЕРСТВО СОЦИАЛЬНОЙ ПОЛИТИКИ"
PREPROCESSOR_VERSION = "consultant_plus_docx_preprocessor_v1_dry_run"

CONSULTANT_NOISE_MARKERS = (
    "КОНСУЛЬТАНТПЛЮС",
    "ДОКУМЕНТ ПРЕДОСТАВЛЕН КОНСУЛЬТАНТПЛЮС",
    "WWW.CONSULTANT.RU",
    "СПИСОК ИЗМЕНЯЮЩИХ ДОКУМЕНТОВ",
    "НУМЕРАЦИЯ ПУНКТОВ ДАНА В СООТВЕТСТВИИ",
)

FORM_OR_TEMPLATE_MARKERS = (
    "ЗАЯВЛЕНИЕ",
    "СОГЛАСИЕ",
    "РАСПИСК",
    "УВЕДОМЛЕНИЕ",
    "ПОДПИСЬ ЗАЯВИТЕЛ",
    "ДОСТОВЕРНОСТЬ И ПОЛНОТУ СВЕДЕНИЙ",
    "ПРИНЯЛ ДОКУМЕНТЫ",
    "ПРИЛАГАЮ СЛЕДУЮЩИЕ ДОКУМЕНТЫ",
    "К ЗАЯВЛЕНИЮ ПРИЛАГАЮ",
    "ПЕРЕЧЕНЬ ПРИЛАГАЕМЫХ ДОКУМЕНТОВ",
    "СВЕДЕНИЯ О ДОКУМЕНТЕ",
    "СВЕДЕНИЯ О ДОКУМЕНТАХ",
    "СВЕДЕНИЯ О ПРЕДСТАВИТЕЛЕ",
    "СВЕДЕНИЯ О ЗАКОННОМ ПРЕДСТАВИТЕЛЕ",
    "БАНКОВСКИЕ РЕКВИЗИТЫ",
    "ИНДИВИДУАЛЬНЫЙ ЛИЦЕВОЙ СЧЕТ",
)


@dataclass(slots=True)
class DocxOrderedItem:
    seq: int
    kind: str
    text: str
    table_index: Optional[int] = None
    rows_count: Optional[int] = None
    table_type: Optional[str] = None
    category: Optional[str] = None


@dataclass(slots=True)
class DocxPreprocessingReport:
    preprocessor_version: str
    filename: str
    file_size_bytes: int
    file_sha256: str
    total_items_count: int
    total_paragraphs_count: int
    total_tables_count: int
    official_start_seq: Optional[int]
    items_before_official_start_count: int
    official_start_found: bool
    core_table_counts: dict[str, int]
    core_table_indexes: dict[str, list[int]]
    has_all_core_tables: bool
    has_exactly_one_each_core_table: bool
    last_core_table_seq: Optional[int]
    last_core_table_index: Optional[int]
    last_core_table_type: Optional[str]
    trim_after_last_core_table_candidate: bool
    kept_items_count: int
    kept_paragraphs_count: int
    kept_tables_count: int
    removed_before_official_start_count: int
    removed_consultant_noise_count: int
    trimmed_after_last_core_count: int
    trimmed_after_last_core_paragraphs_count: int
    trimmed_after_last_core_tables_count: int
    trimmed_after_last_core_chars_count: int
    trimmed_tail_table_type_counts: dict[str, int]
    trimmed_tail_category_counts: dict[str, int]
    tail_contains_core_table: bool
    prepared_text_hash: str
    prepared_text_chars_count: int
    first_trimmed_samples: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DocxPreprocessingResult:
    source_path: Path
    prepared_text: str
    kept_items: list[DocxOrderedItem]
    trimmed_items: list[DocxOrderedItem]
    report: DocxPreprocessingReport


class ConsultantPlusDocxPreprocessor:
    """
    Dry-run DOCX preprocessor for ConsultantPlus exports.

    This class intentionally does not mutate DOCX files. It models what the
    future ingestion preparation layer should keep for RAG:
    - content from the official start of the order;
    - all content up to and including the last core table;
    - strict ConsultantPlus noise blocks are excluded;
    - tail after the last core table is reported as trim candidate.

    It is safe to use before wiring into DocumentIngestionPipeline because the
    output is an explicit report plus prepared plain text only.
    """

    def analyze(self, file_path: str | Path) -> DocxPreprocessingResult:
        path = Path(file_path)
        if path.suffix.lower() != ".docx":
            raise ValueError(f"Expected .docx file, got: {path}")
        if not path.exists():
            raise FileNotFoundError(f"DOCX file not found: {path}")

        doc = DocxDocument(str(path))
        items = self._collect_items(doc)
        self._classify_tables(items)
        self._classify_categories(items)

        official_start_seq = self._find_official_start_seq(items)
        start_seq = official_start_seq if official_start_seq is not None else 1

        core_tables = [item for item in items if item.kind == "table" and item.table_type in CORE_TABLE_TYPES]
        last_core = max(core_tables, key=lambda item: item.seq) if core_tables else None
        last_core_seq = last_core.seq if last_core else None

        kept_items: list[DocxOrderedItem] = []
        trimmed_items: list[DocxOrderedItem] = []
        removed_noise_count = 0

        for item in items:
            if item.seq < start_seq:
                continue
            if self._is_strict_consultant_noise(item.text):
                removed_noise_count += 1
                continue
            if last_core_seq is not None and item.seq > last_core_seq:
                trimmed_items.append(item)
                continue
            kept_items.append(item)

        core_counts: Counter[str] = Counter()
        core_indexes: dict[str, list[int]] = {kind: [] for kind in sorted(CORE_TABLE_TYPES)}
        for item in kept_items:
            if item.kind == "table" and item.table_type in CORE_TABLE_TYPES:
                assert item.table_type is not None
                core_counts[item.table_type] += 1
                if item.table_index is not None:
                    core_indexes[item.table_type].append(item.table_index)

        tail_contains_core = any(item.table_type in CORE_TABLE_TYPES for item in trimmed_items)
        has_all_core = all(core_counts.get(kind, 0) > 0 for kind in CORE_TABLE_TYPES)
        has_exactly_one_each = all(core_counts.get(kind, 0) == 1 for kind in CORE_TABLE_TYPES)
        warnings: list[str] = []
        if official_start_seq is None:
            warnings.append("official_start_not_found")
        if not has_all_core:
            warnings.append("not_all_core_tables_found_in_kept_content")
        if not has_exactly_one_each:
            warnings.append("core_tables_not_exactly_1_1_1_in_kept_content")
        if tail_contains_core:
            warnings.append("tail_contains_core_table")
        if not trimmed_items:
            warnings.append("nothing_to_trim_after_last_core_table")

        prepared_text = self._build_prepared_text(kept_items)
        report = DocxPreprocessingReport(
            preprocessor_version=PREPROCESSOR_VERSION,
            filename=path.name,
            file_size_bytes=path.stat().st_size,
            file_sha256=self._sha256_file(path),
            total_items_count=len(items),
            total_paragraphs_count=sum(1 for item in items if item.kind == "paragraph"),
            total_tables_count=sum(1 for item in items if item.kind == "table"),
            official_start_seq=official_start_seq,
            items_before_official_start_count=sum(1 for item in items if official_start_seq is not None and item.seq < official_start_seq),
            official_start_found=official_start_seq is not None,
            core_table_counts={kind: int(core_counts.get(kind, 0)) for kind in sorted(CORE_TABLE_TYPES)},
            core_table_indexes=core_indexes,
            has_all_core_tables=has_all_core,
            has_exactly_one_each_core_table=has_exactly_one_each,
            last_core_table_seq=last_core.seq if last_core else None,
            last_core_table_index=last_core.table_index if last_core else None,
            last_core_table_type=last_core.table_type if last_core else None,
            trim_after_last_core_table_candidate=has_all_core and last_core is not None and not tail_contains_core,
            kept_items_count=len(kept_items),
            kept_paragraphs_count=sum(1 for item in kept_items if item.kind == "paragraph"),
            kept_tables_count=sum(1 for item in kept_items if item.kind == "table"),
            removed_before_official_start_count=sum(1 for item in items if item.seq < start_seq),
            removed_consultant_noise_count=removed_noise_count,
            trimmed_after_last_core_count=len(trimmed_items),
            trimmed_after_last_core_paragraphs_count=sum(1 for item in trimmed_items if item.kind == "paragraph"),
            trimmed_after_last_core_tables_count=sum(1 for item in trimmed_items if item.kind == "table"),
            trimmed_after_last_core_chars_count=sum(len(item.text) for item in trimmed_items),
            trimmed_tail_table_type_counts=dict(sorted(Counter(item.table_type or "unknown" for item in trimmed_items if item.kind == "table").items())),
            trimmed_tail_category_counts=dict(sorted(Counter(item.category or "unknown" for item in trimmed_items).items())),
            tail_contains_core_table=tail_contains_core,
            prepared_text_hash=self._sha256_text(prepared_text),
            prepared_text_chars_count=len(prepared_text),
            first_trimmed_samples=[self._sample_item(item) for item in trimmed_items[:8]],
            warnings=warnings,
        )
        return DocxPreprocessingResult(
            source_path=path,
            prepared_text=prepared_text,
            kept_items=kept_items,
            trimmed_items=trimmed_items,
            report=report,
        )

    def _collect_items(self, doc: _Document) -> list[DocxOrderedItem]:
        items: list[DocxOrderedItem] = []
        seq = 0
        table_index = 0
        for child in doc.element.body.iterchildren():
            if isinstance(child, CT_P):
                seq += 1
                paragraph = Paragraph(child, doc)
                text = self._clean_text(paragraph.text)
                if text:
                    items.append(DocxOrderedItem(seq=seq, kind="paragraph", text=text))
            elif isinstance(child, CT_Tbl):
                seq += 1
                table_index += 1
                table = Table(child, doc)
                row_texts: list[str] = []
                for row in table.rows[:16]:
                    cell_values = [self._clean_text(cell.text) for cell in row.cells]
                    deduped: list[str] = []
                    for value in cell_values:
                        if value and (not deduped or deduped[-1] != value):
                            deduped.append(value)
                    if deduped:
                        row_texts.append(" | ".join(deduped))
                items.append(
                    DocxOrderedItem(
                        seq=seq,
                        kind="table",
                        text=" || ".join(row_texts),
                        table_index=table_index,
                        rows_count=len(table.rows),
                    )
                )
        return items

    def _classify_tables(self, items: list[DocxOrderedItem]) -> None:
        paragraphs_before: list[DocxOrderedItem] = []
        for item in items:
            if item.kind == "paragraph":
                paragraphs_before.append(item)
                continue
            if item.kind != "table":
                continue
            context = " ".join(p.text for p in paragraphs_before[-16:])
            item.table_type = self._classify_table(item.text, context)

    def _classify_categories(self, items: list[DocxOrderedItem]) -> None:
        for item in items:
            item.category = self._classify_item_category(item)

    def _classify_table(self, table_text: str, context_text: str) -> str:
        table_n = self._norm(table_text)
        context_n = self._norm(" ".join(self._clean_text(context_text).split()[-260:]))
        combined_n = f"{context_n} {table_n}"

        if self._is_strict_consultant_noise(combined_n):
            return "consultant_noise"

        refusal_context = (
            "ИСЧЕРПЫВАЮЩИЙ ПЕРЕЧЕНЬ ОСНОВАНИЙ" in context_n
            or "ПЕРЕЧЕНЬ ОСНОВАНИЙ" in context_n
            or "ТАБЛИЦА 3" in context_n
        )
        refusal_header = (
            "НАИМЕНОВАНИЕ ОСНОВАНИЯ" in table_n
            or "ОСНОВАНИЯ ДЛЯ ОТКАЗА" in table_n
            or "ОТКАЗА В ПРИЕМЕ" in table_n
            or "ОТКАЗ В ПРИЕМЕ" in table_n
            or "ПРИОСТАНОВЛЕНИ" in table_n
        )
        if refusal_context and (refusal_header or "ОТКАЗ" in table_n or "ПРИОСТАНОВ" in table_n):
            return "refusal_reasons"
        if refusal_header and "НАИМЕНОВАН" in table_n and ("ОТКАЗ" in table_n or "ПРИОСТАНОВ" in table_n):
            return "refusal_reasons"

        documents_context = (
            "ИСЧЕРПЫВАЮЩИЙ ПЕРЕЧЕНЬ ДОКУМЕНТОВ" in context_n
            or "ПЕРЕЧЕНЬ ДОКУМЕНТОВ" in context_n
            or "ТАБЛИЦА 2" in context_n
        )
        documents_header = (
            "НАИМЕНОВАНИЕ ДОКУМЕНТА" in table_n
            or "ПЕРЕЧЕНЬ НЕОБХОДИМЫХ" in table_n
            or "ДОКУМЕНТЫ, НЕОБХОДИМЫЕ ДЛЯ ПРЕДОСТАВЛЕНИЯ" in table_n
            or "ДОКУМЕНТОВ, НЕОБХОДИМЫХ ДЛЯ ПРЕДОСТАВЛЕНИЯ" in table_n
        )
        if documents_context and documents_header:
            return "documents"

        identifiers_context = (
            "ИДЕНТИФИКАТОРЫ КАТЕГОРИЙ" in context_n
            or "КАТЕГОРИЙ (ПРИЗНАКОВ) ЗАЯВИТЕЛ" in context_n
            or "ТАБЛИЦА 1" in context_n
        )
        identifiers_header = (
            "НАИМЕНОВАНИЕ ПРИЗНАКА ЗАЯВИТЕЛ" in table_n
            or "НАИМЕНОВАНИЕ ОТДЕЛЬНЫХ ПРИЗНАКОВ ЗАЯВИТЕЛ" in table_n
            or "НАИМЕНОВАНИЕ (ЗНАЧЕНИЕ) ПРИЗНАКОВ ЗАЯВИТЕЛ" in table_n
            or "ПРИЗНАКОВ ЗАЯВИТЕЛ" in table_n
            or "ПРИЗНАКОВ) ЗАЯВИТЕЛ" in table_n
            or ("ИДЕНТИФИКАТОР" in table_n and "ЗАЯВИТЕЛ" in table_n and "КАТЕГОР" in table_n)
        )
        if identifiers_context and identifiers_header:
            return "identifiers"
        if identifiers_header and "ПРИЗНАК" in table_n:
            return "identifiers"

        if self._looks_like_form_or_template(combined_n):
            return "form_fields"
        return "generic"

    def _classify_item_category(self, item: DocxOrderedItem) -> str:
        text_n = self._norm(item.text)
        if item.kind == "table":
            if item.table_type in CORE_TABLE_TYPES:
                return f"core_{item.table_type}"
            if item.table_type == "consultant_noise":
                return "consultant_noise_table"
            if item.table_type == "form_fields":
                return "form_or_template_table"
            return "other_table"
        if self._is_strict_consultant_noise(text_n):
            return "consultant_noise_text"
        if text_n.startswith("ПРИЛОЖЕНИЕ"):
            return "appendix_heading"
        if self._looks_like_form_or_template(text_n):
            return "form_or_template_text"
        return "other_text"

    def _build_prepared_text(self, items: list[DocxOrderedItem]) -> str:
        parts: list[str] = []
        for item in items:
            if item.kind == "paragraph":
                parts.append(item.text)
            elif item.kind == "table":
                marker = f"[Таблица {item.table_index or ''}: {item.table_type or 'unknown'}]".strip()
                parts.append(marker)
                if item.text:
                    parts.append(item.text)
        text = "\n".join(part for part in parts if part)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _find_official_start_seq(self, items: list[DocxOrderedItem]) -> Optional[int]:
        for item in items:
            text_n = self._norm(item.text)
            if text_n == OFFICIAL_START_MARKER or text_n.startswith(OFFICIAL_START_MARKER):
                return item.seq
        return None

    def _looks_like_form_or_template(self, normalized_text: str) -> bool:
        return any(marker in normalized_text for marker in FORM_OR_TEMPLATE_MARKERS)

    def _is_strict_consultant_noise(self, text: str) -> bool:
        text_n = self._norm(text)
        if not text_n:
            return False
        if len(text_n) > 700:
            return False
        return any(marker in text_n for marker in CONSULTANT_NOISE_MARKERS)

    def _sample_item(self, item: DocxOrderedItem) -> dict[str, Any]:
        return {
            "seq": item.seq,
            "kind": item.kind,
            "table_index": item.table_index,
            "table_type": item.table_type,
            "category": item.category,
            "text": item.text[:300],
        }

    def _sha256_file(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _sha256_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _clean_text(self, value: str | None) -> str:
        if not value:
            return ""
        text = value.replace("\xa0", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _norm(self, value: str | None) -> str:
        if not value:
            return ""
        return self._clean_text(value).replace("ё", "е").replace("Ё", "Е").upper()
