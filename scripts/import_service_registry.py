# ============================================================
# File: scripts/import_service_registry.py
# Purpose:
#   Import concrete public services from Актуальный_приказ5.xlsx
#   into service_registry.
#
# Usage:
#   python scripts/import_service_registry.py --xlsx /path/to/Актуальный_приказ5.xlsx
#   python scripts/import_service_registry.py --xlsx /path/to/Актуальный_приказ5.xlsx --dry-run
#
# Notes:
#   - no openpyxl dependency is required;
#   - the script reads the XLSX file through the Python standard library;
#   - cleaned_filename is the main registry anchor and must be unique.
# ============================================================

from __future__ import annotations

import argparse
import asyncio
import hashlib
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
from xml.etree import ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SHEET_NAME_DEFAULT = "Для ИИ"
EXPECTED_HEADERS = {
    "service_name_full": "Наименование услуги по приказу министерства № 242-ОД от 19.03.2026",
    "frgu_1": "ФРГУ 1.0",
    "frgu_3": "ФРГУ 3.0",
    "order_details": "Реквизиты приказа по утверждению административного регламента",
    "service_name_short": "Краткое наименование услуги",
    "aliases": "Ключевые слова / алиасы",
    "raw_filename": "Имя файла из Консультантплюс (raw)",
    "cleaned_filename": "Имя файла обработанного (cleaned)",
    "note": "Примечание",
}

XML_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
XML_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
XML_PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {
    "a": XML_MAIN_NS,
    "r": XML_REL_NS,
    "rel": XML_PACKAGE_REL_NS,
}


@dataclass(slots=True, frozen=True)
class ServiceRegistryImportRow:
    row_number: int
    service_key: str
    service_name_full: str
    service_name_short: str
    frgu_1: Optional[str]
    frgu_3: Optional[str]
    order_details: Optional[str]
    raw_filename: str
    cleaned_filename: str
    aliases: list[str]
    note: Optional[str]


@dataclass(slots=True, frozen=True)
class ImportSummary:
    parsed_rows: int
    created_rows: int
    updated_rows: int
    dry_run: bool


class ServiceRegistryImportError(Exception):
    """Raised when the service registry import cannot be completed safely."""


# ============================================================
# XLSX reading without third-party dependencies
# ============================================================

def read_xlsx_sheet_rows(xlsx_path: Path, sheet_name: str) -> list[list[Any]]:
    if not xlsx_path.exists():
        raise ServiceRegistryImportError(f"XLSX file not found: {xlsx_path}")

    with zipfile.ZipFile(xlsx_path) as archive:
        shared_strings = _read_shared_strings(archive)
        sheet_path = _resolve_sheet_path(archive, sheet_name)
        return _read_sheet_rows(archive, sheet_path, shared_strings)


def _read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []

    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    values: list[str] = []
    for item in root.findall("a:si", NS):
        parts = [node.text or "" for node in item.findall(".//a:t", NS)]
        values.append("".join(parts))
    return values


def _resolve_sheet_path(archive: zipfile.ZipFile, sheet_name: str) -> str:
    workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
    target_rel_id: Optional[str] = None

    for sheet in workbook_root.findall(".//a:sheets/a:sheet", NS):
        if sheet.attrib.get("name") == sheet_name:
            target_rel_id = sheet.attrib.get(f"{{{XML_REL_NS}}}id")
            break

    if not target_rel_id:
        available = [sheet.attrib.get("name", "") for sheet in workbook_root.findall(".//a:sheets/a:sheet", NS)]
        raise ServiceRegistryImportError(
            f"Sheet '{sheet_name}' not found. Available sheets: {', '.join(available)}"
        )

    rels_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    for rel in rels_root.findall("rel:Relationship", NS):
        if rel.attrib.get("Id") == target_rel_id:
            target = rel.attrib.get("Target", "")
            normalized_target = target.replace("../", "")
            if not normalized_target.startswith("xl/"):
                normalized_target = f"xl/{normalized_target}"
            return normalized_target

    raise ServiceRegistryImportError(f"Cannot resolve XLSX relationship for sheet '{sheet_name}'.")


def _read_sheet_rows(
    archive: zipfile.ZipFile,
    sheet_path: str,
    shared_strings: list[str],
) -> list[list[Any]]:
    root = ET.fromstring(archive.read(sheet_path))
    rows: list[list[Any]] = []

    for row in root.findall(".//a:sheetData/a:row", NS):
        cells_by_index: dict[int, Any] = {}
        max_index = -1

        for cell in row.findall("a:c", NS):
            cell_ref = cell.attrib.get("r", "")
            col_index = _column_index_from_cell_ref(cell_ref)
            max_index = max(max_index, col_index)
            cells_by_index[col_index] = _read_cell_value(cell, shared_strings)

        if max_index >= 0:
            rows.append([cells_by_index.get(idx) for idx in range(max_index + 1)])

    return rows


def _read_cell_value(cell: ET.Element, shared_strings: list[str]) -> Any:
    cell_type = cell.attrib.get("t")

    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(".//a:t", NS))

    value_node = cell.find("a:v", NS)
    if value_node is None or value_node.text is None:
        return None

    raw_value = value_node.text

    if cell_type == "s":
        try:
            return shared_strings[int(raw_value)]
        except (IndexError, ValueError) as exc:
            raise ServiceRegistryImportError(f"Broken shared string reference: {raw_value}") from exc

    return raw_value


def _column_index_from_cell_ref(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref)
    if not match:
        raise ServiceRegistryImportError(f"Cannot parse XLSX cell reference: {cell_ref}")

    index = 0
    for char in match.group(1):
        index = index * 26 + ord(char) - ord("A") + 1
    return index - 1


# ============================================================
# Row parsing and validation
# ============================================================

def parse_service_registry_rows(xlsx_path: Path, sheet_name: str) -> list[ServiceRegistryImportRow]:
    raw_rows = read_xlsx_sheet_rows(xlsx_path, sheet_name)
    if not raw_rows:
        raise ServiceRegistryImportError("XLSX sheet is empty.")

    header_map = _build_header_map(raw_rows[0])
    rows: list[ServiceRegistryImportRow] = []

    for zero_based_idx, raw_row in enumerate(raw_rows[1:], start=2):
        if _is_empty_row(raw_row):
            continue

        full_name = _required_cell(raw_row, header_map, "service_name_full", zero_based_idx)
        short_name = _required_cell(raw_row, header_map, "service_name_short", zero_based_idx)
        aliases_raw = _required_cell(raw_row, header_map, "aliases", zero_based_idx)
        raw_filename = _required_cell(raw_row, header_map, "raw_filename", zero_based_idx)
        cleaned_filename = _required_cell(raw_row, header_map, "cleaned_filename", zero_based_idx)

        row = ServiceRegistryImportRow(
            row_number=zero_based_idx,
            service_key=build_service_key(cleaned_filename),
            service_name_full=full_name,
            service_name_short=short_name,
            frgu_1=_optional_cell(raw_row, header_map, "frgu_1"),
            frgu_3=_optional_cell(raw_row, header_map, "frgu_3"),
            order_details=_optional_cell(raw_row, header_map, "order_details"),
            raw_filename=raw_filename,
            cleaned_filename=cleaned_filename,
            aliases=split_aliases(aliases_raw),
            note=_optional_cell(raw_row, header_map, "note"),
        )
        rows.append(row)

    _validate_rows(rows)
    return rows


def _build_header_map(header_row: list[Any]) -> dict[str, int]:
    normalized_headers = {
        normalize_cell(value): idx
        for idx, value in enumerate(header_row)
        if normalize_cell(value)
    }

    result: dict[str, int] = {}
    missing: list[str] = []

    for logical_name, expected_header in EXPECTED_HEADERS.items():
        idx = normalized_headers.get(normalize_cell(expected_header))
        if idx is None:
            missing.append(expected_header)
        else:
            result[logical_name] = idx

    if missing:
        raise ServiceRegistryImportError(
            "Missing required columns in service registry XLSX: " + "; ".join(missing)
        )

    return result


def _is_empty_row(row: list[Any]) -> bool:
    return all(normalize_cell(value) == "" for value in row)


def _required_cell(
    row: list[Any],
    header_map: dict[str, int],
    logical_name: str,
    row_number: int,
) -> str:
    value = _optional_cell(row, header_map, logical_name)
    if not value:
        raise ServiceRegistryImportError(
            f"Required cell is empty: row={row_number}, column='{EXPECTED_HEADERS[logical_name]}'"
        )
    return value


def _optional_cell(row: list[Any], header_map: dict[str, int], logical_name: str) -> Optional[str]:
    idx = header_map[logical_name]
    if idx >= len(row):
        return None

    value = normalize_cell(row[idx])
    return value or None


def normalize_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_aliases(value: str) -> list[str]:
    aliases: list[str] = []
    seen: set[str] = set()

    for item in re.split(r"[;\n]+", value):
        alias = normalize_cell(item)
        if not alias:
            continue
        key = alias.casefold()
        if key in seen:
            continue
        seen.add(key)
        aliases.append(alias)

    return aliases


def _validate_rows(rows: list[ServiceRegistryImportRow]) -> None:
    if not rows:
        raise ServiceRegistryImportError("No service rows found in XLSX.")

    duplicate_errors: list[str] = []
    duplicate_errors.extend(_find_duplicates(rows, "service_key"))
    duplicate_errors.extend(_find_duplicates(rows, "raw_filename"))
    duplicate_errors.extend(_find_duplicates(rows, "cleaned_filename"))
    duplicate_errors.extend(_find_duplicates(rows, "frgu_1", ignore_empty=True))
    duplicate_errors.extend(_find_duplicates(rows, "frgu_3", ignore_empty=True))

    if duplicate_errors:
        raise ServiceRegistryImportError("Duplicate values in XLSX: " + " | ".join(duplicate_errors))


def _find_duplicates(
    rows: Iterable[ServiceRegistryImportRow],
    field_name: str,
    *,
    ignore_empty: bool = False,
) -> list[str]:
    seen: dict[str, int] = {}
    errors: list[str] = []

    for row in rows:
        value = getattr(row, field_name)
        if value is None:
            if ignore_empty:
                continue
            normalized = ""
        else:
            normalized = normalize_cell(value).casefold()

        if ignore_empty and not normalized:
            continue

        previous_row_number = seen.get(normalized)
        if previous_row_number is not None:
            errors.append(
                f"{field_name}='{value}' rows={previous_row_number},{row.row_number}"
            )
        else:
            seen[normalized] = row.row_number

    return errors


# ============================================================
# Service key generation
# ============================================================

def build_service_key(cleaned_filename: str) -> str:
    base = cleaned_filename
    base = re.sub(r"\.docx$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"_cleaned$", "", base, flags=re.IGNORECASE)
    slug = transliterate_ru_to_latin(base)
    slug = slug.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")

    digest = hashlib.sha1(normalize_cell(cleaned_filename).casefold().encode("utf-8")).hexdigest()[:12]

    if not slug:
        return f"svc_{digest}"

    if len(slug) > 80:
        slug = slug[:80].rstrip("_")

    return f"svc_{slug}_{digest}"


def transliterate_ru_to_latin(value: str) -> str:
    table = {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
        "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
        "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
        "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "",
        "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    }
    result: list[str] = []
    for char in value:
        lower = char.lower()
        replacement = table.get(lower)
        if replacement is None:
            result.append(char)
        else:
            result.append(replacement)
    return "".join(result)


# ============================================================
# DB import
# ============================================================

async def import_service_registry_rows(
    rows: list[ServiceRegistryImportRow],
    *,
    dry_run: bool,
) -> ImportSummary:
    if dry_run:
        return ImportSummary(
            parsed_rows=len(rows),
            created_rows=0,
            updated_rows=0,
            dry_run=True,
        )

    # Imported lazily so that --dry-run can validate the XLSX even on a machine
    # where project DB dependencies are not installed.
    from sqlalchemy import select

    from app.config.settings import load_settings
    from app.db.models.services import ServiceRegistry
    from app.db.session import DatabaseSessionManager

    settings = load_settings()
    db_manager = DatabaseSessionManager(settings.database)
    db_manager.initialize()

    created = 0
    updated = 0

    try:
        async with db_manager.session_scope() as session:
            cleaned_filenames = [row.cleaned_filename for row in rows]
            result = await session.execute(
                select(ServiceRegistry).where(ServiceRegistry.cleaned_filename.in_(cleaned_filenames))
            )
            existing_by_cleaned = {
                service.cleaned_filename: service
                for service in result.scalars().all()
            }

            for row in rows:
                service = existing_by_cleaned.get(row.cleaned_filename)
                if service is None:
                    service = ServiceRegistry(
                        service_key=row.service_key,
                        service_name_full=row.service_name_full,
                        service_name_short=row.service_name_short,
                        frgu_1=row.frgu_1,
                        frgu_3=row.frgu_3,
                        order_details=row.order_details,
                        raw_filename=row.raw_filename,
                        cleaned_filename=row.cleaned_filename,
                        aliases_json=row.aliases,
                        note=row.note,
                        is_active=True,
                    )
                    session.add(service)
                    created += 1
                else:
                    service.service_key = row.service_key
                    service.service_name_full = row.service_name_full
                    service.service_name_short = row.service_name_short
                    service.frgu_1 = row.frgu_1
                    service.frgu_3 = row.frgu_3
                    service.order_details = row.order_details
                    service.raw_filename = row.raw_filename
                    service.aliases_json = row.aliases
                    service.note = row.note
                    service.is_active = True
                    updated += 1

        return ImportSummary(
            parsed_rows=len(rows),
            created_rows=created,
            updated_rows=updated,
            dry_run=False,
        )
    finally:
        await db_manager.dispose()


# ============================================================
# CLI
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import service_registry from Актуальный_приказ5.xlsx."
    )
    parser.add_argument(
        "--xlsx",
        required=True,
        type=Path,
        help="Path to Актуальный_приказ5.xlsx.",
    )
    parser.add_argument(
        "--sheet",
        default=SHEET_NAME_DEFAULT,
        help=f"Sheet name. Default: {SHEET_NAME_DEFAULT}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate XLSX without writing to DB.",
    )
    return parser


def print_rows_preview(rows: list[ServiceRegistryImportRow], *, limit: int = 5) -> None:
    print("Preview:")
    for row in rows[:limit]:
        print(
            f"  row={row.row_number} service_key={row.service_key} "
            f"short='{row.service_name_short}' cleaned='{row.cleaned_filename}'"
        )


def main() -> None:
    args = build_arg_parser().parse_args()

    try:
        rows = parse_service_registry_rows(args.xlsx, args.sheet)
        print(f"Parsed service rows: {len(rows)}")
        print_rows_preview(rows)

        summary = asyncio.run(
            import_service_registry_rows(
                rows,
                dry_run=args.dry_run,
            )
        )

        if summary.dry_run:
            print("Dry-run finished successfully. DB was not changed.")
        else:
            print(
                "Import finished successfully: "
                f"parsed={summary.parsed_rows}, created={summary.created_rows}, updated={summary.updated_rows}"
            )
    except ServiceRegistryImportError as exc:
        print(f"Service registry import failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
