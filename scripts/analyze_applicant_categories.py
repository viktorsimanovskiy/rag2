from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.retrieval.applicant_category_taxonomy import (  # noqa: E402
    build_applicant_category_groups,
    classify_applicant_category_text,
    group_definitions_as_json,
    normalize_text,
)


DEFAULT_JSON_PATH = Path("/home/logs/applicant_categories_catalog.json")
DEFAULT_TEXT_PATH = Path("/home/logs/applicant_categories_catalog.txt")

IDENTIFIER_ROWS_SQL = """
select
    dr.service_key,
    dr.service_name_short,
    dr.service_name_full,
    dr.original_filename,
    dt.table_id::text as table_id,
    dt.table_title,
    dt.table_number,
    dtr.row_id::text as row_id,
    dtr.row_order,
    dtr.row_summary,
    dtr.normalized_row_json,
    dtr.metadata_json,
    dtr.citation_json
from public.document_registry dr
join public.document_tables dt on dt.document_id = dr.document_id
join public.document_table_rows dtr on dtr.table_id = dt.table_id
where dr.status = 'active'
  and dt.table_type = 'identifiers'
order by dr.service_name_short nulls last, dr.original_filename, dtr.row_order;
"""

HEADER_LIKE_VALUES = {
    "",
    "2",
    "наименование признака заявителя",
    "наименование отдельных признаков заявителей",
    "наименование значение признаков заявителей",
    "наименование значений признаков заявителей",
    "идентификаторы категорий",
    "идентификатор категории",
    "категории заявителей",
}


@dataclass(slots=True)
class CategoryOccurrence:
    category_text: str
    normalized_category_text: str
    service_key: str | None
    service_name_short: str | None
    original_filename: str | None
    row_id: str | None
    row_order: int | None
    table_title: str | None
    table_number: str | None
    row_summary: str | None
    group_codes: list[str] = field(default_factory=list)
    group_labels: list[str] = field(default_factory=list)


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\xa0", " ")).strip()


def is_header_like(value: str) -> bool:
    normalized = normalize_text(value).strip()
    if normalized in HEADER_LIKE_VALUES:
        return True
    if normalized.startswith("n п п") or normalized.startswith("№ п п"):
        return True
    if "наименование признака заявителя" in normalized:
        return True
    if "идентификаторы категорий" in normalized:
        return True
    return False


def dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def extract_category_text(row: dict[str, Any]) -> str:
    normalized_row = dict_or_empty(row.get("normalized_row_json"))
    metadata = dict_or_empty(row.get("metadata_json"))

    direct = clean_text(normalized_row.get("applicant_category_name"))
    if direct and not is_header_like(direct):
        return direct

    cells = metadata.get("cells_by_semantic_key")
    if isinstance(cells, dict):
        value = clean_text(cells.get("applicant_category_name"))
        if value and not is_header_like(value):
            return value

    for container_name in ("cells_by_header_key", "cells_by_header_normalized"):
        cells = metadata.get(container_name)
        if not isinstance(cells, dict):
            continue
        for key, value in cells.items():
            key_norm = normalize_text(str(key))
            if "заявител" not in key_norm and "признак" not in key_norm and "категор" not in key_norm:
                continue
            text = clean_text(value)
            if text and not is_header_like(text):
                return text

    summary = clean_text(row.get("row_summary"))
    if summary and not is_header_like(summary):
        return summary

    return ""


async def load_identifier_rows() -> list[dict[str, Any]]:
    from sqlalchemy import text
    from app.config.settings import load_settings
    from app.db.session import DatabaseSessionManager

    settings = load_settings()
    manager = DatabaseSessionManager(settings.database)
    manager.initialize()
    try:
        async with manager.session_scope() as session:
            result = await session.execute(text(IDENTIFIER_ROWS_SQL))
            return [dict(row) for row in result.mappings().all()]
    finally:
        await manager.dispose()


def build_occurrences(rows: list[dict[str, Any]]) -> list[CategoryOccurrence]:
    occurrences: list[CategoryOccurrence] = []
    for row in rows:
        category_text = extract_category_text(row)
        if not category_text:
            continue
        normalized = normalize_text(category_text)
        if not normalized or is_header_like(category_text):
            continue

        matches = classify_applicant_category_text(category_text)
        group_codes = [match.code for match in matches]
        group_labels = [match.label for match in matches]

        occurrences.append(
            CategoryOccurrence(
                category_text=category_text,
                normalized_category_text=normalized,
                service_key=clean_text(row.get("service_key")) or None,
                service_name_short=clean_text(row.get("service_name_short")) or None,
                original_filename=clean_text(row.get("original_filename")) or None,
                row_id=clean_text(row.get("row_id")) or None,
                row_order=int(row["row_order"]) if row.get("row_order") is not None else None,
                table_title=clean_text(row.get("table_title")) or None,
                table_number=clean_text(row.get("table_number")) or None,
                row_summary=clean_text(row.get("row_summary")) or None,
                group_codes=group_codes,
                group_labels=group_labels,
            )
        )
    return occurrences


def build_catalog(occurrences: list[CategoryOccurrence]) -> dict[str, Any]:
    group_defs = {definition.code: definition for definition in build_applicant_category_groups()}
    grouped: dict[str, list[CategoryOccurrence]] = defaultdict(list)
    for item in occurrences:
        if item.group_codes:
            for code in item.group_codes:
                grouped[code].append(item)
        else:
            grouped["other"].append(item)

    groups_json: list[dict[str, Any]] = []
    for definition in build_applicant_category_groups():
        items = grouped.get(definition.code, [])
        if not items:
            continue
        unique_by_text: dict[str, list[CategoryOccurrence]] = defaultdict(list)
        for item in items:
            unique_by_text[item.normalized_category_text].append(item)

        examples: list[dict[str, Any]] = []
        for _, example_items in sorted(
            unique_by_text.items(),
            key=lambda pair: (-len(pair[1]), pair[1][0].category_text.lower()),
        )[:20]:
            first = example_items[0]
            services = sorted({item.service_name_short or item.service_key or "" for item in example_items if item.service_name_short or item.service_key})
            examples.append(
                {
                    "category_text": first.category_text,
                    "occurrences_count": len(example_items),
                    "services_count": len(services),
                    "service_examples": services[:8],
                }
            )

        groups_json.append(
            {
                "code": definition.code,
                "label": definition.label,
                "rows_count": len(items),
                "unique_categories_count": len(unique_by_text),
                "examples": examples,
            }
        )

    other_items = grouped.get("other", [])
    other_counter = Counter(item.normalized_category_text for item in other_items)
    other_examples: list[dict[str, Any]] = []
    by_norm: dict[str, CategoryOccurrence] = {}
    for item in other_items:
        by_norm.setdefault(item.normalized_category_text, item)
    for normalized, count in other_counter.most_common(100):
        first = by_norm[normalized]
        other_examples.append(
            {
                "category_text": first.category_text,
                "occurrences_count": count,
                "service_name_short": first.service_name_short,
                "original_filename": first.original_filename,
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "active DB corpus: document_tables.table_type = identifiers",
        "total_identifier_rows": len(occurrences),
        "unique_raw_categories_count": len({item.normalized_category_text for item in occurrences}),
        "group_definitions": group_definitions_as_json(),
        "groups": groups_json,
        "uncategorized": {
            "rows_count": len(other_items),
            "unique_categories_count": len(other_counter),
            "examples": other_examples,
        },
    }


def render_text_report(catalog: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("ПРИМЕРНЫЙ ПЕРЕЧЕНЬ СМЫСЛОВЫХ КАТЕГОРИЙ ЗАЯВИТЕЛЕЙ")
    lines.append("=" * 90)
    lines.append(f"Всего строк категорий: {catalog['total_identifier_rows']}")
    lines.append(f"Уникальных исходных формулировок: {catalog['unique_raw_categories_count']}")
    lines.append("")
    lines.append("Важно: это рабочая смысловая группировка для подбора возможных мер, а не юридический вывод о праве заявителя.")
    lines.append("")

    for group in catalog.get("groups") or []:
        lines.append(f"## {group['label']}")
        lines.append(f"Код: {group['code']}")
        lines.append(f"Строк: {group['rows_count']}; уникальных формулировок: {group['unique_categories_count']}")
        lines.append("Примеры формулировок из НПА:")
        for example in group.get("examples", [])[:10]:
            services = "; ".join(example.get("service_examples") or [])
            suffix = f" [{services}]" if services else ""
            lines.append(f"- {example['category_text']}{suffix}")
        lines.append("")

    uncategorized = catalog.get("uncategorized") or {}
    lines.append("## Не распределено по текущим правилам")
    lines.append(f"Строк: {uncategorized.get('rows_count', 0)}; уникальных формулировок: {uncategorized.get('unique_categories_count', 0)}")
    for example in (uncategorized.get("examples") or [])[:40]:
        service = example.get("service_name_short") or example.get("original_filename") or ""
        suffix = f" [{service}]" if service else ""
        lines.append(f"- {example['category_text']}{suffix}")
    lines.append("")
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> int:
    rows = await load_identifier_rows()
    occurrences = build_occurrences(rows)
    catalog = build_catalog(occurrences)

    json_path = Path(args.json_out).expanduser().resolve()
    text_path = Path(args.text_out).expanduser().resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    text_report = render_text_report(catalog)
    text_path.write_text(text_report, encoding="utf-8")

    if not args.quiet:
        print(text_report)
        print(f"\nJSON сохранён: {json_path}")
        print(f"Текстовый отчёт сохранён: {text_path}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Собрать смысловой перечень категорий заявителей из таблиц identifiers активного корпуса."
    )
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_PATH), help="Куда сохранить JSON-отчёт.")
    parser.add_argument("--text-out", default=str(DEFAULT_TEXT_PATH), help="Куда сохранить текстовый отчёт.")
    parser.add_argument("--quiet", action="store_true", help="Не печатать отчёт в консоль.")
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
