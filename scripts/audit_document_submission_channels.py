from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config.settings import load_settings
from app.db.session import DatabaseSessionManager


CHANNEL_PATTERNS: dict[str, list[str]] = {
    "mfc": [
        r"\bмфц\b",
        r"многофункциональн",
    ],
    "epgu": [
        r"\bепгу\b",
        r"единый\s+портал",
        r"единый\s+портал\s+государственных",
        r"госуслуг",
    ],
    "rpgu": [
        r"\bрпгу\b",
        r"региональн\w*\s+портал",
        r"краев\w*\s+портал",
    ],
    "in_person": [
        r"личн",
        r"личный\s+прием",
        r"уполномоченн\w*\s+учрежден",
        r"министерств",
        r"учрежден",
    ],
    "postal": [
        r"почт",
        r"почтов",
    ],
    "common_submission": [
        r"способ\w*\s+подач",
        r"способ\w*\s+представлен",
        r"канал\w*\s+подач",
        r"форма\w*\s+подач",
        r"submission",
        r"present",
    ],
}

NUMBER_KEY_PATTERNS = [
    r"(^|[_\s])n([_\s]|$)",
    r"номер",
    r"№",
    r"п/п",
    r"n_п_п",
    r"nпп",
]

DOCUMENT_KEY_PATTERNS = [
    r"наименован.*документ",
    r"назван.*документ",
    r"document.*name",
    r"doc.*name",
    r"документ",
]

REQUIREMENT_KEY_PATTERNS = [
    r"requirement_group",
    r"групп.*треб",
    r"обязательн",
    r"инициатив",
]

AMENDMENT_PATTERNS = [
    r"\(\s*в\s+ред\.",
    r"в\s+редакции\s+приказ",
    r"утратил\s+силу",
]

SECTION_PATTERNS = [
    r"документы,\s*необходимые\s+для\s+предоставления\s+услуги",
    r"представляемые\s+заявителем\s+самостоятельно",
    r"представляемые\s+заявителем\s+по\s+собственной\s+инициативе",
]

BEZ_STATEMENT_PATTERNS = [
    r"беззаявительн",
    r"без\s+заявлен",
]


DOCUMENT_TABLES_SQL = """
select
    dr.service_key,
    dr.service_name_short,
    dr.service_name_full,
    dr.original_filename,
    dr.document_id::text as document_id,
    dt.table_id::text as table_id,
    dt.table_number,
    dt.appendix_number,
    dt.table_title,
    dt.rows_count,
    dt.header_schema_json,
    dt.metadata_json as table_metadata_json,
    dtr.row_id::text as row_id,
    dtr.row_order,
    dtr.row_json,
    dtr.normalized_row_json,
    dtr.metadata_json as row_metadata_json,
    dtr.row_summary
from public.document_registry dr
join public.document_tables dt on dt.document_id = dr.document_id
left join public.document_table_rows dtr on dtr.table_id = dt.table_id
where dr.status = 'active'
  and dt.table_type = 'documents'
  and (:service_key = '' or dr.service_key = :service_key)
  and (:service_name = '' or coalesce(dr.service_name_short, dr.service_name_full, '') ilike '%' || :service_name || '%')
  and (:filename = '' or coalesce(dr.original_filename, '') ilike '%' || :filename || '%')
order by dr.service_key, dt.table_number nulls last, dt.table_id, dtr.row_order nulls last;
"""


@dataclass(slots=True)
class RowDiagnostics:
    service_key: str
    service_name_short: str
    service_name_full: str
    original_filename: str
    document_id: str
    table_id: str
    table_number: str
    appendix_number: str
    table_title: str
    row_id: str
    row_order: int | None
    npp: str
    row_kind: str
    requirement_group: str
    document_text: str
    mfc: str
    epgu: str
    rpgu: str
    in_person: str
    postal: str
    common_submission: str
    channel_key_hits: str
    normalized_keys: str
    raw_keys: str
    metadata_keys: str
    row_summary: str


@dataclass(slots=True)
class TableDiagnostics:
    service_key: str
    service_name_short: str
    service_name_full: str
    original_filename: str
    document_id: str
    table_id: str
    table_number: str
    appendix_number: str
    table_title: str
    rows_count_declared: int | None
    rows_count_loaded: int = 0
    npp_rows_count: int = 0
    dotted_npp_rows_count: int = 0
    max_npp_depth: int = 0
    section_rows_count: int = 0
    group_like_rows_count: int = 0
    document_like_rows_count: int = 0
    amendment_rows_count: int = 0
    bez_statement_rows_count: int = 0
    mfc_non_empty_rows: int = 0
    epgu_non_empty_rows: int = 0
    rpgu_non_empty_rows: int = 0
    in_person_non_empty_rows: int = 0
    postal_non_empty_rows: int = 0
    common_submission_non_empty_rows: int = 0
    header_schema_keys: set[str] = field(default_factory=set)
    row_json_keys: set[str] = field(default_factory=set)
    normalized_row_json_keys: set[str] = field(default_factory=set)
    row_metadata_keys: set[str] = field(default_factory=set)
    channel_key_hits: set[str] = field(default_factory=set)


def norm(value: Any) -> str:
    if value is None:
        return ""
    text_value = str(value)
    text_value = text_value.replace("\xa0", " ")
    text_value = re.sub(r"\s+", " ", text_value)
    return text_value.strip()


def norm_l(value: Any) -> str:
    return norm(value).lower().replace("ё", "е")


def compile_patterns(patterns: Iterable[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern, re.IGNORECASE | re.UNICODE) for pattern in patterns]


COMPILED_CHANNEL_PATTERNS = {
    name: compile_patterns(patterns) for name, patterns in CHANNEL_PATTERNS.items()
}
COMPILED_NUMBER_KEY_PATTERNS = compile_patterns(NUMBER_KEY_PATTERNS)
COMPILED_DOCUMENT_KEY_PATTERNS = compile_patterns(DOCUMENT_KEY_PATTERNS)
COMPILED_REQUIREMENT_KEY_PATTERNS = compile_patterns(REQUIREMENT_KEY_PATTERNS)
COMPILED_AMENDMENT_PATTERNS = compile_patterns(AMENDMENT_PATTERNS)
COMPILED_SECTION_PATTERNS = compile_patterns(SECTION_PATTERNS)
COMPILED_BEZ_PATTERNS = compile_patterns(BEZ_STATEMENT_PATTERNS)


def is_mapping(value: Any) -> bool:
    return isinstance(value, dict)


def as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def flatten_json(value: Any, *, prefix: str = "") -> list[tuple[str, Any]]:
    result: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, sub_value in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            result.extend(flatten_json(sub_value, prefix=path))
    elif isinstance(value, list):
        for index, sub_value in enumerate(value):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            result.extend(flatten_json(sub_value, prefix=path))
    else:
        result.append((prefix, value))
    return result


def non_empty_items(*objects: dict[str, Any]) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for obj in objects:
        for key_path, value in flatten_json(obj):
            value_text = norm(value)
            if value_text:
                items.append((key_path, value_text))
    return items


def key_matches(key: str, patterns: list[re.Pattern[str]]) -> bool:
    normalized_key = norm_l(key).replace(".", "_")
    return any(pattern.search(normalized_key) for pattern in patterns)


def text_matches(value: str, patterns: list[re.Pattern[str]]) -> bool:
    normalized_value = norm_l(value)
    return any(pattern.search(normalized_value) for pattern in patterns)


def find_first_by_key_patterns(objects: list[dict[str, Any]], patterns: list[re.Pattern[str]]) -> str:
    for obj in objects:
        for key_path, value in flatten_json(obj):
            if key_matches(key_path, patterns):
                text_value = norm(value)
                if text_value:
                    return text_value
    return ""


def find_document_text(row_json: dict[str, Any], normalized_row_json: dict[str, Any], row_metadata_json: dict[str, Any], row_summary: str) -> str:
    for obj in (normalized_row_json, row_json, row_metadata_json):
        value = find_first_by_key_patterns([obj], COMPILED_DOCUMENT_KEY_PATTERNS)
        if value and not re.fullmatch(r"\d+(?:\.\d+)*", value.strip()):
            return value
    return norm(row_summary)


def find_requirement_group(normalized_row_json: dict[str, Any], row_metadata_json: dict[str, Any], row_json: dict[str, Any]) -> str:
    for key in ("requirement_group", "document_group", "row_scope"):
        for obj in (normalized_row_json, row_metadata_json, row_json):
            value = norm(obj.get(key)) if isinstance(obj, dict) else ""
            if value:
                return value
    return find_first_by_key_patterns([normalized_row_json, row_metadata_json, row_json], COMPILED_REQUIREMENT_KEY_PATTERNS)


def find_npp(row_json: dict[str, Any], normalized_row_json: dict[str, Any], row_metadata_json: dict[str, Any]) -> str:
    for key in ("npp", "n_p_p", "number", "row_number", "номер", "№", "n"):
        for obj in (normalized_row_json, row_metadata_json, row_json):
            if isinstance(obj, dict):
                value = norm(obj.get(key))
                if value:
                    return value
    value = find_first_by_key_patterns([normalized_row_json, row_metadata_json, row_json], COMPILED_NUMBER_KEY_PATTERNS)
    if value:
        return value

    # Fallback: sometimes row summary begins with the document table number.
    row_summary = norm(row_metadata_json.get("row_summary") if isinstance(row_metadata_json, dict) else "")
    match = re.match(r"^\s*(\d+(?:\.\d+)*)\b", row_summary)
    return match.group(1) if match else ""


def npp_depth(npp: str) -> int:
    if not re.fullmatch(r"\d+(?:\.\d+)*", norm(npp)):
        return 0
    return norm(npp).count(".") + 1


def collect_channel_values(
    row_json: dict[str, Any],
    normalized_row_json: dict[str, Any],
    row_metadata_json: dict[str, Any],
) -> tuple[dict[str, str], list[str]]:
    values_by_channel: dict[str, list[str]] = {channel: [] for channel in CHANNEL_PATTERNS}
    hits: list[str] = []

    # Prefer raw row cells because headers often keep the real submission-channel text.
    for source_name, obj in (
        ("row_json", row_json),
        ("normalized_row_json", normalized_row_json),
        ("row_metadata_json", row_metadata_json),
    ):
        for key_path, value in flatten_json(obj):
            value_text = norm(value)
            if not value_text:
                continue
            lookup = f"{key_path} {value_text}"
            for channel, patterns in COMPILED_CHANNEL_PATTERNS.items():
                key_hit = key_matches(key_path, patterns)
                # For common_submission, value text alone is also meaningful.
                value_hit = channel == "common_submission" and text_matches(value_text, patterns)
                if key_hit or value_hit:
                    values_by_channel[channel].append(value_text)
                    hits.append(f"{source_name}:{channel}:{key_path}")

    compact: dict[str, str] = {}
    for channel, values in values_by_channel.items():
        compact[channel] = join_unique(values, limit=4, max_chars=900)
    return compact, sorted(set(hits))


def join_unique(values: Iterable[str], *, limit: int = 20, max_chars: int = 2000) -> str:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = norm(value)
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(clean)
        if len(result) >= limit:
            break
    text_value = " | ".join(result)
    if len(text_value) > max_chars:
        return text_value[: max_chars - 1].rstrip() + "…"
    return text_value


def classify_row_kind(npp: str, document_text: str, requirement_group: str, channel_values: dict[str, str]) -> str:
    text_value = norm_l(" ".join([npp, document_text, requirement_group]))
    if not text_value:
        return "empty"
    if text_matches(text_value, COMPILED_AMENDMENT_PATTERNS):
        return "amendment_note"
    if text_matches(text_value, COMPILED_BEZ_PATTERNS):
        return "bez_statement"
    if text_matches(text_value, COMPILED_SECTION_PATTERNS):
        return "section"

    has_channel = any(norm(v) for v in channel_values.values())
    depth = npp_depth(npp)

    # If the row has a number, text, but no submission-channel values, it is likely a group in hierarchical tables.
    if depth > 0 and not has_channel:
        group_markers = [
            "документы",
            "сведения",
            "подтверждающие",
            "представляемые",
            "необходимые",
            "в случае",
        ]
        if any(marker in text_value for marker in group_markers):
            return "group"

    if depth > 0:
        return "document"

    if not has_channel and len(document_text) < 180:
        return "section_or_group"

    return "other_text"


def format_keys(obj: dict[str, Any]) -> str:
    keys = sorted({key for key, _ in flatten_json(obj) if key})
    return join_unique(keys, limit=60, max_chars=1800)


async def fetch_rows(manager: DatabaseSessionManager, args: argparse.Namespace) -> list[dict[str, Any]]:
    async with manager.session_scope() as session:
        result = await session.execute(
            text(DOCUMENT_TABLES_SQL),
            {
                "service_key": args.service_key or "",
                "service_name": args.service_name or "",
                "filename": args.filename or "",
            },
        )
        return [dict(row) for row in result.mappings().all()]


def build_diagnostics(raw_rows: list[dict[str, Any]]) -> tuple[list[TableDiagnostics], list[RowDiagnostics], list[dict[str, Any]]]:
    tables: dict[str, TableDiagnostics] = {}
    rows: list[RowDiagnostics] = []
    key_counter: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)

    for raw in raw_rows:
        table_id = norm(raw.get("table_id"))
        if table_id not in tables:
            tables[table_id] = TableDiagnostics(
                service_key=norm(raw.get("service_key")),
                service_name_short=norm(raw.get("service_name_short")),
                service_name_full=norm(raw.get("service_name_full")),
                original_filename=norm(raw.get("original_filename")),
                document_id=norm(raw.get("document_id")),
                table_id=table_id,
                table_number=norm(raw.get("table_number")),
                appendix_number=norm(raw.get("appendix_number")),
                table_title=norm(raw.get("table_title")),
                rows_count_declared=raw.get("rows_count"),
            )

        table = tables[table_id]
        header_schema_json = as_mapping(raw.get("header_schema_json"))
        table_metadata_json = as_mapping(raw.get("table_metadata_json"))
        row_json = as_mapping(raw.get("row_json"))
        normalized_row_json = as_mapping(raw.get("normalized_row_json"))
        row_metadata_json = as_mapping(raw.get("row_metadata_json"))
        row_summary = norm(raw.get("row_summary"))

        table.header_schema_keys.update(key for key, _ in flatten_json(header_schema_json) if key)
        table.header_schema_keys.update(key for key, _ in flatten_json(table_metadata_json) if key.startswith("header") or key.startswith("columns"))
        table.row_json_keys.update(key for key, _ in flatten_json(row_json) if key)
        table.normalized_row_json_keys.update(key for key, _ in flatten_json(normalized_row_json) if key)
        table.row_metadata_keys.update(key for key, _ in flatten_json(row_metadata_json) if key)

        if not raw.get("row_id"):
            continue

        table.rows_count_loaded += 1
        row_metadata_with_summary = dict(row_metadata_json)
        row_metadata_with_summary.setdefault("row_summary", row_summary)

        npp = find_npp(row_json, normalized_row_json, row_metadata_with_summary)
        document_text = find_document_text(row_json, normalized_row_json, row_metadata_json, row_summary)
        requirement_group = find_requirement_group(normalized_row_json, row_metadata_json, row_json)
        channel_values, channel_hits = collect_channel_values(row_json, normalized_row_json, row_metadata_json)
        row_kind = classify_row_kind(npp, document_text, requirement_group, channel_values)

        table.channel_key_hits.update(channel_hits)
        if npp:
            table.npp_rows_count += 1
        depth = npp_depth(npp)
        if depth > 1:
            table.dotted_npp_rows_count += 1
        table.max_npp_depth = max(table.max_npp_depth, depth)
        if row_kind == "section":
            table.section_rows_count += 1
        elif row_kind == "group":
            table.group_like_rows_count += 1
        elif row_kind == "document":
            table.document_like_rows_count += 1
        elif row_kind == "amendment_note":
            table.amendment_rows_count += 1
        elif row_kind == "bez_statement":
            table.bez_statement_rows_count += 1
        if channel_values.get("mfc"):
            table.mfc_non_empty_rows += 1
        if channel_values.get("epgu"):
            table.epgu_non_empty_rows += 1
        if channel_values.get("rpgu"):
            table.rpgu_non_empty_rows += 1
        if channel_values.get("in_person"):
            table.in_person_non_empty_rows += 1
        if channel_values.get("postal"):
            table.postal_non_empty_rows += 1
        if channel_values.get("common_submission"):
            table.common_submission_non_empty_rows += 1

        for source_name, obj in (
            ("header_schema_json", header_schema_json),
            ("row_json", row_json),
            ("normalized_row_json", normalized_row_json),
            ("row_metadata_json", row_metadata_json),
        ):
            for key_path, value in flatten_json(obj):
                value_text = norm(value)
                if not value_text:
                    continue
                for channel, patterns in COMPILED_CHANNEL_PATTERNS.items():
                    if key_matches(key_path, patterns) or (channel == "common_submission" and text_matches(value_text, patterns)):
                        counter_key = (source_name, channel, key_path)
                        key_counter[counter_key][value_text[:220]] += 1

        rows.append(
            RowDiagnostics(
                service_key=table.service_key,
                service_name_short=table.service_name_short,
                service_name_full=table.service_name_full,
                original_filename=table.original_filename,
                document_id=table.document_id,
                table_id=table.table_id,
                table_number=table.table_number,
                appendix_number=table.appendix_number,
                table_title=table.table_title,
                row_id=norm(raw.get("row_id")),
                row_order=raw.get("row_order"),
                npp=npp,
                row_kind=row_kind,
                requirement_group=requirement_group,
                document_text=document_text,
                mfc=channel_values.get("mfc", ""),
                epgu=channel_values.get("epgu", ""),
                rpgu=channel_values.get("rpgu", ""),
                in_person=channel_values.get("in_person", ""),
                postal=channel_values.get("postal", ""),
                common_submission=channel_values.get("common_submission", ""),
                channel_key_hits=join_unique(channel_hits, limit=40, max_chars=1800),
                normalized_keys=format_keys(normalized_row_json),
                raw_keys=format_keys(row_json),
                metadata_keys=format_keys(row_metadata_json),
                row_summary=row_summary,
            )
        )

    key_rows: list[dict[str, Any]] = []
    for (source_name, channel, key_path), counter in sorted(key_counter.items()):
        key_rows.append(
            {
                "source": source_name,
                "channel": channel,
                "key_path": key_path,
                "non_empty_count": sum(counter.values()),
                "distinct_values_count": len(counter),
                "sample_values": join_unique(counter.keys(), limit=8, max_chars=1600),
            }
        )

    return list(tables.values()), rows, key_rows


def write_tsv(path: Path, rows: list[Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            if hasattr(row, "__dataclass_fields__"):
                data = {name: getattr(row, name) for name in fieldnames}
            else:
                data = dict(row)
            writer.writerow(data)


def table_to_dict(table: TableDiagnostics) -> dict[str, Any]:
    return {
        "service_key": table.service_key,
        "service_name_short": table.service_name_short,
        "service_name_full": table.service_name_full,
        "original_filename": table.original_filename,
        "document_id": table.document_id,
        "table_id": table.table_id,
        "table_number": table.table_number,
        "appendix_number": table.appendix_number,
        "table_title": table.table_title,
        "rows_count_declared": table.rows_count_declared,
        "rows_count_loaded": table.rows_count_loaded,
        "npp_rows_count": table.npp_rows_count,
        "dotted_npp_rows_count": table.dotted_npp_rows_count,
        "max_npp_depth": table.max_npp_depth,
        "section_rows_count": table.section_rows_count,
        "group_like_rows_count": table.group_like_rows_count,
        "document_like_rows_count": table.document_like_rows_count,
        "amendment_rows_count": table.amendment_rows_count,
        "bez_statement_rows_count": table.bez_statement_rows_count,
        "mfc_non_empty_rows": table.mfc_non_empty_rows,
        "epgu_non_empty_rows": table.epgu_non_empty_rows,
        "rpgu_non_empty_rows": table.rpgu_non_empty_rows,
        "in_person_non_empty_rows": table.in_person_non_empty_rows,
        "postal_non_empty_rows": table.postal_non_empty_rows,
        "common_submission_non_empty_rows": table.common_submission_non_empty_rows,
        "header_schema_keys": join_unique(sorted(table.header_schema_keys), limit=120, max_chars=3000),
        "row_json_keys": join_unique(sorted(table.row_json_keys), limit=120, max_chars=3000),
        "normalized_row_json_keys": join_unique(sorted(table.normalized_row_json_keys), limit=120, max_chars=3000),
        "row_metadata_keys": join_unique(sorted(table.row_metadata_keys), limit=120, max_chars=3000),
        "channel_key_hits": join_unique(sorted(table.channel_key_hits), limit=120, max_chars=3000),
    }


def build_summary(tables: list[TableDiagnostics], rows: list[RowDiagnostics], key_rows: list[dict[str, Any]]) -> dict[str, Any]:
    row_kind_counts = Counter(row.row_kind for row in rows)
    max_depth_counts = Counter(str(table.max_npp_depth) for table in tables)
    column_format_counts = Counter()
    for table in tables:
        if table.mfc_non_empty_rows or table.epgu_non_empty_rows or table.rpgu_non_empty_rows or table.in_person_non_empty_rows or table.postal_non_empty_rows:
            column_format_counts["separate_channel_columns_detected"] += 1
        elif table.common_submission_non_empty_rows:
            column_format_counts["common_submission_column_detected"] += 1
        else:
            column_format_counts["no_submission_channel_detected"] += 1

    channel_non_empty_tables = {
        "mfc": sum(1 for table in tables if table.mfc_non_empty_rows > 0),
        "epgu": sum(1 for table in tables if table.epgu_non_empty_rows > 0),
        "rpgu": sum(1 for table in tables if table.rpgu_non_empty_rows > 0),
        "in_person": sum(1 for table in tables if table.in_person_non_empty_rows > 0),
        "postal": sum(1 for table in tables if table.postal_non_empty_rows > 0),
        "common_submission": sum(1 for table in tables if table.common_submission_non_empty_rows > 0),
    }

    return {
        "tables_count": len(tables),
        "rows_count": len(rows),
        "services_count": len({table.service_key for table in tables if table.service_key}),
        "row_kind_counts": dict(row_kind_counts),
        "max_npp_depth_counts": dict(max_depth_counts),
        "column_format_counts": dict(column_format_counts),
        "channel_non_empty_tables": channel_non_empty_tables,
        "channel_key_rows_count": len(key_rows),
    }


def write_markdown(path: Path, summary: dict[str, Any], tables: list[TableDiagnostics], rows: list[RowDiagnostics]) -> None:
    most_structured = sorted(tables, key=lambda t: (t.max_npp_depth, t.group_like_rows_count, t.bez_statement_rows_count), reverse=True)[:20]
    channel_problem_candidates = [
        table for table in tables
        if table.max_npp_depth >= 2 and table.mfc_non_empty_rows == 0 and table.common_submission_non_empty_rows == 0
    ][:30]

    lines: list[str] = []
    lines.append("# Аудит способов подачи в таблицах документов")
    lines.append("")
    lines.append("## Сводка")
    lines.append("")
    for key, value in summary.items():
        lines.append(f"- **{key}**: `{json.dumps(value, ensure_ascii=False)}`")
    lines.append("")
    lines.append("## Самые структурные таблицы")
    lines.append("")
    for table in most_structured:
        lines.append(
            f"- `{table.service_key}` — {table.service_name_short}; "
            f"depth={table.max_npp_depth}, groups={table.group_like_rows_count}, "
            f"bez={table.bez_statement_rows_count}, mfc_rows={table.mfc_non_empty_rows}, "
            f"common_submission_rows={table.common_submission_non_empty_rows}"
        )
    lines.append("")
    lines.append("## Таблицы, где стоит вручную проверить ключи каналов подачи")
    lines.append("")
    if channel_problem_candidates:
        for table in channel_problem_candidates:
            lines.append(
                f"- `{table.service_key}` — {table.service_name_short}; "
                f"depth={table.max_npp_depth}, rows={table.rows_count_loaded}, "
                f"mfc_rows=0, common_submission_rows=0, file={table.original_filename}"
            )
    else:
        lines.append("- Явных кандидатов не найдено.")
    lines.append("")
    lines.append("## Пример строк для проверки")
    lines.append("")
    for row in rows[:80]:
        if row.mfc or row.epgu or row.common_submission or row.row_kind in {"group", "bez_statement"}:
            lines.append(f"### {row.service_key} / row_order={row.row_order} / n={row.npp} / {row.row_kind}")
            lines.append("")
            lines.append(f"Документ: {row.document_text[:600]}")
            if row.mfc:
                lines.append(f"- МФЦ: {row.mfc[:600]}")
            if row.epgu:
                lines.append(f"- ЕПГУ: {row.epgu[:600]}")
            if row.rpgu:
                lines.append(f"- РПГУ: {row.rpgu[:600]}")
            if row.common_submission:
                lines.append(f"- Общий способ подачи: {row.common_submission[:600]}")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit submission-channel fields in active document tables. Read-only diagnostic script."
    )
    parser.add_argument("--out-dir", default="/home/logs/document_submission_channels", help="Output directory.")
    parser.add_argument("--service-key", default="", help="Optional exact service_key filter.")
    parser.add_argument("--service-name", default="", help="Optional service short/full name substring filter, case-insensitive.")
    parser.add_argument("--filename", default="", help="Optional original filename substring filter, case-insensitive.")
    parser.add_argument("--json", action="store_true", help="Print summary JSON to stdout.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = load_settings()
    manager = DatabaseSessionManager(settings.database)
    manager.initialize()

    try:
        raw_rows = await fetch_rows(manager, args)
        tables, rows, key_rows = build_diagnostics(raw_rows)
        table_dicts = [table_to_dict(table) for table in tables]
        summary = build_summary(tables, rows, key_rows)

        write_tsv(
            out_dir / "document_submission_tables.tsv",
            table_dicts,
            [
                "service_key", "service_name_short", "service_name_full", "original_filename",
                "document_id", "table_id", "table_number", "appendix_number", "table_title",
                "rows_count_declared", "rows_count_loaded", "npp_rows_count", "dotted_npp_rows_count",
                "max_npp_depth", "section_rows_count", "group_like_rows_count", "document_like_rows_count",
                "amendment_rows_count", "bez_statement_rows_count", "mfc_non_empty_rows", "epgu_non_empty_rows",
                "rpgu_non_empty_rows", "in_person_non_empty_rows", "postal_non_empty_rows",
                "common_submission_non_empty_rows", "header_schema_keys", "row_json_keys",
                "normalized_row_json_keys", "row_metadata_keys", "channel_key_hits",
            ],
        )
        write_tsv(
            out_dir / "document_submission_rows.tsv",
            rows,
            [
                "service_key", "service_name_short", "service_name_full", "original_filename",
                "document_id", "table_id", "table_number", "appendix_number", "table_title",
                "row_id", "row_order", "npp", "row_kind", "requirement_group", "document_text",
                "mfc", "epgu", "rpgu", "in_person", "postal", "common_submission",
                "channel_key_hits", "normalized_keys", "raw_keys", "metadata_keys", "row_summary",
            ],
        )
        write_tsv(
            out_dir / "document_submission_channel_keys.tsv",
            key_rows,
            ["source", "channel", "key_path", "non_empty_count", "distinct_values_count", "sample_values"],
        )
        (out_dir / "document_submission_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_markdown(out_dir / "document_submission_audit.md", summary, tables, rows)

        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            print("DOCUMENT SUBMISSION CHANNEL AUDIT")
            print("=================================")
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            print(f"\nОтчёты записаны в: {out_dir}")
        return 0
    finally:
        await manager.dispose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
