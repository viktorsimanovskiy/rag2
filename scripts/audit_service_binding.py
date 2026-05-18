from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config.settings import load_settings
from app.db.session import DatabaseSessionManager


SUMMARY_SQL = """
select
    (select count(*) from public.service_registry) as service_registry_total,
    (select count(*) from public.service_registry where is_active = true) as active_services,
    (select count(*) from public.document_registry where status = 'active') as active_documents,
    (
        select count(*)
        from public.document_registry
        where status = 'active'
          and service_key is null
    ) as active_documents_without_service_key,
    (
        select count(*)
        from public.document_registry dr
        left join public.service_registry sr on sr.service_key = dr.service_key
        where dr.status = 'active'
          and dr.service_key is not null
          and sr.service_key is null
    ) as active_documents_with_broken_service_link,
    (
        select count(*)
        from public.service_registry sr
        left join public.document_registry dr
          on dr.service_key = sr.service_key
         and dr.status = 'active'
        where sr.is_active = true
          and dr.document_id is null
    ) as active_services_without_active_document,
    (
        select count(*)
        from public.service_registry
        where aliases_json is null
           or jsonb_array_length(aliases_json) = 0
    ) as services_without_aliases;
"""

SERVICE_DUPLICATES_SQL = """
select
    dr.service_key,
    count(*) as active_documents_count,
    string_agg(coalesce(dr.original_filename, '<no filename>'), ' | ' order by dr.original_filename) as files
from public.document_registry dr
where dr.status = 'active'
group by dr.service_key
having count(*) <> 1
order by count(*) desc, dr.service_key nulls first
limit :limit;
"""

KEY_TABLES_SQL = """
with table_counts as (
    select
        dr.document_id,
        dr.service_key,
        dr.service_name_short,
        dr.original_filename,
        count(*) filter (where dt.table_type = 'identifiers') as identifiers_tables,
        count(*) filter (where dt.table_type = 'documents') as documents_tables,
        count(*) filter (where dt.table_type = 'refusal_reasons') as refusal_reasons_tables,
        count(*) filter (where dt.table_type = 'consultant_noise') as consultant_noise_tables
    from public.document_registry dr
    left join public.document_tables dt on dt.document_id = dr.document_id
    where dr.status = 'active'
    group by dr.document_id, dr.service_key, dr.service_name_short, dr.original_filename
)
select *
from table_counts
where identifiers_tables <> 1
   or documents_tables <> 1
   or refusal_reasons_tables <> 1
   or consultant_noise_tables <> 0
order by original_filename
limit :limit;
"""

UNEXPECTED_REQUIREMENT_GROUP_SQL = """
select
    dr.service_key,
    dr.service_name_short,
    dr.original_filename,
    coalesce(
        nullif(dtr.normalized_row_json ->> 'requirement_group', ''),
        nullif(dtr.metadata_json ->> 'requirement_group', ''),
        '<empty>'
    ) as requirement_group,
    count(*) as rows_count
from public.document_registry dr
join public.document_tables dt on dt.document_id = dr.document_id
join public.document_table_rows dtr on dtr.table_id = dt.table_id
where dr.status = 'active'
  and dt.table_type = 'documents'
  and coalesce(
        nullif(dtr.normalized_row_json ->> 'requirement_group', ''),
        nullif(dtr.metadata_json ->> 'requirement_group', ''),
        '<empty>'
      ) not in ('required', 'optional')
group by dr.service_key, dr.service_name_short, dr.original_filename, requirement_group
order by dr.original_filename, requirement_group
limit :limit;
"""

REFUSAL_ROWS_WITHOUT_SCOPE_SQL = """
select
    dr.service_key,
    dr.service_name_short,
    dr.original_filename,
    count(*) as rows_without_scope
from public.document_registry dr
join public.document_tables dt on dt.document_id = dr.document_id
join public.document_table_rows dtr on dtr.table_id = dt.table_id
where dr.status = 'active'
  and dt.table_type = 'refusal_reasons'
  and coalesce(
        nullif(dtr.normalized_row_json ->> 'row_scope', ''),
        nullif(dtr.metadata_json ->> 'row_scope', '')
      ) is null
group by dr.service_key, dr.service_name_short, dr.original_filename
order by dr.original_filename
limit :limit;
"""


async def fetch_all(manager: DatabaseSessionManager, sql: str, *, limit: int) -> list[dict[str, Any]]:
    async with manager.session_scope() as session:
        result = await session.execute(text(sql), {"limit": limit})
        return [dict(row) for row in result.mappings().all()]


async def fetch_one(manager: DatabaseSessionManager, sql: str) -> dict[str, Any]:
    async with manager.session_scope() as session:
        result = await session.execute(text(sql))
        return dict(result.mappings().one())


def build_verdict(report: dict[str, Any]) -> str:
    summary = report["summary"]
    problems = []

    expected_summary = {
        "service_registry_total": 110,
        "active_services": 110,
        "active_documents": 110,
        "active_documents_without_service_key": 0,
        "active_documents_with_broken_service_link": 0,
        "active_services_without_active_document": 0,
        "services_without_aliases": 0,
    }

    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            problems.append(f"{key}: expected {expected}, got {summary.get(key)}")

    for section in (
        "service_duplicates",
        "key_table_problems",
        "unexpected_requirement_groups",
        "refusal_rows_without_scope",
    ):
        if report.get(section):
            problems.append(f"{section}: {len(report[section])} sample row(s)")

    return "ok" if not problems else "problems_found"


async def main() -> int:
    parser = argparse.ArgumentParser(description="Audit service_registry binding after full DOCX ingestion.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum sample rows per problem section.")
    args = parser.parse_args()

    settings = load_settings()
    manager = DatabaseSessionManager(settings.database)
    manager.initialize()

    try:
        report = {
            "summary": await fetch_one(manager, SUMMARY_SQL),
            "service_duplicates": await fetch_all(manager, SERVICE_DUPLICATES_SQL, limit=args.limit),
            "key_table_problems": await fetch_all(manager, KEY_TABLES_SQL, limit=args.limit),
            "unexpected_requirement_groups": await fetch_all(
                manager,
                UNEXPECTED_REQUIREMENT_GROUP_SQL,
                limit=args.limit,
            ),
            "refusal_rows_without_scope": await fetch_all(
                manager,
                REFUSAL_ROWS_WITHOUT_SCOPE_SQL,
                limit=args.limit,
            ),
        }
        report["verdict"] = build_verdict(report)

        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        else:
            print("SERVICE BINDING AUDIT")
            print("=====================")
            print(f"verdict: {report['verdict']}")
            print("\nsummary:")
            for key, value in report["summary"].items():
                print(f"  {key}: {value}")

            for section in (
                "service_duplicates",
                "key_table_problems",
                "unexpected_requirement_groups",
                "refusal_rows_without_scope",
            ):
                rows = report[section]
                print(f"\n{section}: {len(rows)}")
                for row in rows[: args.limit]:
                    print("  - " + json.dumps(row, ensure_ascii=False, default=str))

        return 0 if report["verdict"] == "ok" else 1
    finally:
        await manager.dispose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
