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
    (select count(*) from public.document_registry where status = 'active') as active_documents,
    (
        select count(*)
        from public.document_registry dr
        where dr.status = 'active'
          and dr.publication_payload_json #>> '{parser_payload_json,docx_preprocessing,mode}' = 'trim_for_rag'
    ) as trim_for_rag_documents,
    (
        select count(*)
        from public.document_registry dr
        where dr.status = 'active'
          and coalesce((dr.publication_payload_json #>> '{parser_payload_json,docx_preprocessing,applied_to_published_content}')::boolean, false) = true
    ) as preprocessing_applied_documents,
    (
        select count(*)
        from public.document_registry dr
        where dr.status = 'active'
          and coalesce((dr.publication_payload_json #>> '{parser_payload_json,docx_preprocessing,trim_safety_passed}')::boolean, false) = true
    ) as trim_safety_passed_documents,
    (
        select count(*)
        from public.document_registry dr
        where dr.status = 'active'
          and coalesce((dr.publication_payload_json #>> '{parser_payload_json,docx_preprocessing,report,tail_contains_core_table}')::boolean, true) = true
    ) as tail_contains_core_table_documents,
    (
        select count(*)
        from public.document_registry dr
        where dr.status = 'active'
          and coalesce((dr.publication_payload_json #>> '{parser_payload_json,docx_preprocessing,report,has_exactly_one_each_core_table}')::boolean, false) = true
    ) as report_core_tables_1_1_1_documents,
    (
        select count(*)
        from public.document_tables dt
        join public.document_registry dr on dr.document_id = dt.document_id
        where dr.status = 'active'
          and dt.table_type in ('form_fields', 'generic')
    ) as residual_form_or_generic_tables,
    (
        select count(*)
        from public.document_tables dt
        join public.document_registry dr on dr.document_id = dt.document_id
        where dr.status = 'active'
          and dt.table_type = 'consultant_noise'
    ) as residual_consultant_noise_tables;
"""

TABLE_TYPE_COUNTS_SQL = """
select
    coalesce(dt.table_type, '<null>') as table_type,
    count(*) as tables_count
from public.document_tables dt
join public.document_registry dr on dr.document_id = dt.document_id
where dr.status = 'active'
group by coalesce(dt.table_type, '<null>')
order by tables_count desc, table_type;
"""

RESIDUAL_BY_DOCUMENT_SQL = """
select
    dr.service_key,
    dr.service_name_short,
    dr.original_filename,
    count(*) filter (where dt.table_type = 'form_fields') as form_fields_tables,
    count(*) filter (where dt.table_type = 'generic') as generic_tables,
    count(*) as total_form_or_generic_tables
from public.document_registry dr
join public.document_tables dt on dt.document_id = dr.document_id
where dr.status = 'active'
  and dt.table_type in ('form_fields', 'generic')
group by dr.service_key, dr.service_name_short, dr.original_filename
order by total_form_or_generic_tables desc, dr.original_filename
limit :limit;
"""

RESIDUAL_TABLE_SAMPLES_SQL = """
select
    dr.service_key,
    dr.service_name_short,
    dr.original_filename,
    dt.table_type,
    dt.table_number,
    dt.appendix_number,
    left(coalesce(dt.table_title, ''), 220) as table_title,
    dt.rows_count,
    left(coalesce(dt.summary, ''), 300) as summary,
    left(coalesce(dt.markdown_preview, ''), 500) as markdown_preview
from public.document_registry dr
join public.document_tables dt on dt.document_id = dr.document_id
where dr.status = 'active'
  and dt.table_type in ('form_fields', 'generic')
order by dr.original_filename, dt.table_type, dt.table_number nulls last, dt.table_id
limit :limit;
"""

PREPROCESSING_PROBLEMS_SQL = """
select
    dr.service_key,
    dr.service_name_short,
    dr.original_filename,
    dr.publication_payload_json #>> '{parser_payload_json,docx_preprocessing,mode}' as mode,
    dr.publication_payload_json #>> '{parser_payload_json,docx_preprocessing,trim_safety_passed}' as trim_safety_passed,
    dr.publication_payload_json #>> '{parser_payload_json,docx_preprocessing,report,warnings}' as warnings,
    dr.publication_payload_json #>> '{parser_payload_json,docx_preprocessing,report,core_table_counts}' as core_table_counts,
    dr.publication_payload_json #>> '{parser_payload_json,docx_preprocessing,report,trimmed_after_last_core_tables_count}' as trimmed_tail_tables
from public.document_registry dr
where dr.status = 'active'
  and (
       dr.publication_payload_json #>> '{parser_payload_json,docx_preprocessing,mode}' is distinct from 'trim_for_rag'
    or coalesce((dr.publication_payload_json #>> '{parser_payload_json,docx_preprocessing,trim_safety_passed}')::boolean, false) = false
    or coalesce((dr.publication_payload_json #>> '{parser_payload_json,docx_preprocessing,report,tail_contains_core_table}')::boolean, true) = true
    or coalesce((dr.publication_payload_json #>> '{parser_payload_json,docx_preprocessing,report,has_exactly_one_each_core_table}')::boolean, false) = false
  )
order by dr.original_filename
limit :limit;
"""

TRIM_TOTALS_SQL = """
select
    sum(coalesce((dr.publication_payload_json #>> '{parser_payload_json,docx_preprocessing,report,removed_before_official_start_count}')::int, 0)) as removed_before_official_start_total,
    sum(coalesce((dr.publication_payload_json #>> '{parser_payload_json,docx_preprocessing,report,removed_consultant_noise_count}')::int, 0)) as removed_consultant_noise_total,
    sum(coalesce((dr.publication_payload_json #>> '{parser_payload_json,docx_preprocessing,report,trimmed_after_last_core_count}')::int, 0)) as trimmed_after_last_core_items_total,
    sum(coalesce((dr.publication_payload_json #>> '{parser_payload_json,docx_preprocessing,report,trimmed_after_last_core_tables_count}')::int, 0)) as trimmed_after_last_core_tables_total,
    sum(coalesce((dr.publication_payload_json #>> '{parser_payload_json,docx_preprocessing,report,trimmed_after_last_core_chars_count}')::int, 0)) as trimmed_after_last_core_chars_total
from public.document_registry dr
where dr.status = 'active';
"""


async def fetch_one(manager: DatabaseSessionManager, sql: str) -> dict[str, Any]:
    async with manager.session_scope() as session:
        result = await session.execute(text(sql))
        return dict(result.mappings().one())


async def fetch_all(manager: DatabaseSessionManager, sql: str, *, limit: int = 50) -> list[dict[str, Any]]:
    async with manager.session_scope() as session:
        result = await session.execute(text(sql), {"limit": limit})
        return [dict(row) for row in result.mappings().all()]


def build_verdict(report: dict[str, Any]) -> str:
    summary = report["summary"]
    problems: list[str] = []

    expected = {
        "active_documents": 110,
        "trim_for_rag_documents": 110,
        "preprocessing_applied_documents": 110,
        "trim_safety_passed_documents": 110,
        "tail_contains_core_table_documents": 0,
        "report_core_tables_1_1_1_documents": 110,
        "residual_consultant_noise_tables": 0,
    }
    for key, expected_value in expected.items():
        if summary.get(key) != expected_value:
            problems.append(f"{key}: expected {expected_value}, got {summary.get(key)}")

    if report.get("preprocessing_problems"):
        problems.append("preprocessing_problems: sample rows present")

    # Residual form/generic tables are not automatically a failed verdict yet.
    # Some can be before the trim boundary. They are reported for the next review step.
    return "ok" if not problems else "problems_found"


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit trim_for_rag ingestion result and remaining form/generic tables."
    )
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    parser.add_argument("--limit", type=int, default=30, help="Maximum sample rows per section.")
    args = parser.parse_args()

    settings = load_settings()
    manager = DatabaseSessionManager(settings.database)
    manager.initialize()

    try:
        report: dict[str, Any] = {
            "summary": await fetch_one(manager, SUMMARY_SQL),
            "trim_totals": await fetch_one(manager, TRIM_TOTALS_SQL),
            "table_type_counts": await fetch_all(manager, TABLE_TYPE_COUNTS_SQL, limit=args.limit),
            "residual_by_document": await fetch_all(manager, RESIDUAL_BY_DOCUMENT_SQL, limit=args.limit),
            "residual_table_samples": await fetch_all(manager, RESIDUAL_TABLE_SAMPLES_SQL, limit=args.limit),
            "preprocessing_problems": await fetch_all(manager, PREPROCESSING_PROBLEMS_SQL, limit=args.limit),
        }
        report["verdict"] = build_verdict(report)

        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        else:
            print("TRIMMED INGESTION AUDIT")
            print("=======================")
            print(f"verdict: {report['verdict']}")
            print("\nsummary:")
            for key, value in report["summary"].items():
                print(f"  {key}: {value}")
            print("\ntrim_totals:")
            for key, value in report["trim_totals"].items():
                print(f"  {key}: {value}")
            print("\ntable_type_counts:")
            for row in report["table_type_counts"]:
                print("  - " + json.dumps(row, ensure_ascii=False, default=str))
            for section in ("preprocessing_problems", "residual_by_document", "residual_table_samples"):
                rows = report[section]
                print(f"\n{section}: {len(rows)} sample row(s)")
                for row in rows[: args.limit]:
                    print("  - " + json.dumps(row, ensure_ascii=False, default=str))

        return 0 if report["verdict"] == "ok" else 1
    finally:
        await manager.dispose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
