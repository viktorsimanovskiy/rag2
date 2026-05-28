from __future__ import annotations

import argparse
import asyncio
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import asyncpg


def read_database_url(env_path: Path) -> str:
    if not env_path.exists():
        raise FileNotFoundError(f"Не найден .env: {env_path}")

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        if key.strip() == "APP_DATABASE_URL":
            value = value.strip().strip('"').strip("'")
            return value.replace("postgresql+asyncpg://", "postgresql://", 1)

    raise RuntimeError("APP_DATABASE_URL не найден в .env")


def value_to_cell(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)

    if isinstance(value, datetime):
        return value.isoformat()

    return str(value)


def write_tsv(path: Path, rows: list[asyncpg.Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    columns = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([value_to_cell(row.get(col)) for col in columns])


async def fetch_rows(conn: asyncpg.Connection, sql: str) -> list[asyncpg.Record]:
    return list(await conn.fetch(sql))


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Выгрузка отчётов по задержкам RAG2 из БД без psql и \\copy."
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Папка для отчётов. Если не указана, будет создана в /home/logs/db_reports/.",
    )
    parser.add_argument(
        "--env-path",
        default=".env",
        help="Путь к .env проекта.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Сколько последних ответов выгружать.",
    )
    args = parser.parse_args()

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = Path("/home/logs/db_reports") / stamp

    out_dir.mkdir(parents=True, exist_ok=True)

    database_url = read_database_url(Path(args.env_path))

    conn = await asyncpg.connect(database_url)
    try:
        last_answers_sql = f"""
        SELECT
            ae.created_at AS answer_created_at,
            qe.created_at AS question_created_at,
            c.channel_code::text AS channel_code,
            cs.external_session_id,
            cs.external_user_id,
            cs.external_chat_id,
            qe.intent_type::text AS intent_type,
            ae.answer_mode::text AS answer_mode,
            ae.reuse_allowed,
            (ae.reused_from_answer_event_id IS NOT NULL) AS was_reused,
            qe.question_text_raw,
            length(ae.answer_text) AS answer_length,
            round(EXTRACT(EPOCH FROM (ae.created_at - qe.created_at))::numeric, 3) AS db_question_to_answer_sec,

            NULLIF(ae.answer_payload_json #>> '{{runtime_answer_service_timings,total_sec}}', '')::numeric AS runtime_total_sec,
            NULLIF(ae.answer_payload_json #>> '{{runtime_answer_service_timings,service_resolution_sec}}', '')::numeric AS service_resolution_sec,
            NULLIF(ae.answer_payload_json #>> '{{runtime_answer_service_timings,retrieval_sec}}', '')::numeric AS retrieval_sec,
            NULLIF(ae.answer_payload_json #>> '{{runtime_answer_service_timings,generation_sec}}', '')::numeric AS generation_sec,

            NULLIF(ae.answer_payload_json #>> '{{answer_orchestrator_timings,total_sec}}', '')::numeric AS orchestrator_total_sec,
            NULLIF(ae.answer_payload_json #>> '{{answer_orchestrator_timings,resolve_or_create_session_sec}}', '')::numeric AS session_total_sec,
            NULLIF(ae.answer_payload_json #>> '{{answer_orchestrator_timings,session_lookup_sec}}', '')::numeric AS session_lookup_sec,
            NULLIF(ae.answer_payload_json #>> '{{answer_orchestrator_timings,session_commit_refresh_sec}}', '')::numeric AS session_commit_refresh_sec,
            NULLIF(ae.answer_payload_json #>> '{{answer_orchestrator_timings,run_full_generation_sec}}', '')::numeric AS run_full_generation_sec,
            NULLIF(ae.answer_payload_json #>> '{{answer_orchestrator_timings,persist_generated_answer_event_sec}}', '')::numeric AS persist_answer_sec,

            ae.answer_payload_json #>> '{{answer_orchestrator_timings,question_embedding_skipped}}' AS question_embedding_skipped,
            ae.answer_payload_json #>> '{{answer_orchestrator_timings,session_existing_fast_path}}' AS session_existing_fast_path,

            ae.answer_payload_json #>> '{{strategy_code}}' AS strategy_code,
            ae.answer_payload_json #>> '{{runtime_answer_service_runtime_payload,service_resolution,status}}' AS service_status,
            ae.answer_payload_json #>> '{{runtime_answer_service_runtime_payload,service_resolution,service_key}}' AS resolved_service_key,
            ae.answer_payload_json #>> '{{runtime_answer_service_runtime_payload,service_resolution,service_name_short}}' AS resolved_service_name_short
        FROM answer_events ae
        JOIN question_events qe ON qe.question_event_id = ae.question_event_id
        JOIN conversation_sessions cs ON cs.session_id = qe.session_id
        JOIN channels c ON c.channel_id = cs.channel_id
        ORDER BY ae.created_at DESC
        LIMIT {int(args.limit)}
        """

        latency_by_channel_sql = """
        WITH rows AS (
            SELECT
                c.channel_code::text AS channel_code,
                qe.intent_type::text AS intent_type,
                ae.answer_mode::text AS answer_mode,
                round(EXTRACT(EPOCH FROM (ae.created_at - qe.created_at))::numeric, 3) AS db_question_to_answer_sec,
                NULLIF(ae.answer_payload_json #>> '{runtime_answer_service_timings,total_sec}', '')::numeric AS runtime_total_sec,
                NULLIF(ae.answer_payload_json #>> '{answer_orchestrator_timings,total_sec}', '')::numeric AS orchestrator_total_sec,
                NULLIF(ae.answer_payload_json #>> '{answer_orchestrator_timings,resolve_or_create_session_sec}', '')::numeric AS session_total_sec,
                NULLIF(ae.answer_payload_json #>> '{answer_orchestrator_timings,session_lookup_sec}', '')::numeric AS session_lookup_sec,
                length(ae.answer_text) AS answer_length
            FROM answer_events ae
            JOIN question_events qe ON qe.question_event_id = ae.question_event_id
            JOIN conversation_sessions cs ON cs.session_id = qe.session_id
            JOIN channels c ON c.channel_id = cs.channel_id
            WHERE ae.created_at >= NOW() - INTERVAL '3 days'
        )
        SELECT
            channel_code,
            intent_type,
            answer_mode,
            count(*) AS cnt,
            round(avg(db_question_to_answer_sec), 3) AS avg_db_sec,
            round(percentile_cont(0.5) WITHIN GROUP (ORDER BY db_question_to_answer_sec)::numeric, 3) AS p50_db_sec,
            round(percentile_cont(0.95) WITHIN GROUP (ORDER BY db_question_to_answer_sec)::numeric, 3) AS p95_db_sec,
            round(max(db_question_to_answer_sec), 3) AS max_db_sec,
            round(avg(runtime_total_sec), 3) AS avg_runtime_sec,
            round(avg(orchestrator_total_sec), 3) AS avg_orchestrator_sec,
            round(avg(session_total_sec), 3) AS avg_session_sec,
            round(avg(session_lookup_sec), 3) AS avg_session_lookup_sec,
            round(avg(answer_length), 0) AS avg_answer_length
        FROM rows
        GROUP BY channel_code, intent_type, answer_mode
        ORDER BY p95_db_sec DESC NULLS LAST, cnt DESC
        """

        sessions_by_channel_sql = """
        SELECT
            c.channel_code::text AS channel_code,
            count(*) AS sessions_count,
            max(cs.session_last_activity_at) AS last_activity_at
        FROM conversation_sessions cs
        JOIN channels c ON c.channel_id = cs.channel_id
        GROUP BY c.channel_code::text
        ORDER BY sessions_count DESC
        """

        indexes_sql = """
        SELECT
            schemaname,
            tablename,
            indexname,
            indexdef
        FROM pg_indexes
        WHERE tablename IN ('conversation_sessions', 'question_events', 'answer_events')
        ORDER BY tablename, indexname
        """

        slowest_sql = """
        SELECT
            ae.created_at AS answer_created_at,
            c.channel_code::text AS channel_code,
            qe.intent_type::text AS intent_type,
            ae.answer_mode::text AS answer_mode,
            round(EXTRACT(EPOCH FROM (ae.created_at - qe.created_at))::numeric, 3) AS db_question_to_answer_sec,
            NULLIF(ae.answer_payload_json #>> '{answer_orchestrator_timings,resolve_or_create_session_sec}', '')::numeric AS session_total_sec,
            NULLIF(ae.answer_payload_json #>> '{answer_orchestrator_timings,session_lookup_sec}', '')::numeric AS session_lookup_sec,
            NULLIF(ae.answer_payload_json #>> '{runtime_answer_service_timings,total_sec}', '')::numeric AS runtime_total_sec,
            left(qe.question_text_raw, 300) AS question_text_raw
        FROM answer_events ae
        JOIN question_events qe ON qe.question_event_id = ae.question_event_id
        JOIN conversation_sessions cs ON cs.session_id = qe.session_id
        JOIN channels c ON c.channel_id = cs.channel_id
        ORDER BY db_question_to_answer_sec DESC NULLS LAST
        LIMIT 100
        """

        reports = {
            "answer_events_last1000.tsv": last_answers_sql,
            "answer_latency_by_channel_intent.tsv": latency_by_channel_sql,
            "sessions_by_channel.tsv": sessions_by_channel_sql,
            "indexes_feedback_tables.tsv": indexes_sql,
            "slowest_answers.tsv": slowest_sql,
        }

        for filename, sql in reports.items():
            rows = await fetch_rows(conn, sql)
            write_tsv(out_dir / filename, rows)

        explain_sql = """
        EXPLAIN (ANALYZE, BUFFERS)
        SELECT
            cs.session_id,
            cs.channel_id,
            cs.external_session_id
        FROM conversation_sessions cs
        WHERE (cs.channel_id, cs.external_session_id) IN (
            SELECT
                recent.channel_id,
                recent.external_session_id
            FROM conversation_sessions recent
            ORDER BY recent.session_last_activity_at DESC
            LIMIT 1
        )
        LIMIT 1
        """
        explain_rows = await conn.fetch(explain_sql)
        (out_dir / "session_lookup_explain.txt").write_text(
            "\n".join(row[0] for row in explain_rows),
            encoding="utf-8",
        )

        meta = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "out_dir": str(out_dir),
            "limit": args.limit,
            "reports": sorted(reports.keys()) + ["session_lookup_explain.txt"],
        }
        (out_dir / "report_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    finally:
        await conn.close()

    print(out_dir)


if __name__ == "__main__":
    asyncio.run(main())
