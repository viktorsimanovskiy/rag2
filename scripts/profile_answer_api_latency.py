# ============================================================
# File: scripts/profile_answer_api_latency.py
# Purpose:
#   Profile HTTP API answer latency for a small set of questions.
#
# This script does not change the database. It calls /api/v1/answer with
# debug=true, saves every raw response and extracts timing fields exposed by
# RAG2. It is intended to separate:
#   - total HTTP/API time;
#   - runtime service resolution;
#   - retrieval;
#   - generation;
#   - unknown/unaccounted time.
# ============================================================

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_QUESTIONS = [
    "я ветеран труда края какие документы нужны для получения едв",
    "список документов для тжс через мфц",
    "какие документы нужны для тжс через мфц",
    "полный перечень документов для тжс через мфц",
    "в какой форме подавать документы для тжс через мфц — оригиналы или копии?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Профилирование времени ответа RAG2 API.")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/api/v1/answer",
        help="Адрес endpoint /api/v1/answer.",
    )
    parser.add_argument(
        "--questions-file",
        default="",
        help="Текстовый файл с вопросами, по одному вопросу на строку.",
    )
    parser.add_argument(
        "--question",
        action="append",
        default=[],
        help="Один вопрос. Можно указать несколько раз.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Папка для отчётов.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="Сколько раз повторить каждый вопрос.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Сколько прогревочных запросов выполнить перед измерением.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=120,
        help="HTTP timeout.",
    )
    parser.add_argument(
        "--channel",
        default="test_console",
        help="Канал запроса.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Пауза между запросами.",
    )
    return parser.parse_args()


def read_questions(args: argparse.Namespace) -> list[str]:
    result: list[str] = []
    result.extend(q.strip() for q in args.question if q and q.strip())

    if args.questions_file:
        path = Path(args.questions_file)
        for line in path.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            result.append(value)

    if not result:
        result = list(DEFAULT_QUESTIONS)

    deduped: list[str] = []
    seen: set[str] = set()
    for question in result:
        key = " ".join(question.split()).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(question)
    return deduped


def post_answer(
    *,
    url: str,
    question: str,
    channel: str,
    external_user_id: str,
    external_chat_id: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    payload = {
        "question_text": question,
        "channel": channel,
        "external_user_id": external_user_id,
        "external_chat_id": external_chat_id,
        "debug": True,
    }
    raw_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        url,
        data=raw_body,
        method="POST",
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
        },
    )

    started = time.perf_counter()
    report: dict[str, Any] = {
        "ok": False,
        "question": question,
        "request": payload,
        "elapsed_seconds": None,
        "status_code": None,
        "response": None,
        "error": None,
    }
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
            response_json = json.loads(body)
            report["status_code"] = int(response.status)
            report["response"] = response_json
            report["ok"] = bool(response_json.get("ok"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        report["status_code"] = int(exc.code)
        report["error"] = body
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        report["error"] = repr(exc)
    report["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    return report


def dig(payload: Any, path: list[str], default: Any = None) -> Any:
    current = payload
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return current if current is not None else default


def as_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def extract_row(report: dict[str, Any], *, run_index: int, warmup: bool) -> dict[str, Any]:
    response = report.get("response") or {}
    debug = response.get("debug") or {}
    answer_payload = dig(debug, ["answer_payload_json"], {}) or {}
    debug_payload = dig(debug, ["debug_payload_json"], {}) or {}
    orchestrator_timings = debug_payload.get("orchestrator_timings_sec") or {}
    session_details = orchestrator_timings.get("session_resolution_details") or {}

    runtime_timings = answer_payload.get("runtime_answer_service_timings") or {}
    runtime_payload = answer_payload.get("runtime_answer_service_runtime_payload") or {}
    service_resolution = response.get("service_resolution") or {}
    service_resolution_debug = service_resolution.get("debug_payload_json") or {}
    service_resolver_timings = service_resolution_debug.get("timings_sec") or {}
    generation_timings = answer_payload.get("generation_timings_sec") or {}

    elapsed = as_float(report.get("elapsed_seconds"))
    runtime_total = as_float(runtime_timings.get("total_sec"))
    service_resolution_sec = as_float(runtime_timings.get("service_resolution_sec"))
    retrieval_sec = as_float(runtime_timings.get("retrieval_sec"))
    generation_sec = as_float(runtime_timings.get("generation_sec"))
    service_discovery_sec = as_float(runtime_timings.get("service_discovery_sec"))

    known_runtime = sum(
        as_float(runtime_timings.get(key))
        for key in [
            "validation_sec",
            "query_terms_sec",
            "service_resolution_sec",
            "build_retrieval_input_sec",
            "retrieval_sec",
            "build_generation_request_sec",
            "generation_sec",
            "enrich_generation_result_sec",
            "service_discovery_sec",
        ]
    )

    orchestrator_total = as_float(orchestrator_timings.get("total_sec"))
    orchestrator_known = sum(
        as_float(orchestrator_timings.get(key))
        for key in [
            "validate_input_sec",
            "resolve_or_create_session_sec",
            "build_question_routing_sec",
            "create_question_event_sec",
            "reuse_gate_sec",
            "run_full_generation_sec",
            "persist_generated_answer_event_sec",
            "persist_reused_answer_event_sec",
            "sampling_policy_sec",
            "build_outgoing_payload_sec",
        ]
    )

    return {
        "warmup": warmup,
        "run_index": run_index,
        "ok": report.get("ok"),
        "status_code": report.get("status_code"),
        "question": report.get("question"),
        "elapsed_seconds": elapsed,
        "orchestrator_total_sec": orchestrator_total,
        "orchestrator_http_overhead_sec": round(max(elapsed - orchestrator_total, 0.0), 6),
        "orchestrator_unaccounted_sec": round(max(orchestrator_total - orchestrator_known, 0.0), 6),
        "orchestrator_validate_input_sec": as_float(orchestrator_timings.get("validate_input_sec")),
        "orchestrator_resolve_or_create_session_sec": as_float(orchestrator_timings.get("resolve_or_create_session_sec")),
        "session_channel_lookup_sec": as_float(session_details.get("channel_lookup_sec")),
        "session_lookup_sec": as_float(session_details.get("session_lookup_sec")),
        "session_commit_refresh_sec": as_float(session_details.get("commit_refresh_sec")),
        "session_created": session_details.get("session_created"),
        "session_existing_fast_path": session_details.get("existing_session_fast_path"),
        "orchestrator_build_question_routing_sec": as_float(orchestrator_timings.get("build_question_routing_sec")),
        "orchestrator_create_question_event_sec": as_float(orchestrator_timings.get("create_question_event_sec")),
        "orchestrator_reuse_gate_sec": as_float(orchestrator_timings.get("reuse_gate_sec")),
        "orchestrator_run_full_generation_sec": as_float(orchestrator_timings.get("run_full_generation_sec")),
        "orchestrator_persist_generated_answer_event_sec": as_float(orchestrator_timings.get("persist_generated_answer_event_sec")),
        "orchestrator_sampling_policy_sec": as_float(orchestrator_timings.get("sampling_policy_sec")),
        "orchestrator_build_outgoing_payload_sec": as_float(orchestrator_timings.get("build_outgoing_payload_sec")),
        "question_embedding_skipped": orchestrator_timings.get("question_embedding_skipped"),
        "reuse_skipped": orchestrator_timings.get("reuse_skipped"),
        "runtime_total_sec": runtime_total,
        "runtime_unaccounted_sec": round(max(runtime_total - known_runtime, 0.0), 6),
        "http_overhead_or_persistence_sec": round(max(elapsed - runtime_total, 0.0), 6),
        "service_resolution_sec": service_resolution_sec,
        "service_resolver_total_sec": as_float(service_resolver_timings.get("total")),
        "service_resolver_index_cache_hit": service_resolution_debug.get("index_cache_hit"),
        "service_resolver_load_or_get_index_sec": as_float(service_resolver_timings.get("load_or_get_index")),
        "retrieval_sec": retrieval_sec,
        "generation_sec": generation_sec,
        "generation_total_sec": as_float(generation_timings.get("total_sec")),
        "prepare_documents_answer_sec": as_float(generation_timings.get("prepare_documents_answer_sec")),
        "service_discovery_sec": service_discovery_sec,
        "answer_mode": response.get("answer_mode"),
        "was_reused": response.get("was_reused"),
        "service_status": service_resolution.get("resolution_status"),
        "service_key": service_resolution.get("service_key"),
        "service_name_short": service_resolution.get("service_name_short"),
        "strategy_code": answer_payload.get("strategy_code") or runtime_payload.get("strategy_code"),
        "answer_length": len(str(response.get("answer_text") or "")),
        "error": report.get("error"),
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * p))
    return ordered[max(0, min(index, len(ordered) - 1))]


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [r for r in rows if not r.get("warmup")]
    elapsed = [as_float(r.get("elapsed_seconds")) for r in measured]
    retrieval = [as_float(r.get("retrieval_sec")) for r in measured]
    resolution = [as_float(r.get("service_resolution_sec")) for r in measured]
    generation = [as_float(r.get("generation_sec")) for r in measured]
    overhead = [as_float(r.get("http_overhead_or_persistence_sec")) for r in measured]
    orchestrator_total = [as_float(r.get("orchestrator_total_sec")) for r in measured]
    orchestrator_routing = [as_float(r.get("orchestrator_build_question_routing_sec")) for r in measured]
    session_channel_lookup = [as_float(r.get("session_channel_lookup_sec")) for r in measured]
    session_lookup = [as_float(r.get("session_lookup_sec")) for r in measured]
    session_commit_refresh = [as_float(r.get("session_commit_refresh_sec")) for r in measured]
    orchestrator_persist = [as_float(r.get("orchestrator_persist_generated_answer_event_sec")) for r in measured]
    orchestrator_http_overhead = [as_float(r.get("orchestrator_http_overhead_sec")) for r in measured]

    return {
        "measured_requests": len(measured),
        "ok_requests": sum(1 for r in measured if r.get("ok")),
        "elapsed_avg_sec": round(statistics.mean(elapsed), 6) if elapsed else 0.0,
        "elapsed_p95_sec": round(percentile(elapsed, 0.95), 6),
        "elapsed_max_sec": round(max(elapsed), 6) if elapsed else 0.0,
        "orchestrator_total_avg_sec": round(statistics.mean(orchestrator_total), 6) if orchestrator_total else 0.0,
        "orchestrator_build_question_routing_avg_sec": round(statistics.mean(orchestrator_routing), 6) if orchestrator_routing else 0.0,
        "session_channel_lookup_avg_sec": round(statistics.mean(session_channel_lookup), 6) if session_channel_lookup else 0.0,
        "session_lookup_avg_sec": round(statistics.mean(session_lookup), 6) if session_lookup else 0.0,
        "session_commit_refresh_avg_sec": round(statistics.mean(session_commit_refresh), 6) if session_commit_refresh else 0.0,
        "orchestrator_persist_generated_answer_event_avg_sec": round(statistics.mean(orchestrator_persist), 6) if orchestrator_persist else 0.0,
        "orchestrator_http_overhead_avg_sec": round(statistics.mean(orchestrator_http_overhead), 6) if orchestrator_http_overhead else 0.0,
        "retrieval_avg_sec": round(statistics.mean(retrieval), 6) if retrieval else 0.0,
        "service_resolution_avg_sec": round(statistics.mean(resolution), 6) if resolution else 0.0,
        "generation_avg_sec": round(statistics.mean(generation), 6) if generation else 0.0,
        "http_overhead_or_persistence_avg_sec": round(statistics.mean(overhead), 6) if overhead else 0.0,
    }


def write_summary_md(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    slowest = sorted(
        [r for r in rows if not r.get("warmup")],
        key=lambda r: as_float(r.get("elapsed_seconds")),
        reverse=True,
    )[:10]

    lines = [
        "# Профиль времени ответа RAG2 API",
        "",
        f"Всего измеренных запросов: {summary['measured_requests']}",
        f"Успешных запросов: {summary['ok_requests']}",
        f"Среднее время HTTP: {summary['elapsed_avg_sec']} сек.",
        f"95-й процентиль HTTP: {summary['elapsed_p95_sec']} сек.",
        f"Максимум HTTP: {summary['elapsed_max_sec']} сек.",
        "",
        "## Средние внутренние времена",
        "",
        f"- orchestrator_total: {summary.get('orchestrator_total_avg_sec', 0.0)} сек.",
        f"- orchestrator_build_question_routing: {summary.get('orchestrator_build_question_routing_avg_sec', 0.0)} сек.",
        f"- session_channel_lookup: {summary.get('session_channel_lookup_avg_sec', 0.0)} сек.",
        f"- session_lookup: {summary.get('session_lookup_avg_sec', 0.0)} сек.",
        f"- session_commit_refresh: {summary.get('session_commit_refresh_avg_sec', 0.0)} сек.",
        f"- orchestrator_persist_generated_answer_event: {summary.get('orchestrator_persist_generated_answer_event_avg_sec', 0.0)} сек.",
        f"- orchestrator_http_overhead: {summary.get('orchestrator_http_overhead_avg_sec', 0.0)} сек.",
        f"- service_resolution: {summary['service_resolution_avg_sec']} сек.",
        f"- retrieval: {summary['retrieval_avg_sec']} сек.",
        f"- generation: {summary['generation_avg_sec']} сек.",
        f"- legacy HTTP/persistence overhead: {summary['http_overhead_or_persistence_avg_sec']} сек.",
        "",
        "## Самые медленные запросы",
        "",
    ]
    for row in slowest:
        lines.append(
            f"- {row['elapsed_seconds']} сек. | orchestrator={row.get('orchestrator_total_sec')} | "
            f"routing={row.get('orchestrator_build_question_routing_sec')} | "
            f"session={row.get('orchestrator_resolve_or_create_session_sec')} "
            f"(lookup={row.get('session_lookup_sec')}, commit={row.get('session_commit_refresh_sec')}, "
            f"fast={row.get('session_existing_fast_path')}) | "
            f"persist={row.get('orchestrator_persist_generated_answer_event_sec')} | "
            f"retrieval={row['retrieval_sec']} | resolver={row['service_resolution_sec']} | "
            f"generation={row['generation_sec']} | http_overhead={row.get('orchestrator_http_overhead_sec')} | "
            f"{row['question']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    questions = read_questions(args)

    raw_path = out_dir / "raw_responses.jsonl"
    rows: list[dict[str, Any]] = []

    with raw_path.open("w", encoding="utf-8") as raw_file:
        request_index = 0
        for question_index, question in enumerate(questions, start=1):
            for warmup_index in range(args.warmup_runs):
                request_index += 1
                report = post_answer(
                    url=args.url,
                    question=question,
                    channel=args.channel,
                    external_user_id="latency_profile_user",
                    external_chat_id=f"latency_profile_chat_{question_index}",
                    timeout_seconds=args.timeout_seconds,
                )
                raw_file.write(json.dumps(report, ensure_ascii=False) + "\n")
                rows.append(extract_row(report, run_index=warmup_index + 1, warmup=True))
                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)

            for run_index in range(args.runs):
                request_index += 1
                report = post_answer(
                    url=args.url,
                    question=question,
                    channel=args.channel,
                    external_user_id="latency_profile_user",
                    external_chat_id=f"latency_profile_chat_{question_index}",
                    timeout_seconds=args.timeout_seconds,
                )
                raw_file.write(json.dumps(report, ensure_ascii=False) + "\n")
                row = extract_row(report, run_index=run_index + 1, warmup=False)
                rows.append(row)
                print(
                    f"[{question_index}/{len(questions)} run {run_index + 1}] "
                    f"{row['elapsed_seconds']} сек. | "
                    f"retrieval={row['retrieval_sec']} | "
                    f"resolver={row['service_resolution_sec']} | "
                    f"generation={row['generation_sec']} | "
                    f"{question}"
                )
                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)

    write_tsv(out_dir / "latency_rows.tsv", rows)
    summary = build_summary(rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary_md(out_dir / "summary.md", summary, rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
