# ============================================================
# File: scripts/run_message_understanding_api_probe.py
# Purpose:
#   Probe actual HTTP API path for optional LLM message understanding diagnostics.
#
# Version:
#   second_step_45_message_understanding_api_probe_guard_aware_v1
#
# Notes:
#   - Uses POST /api/v1/answer with debug=true.
#   - Forces message understanding through request_metadata_json when the API
#     process has the service enabled.
#   - If API service is disabled by env, the report will show disabled policy.
# ============================================================

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VERSION = "second_step_45_message_understanding_api_probe_guard_aware_v1"
DEFAULT_CASES_FILE = "scripts/temp/message_understanding_regression_cases.tsv"


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = _load_cases(Path(args.cases_file), limit=args.limit)
    if not cases:
        raise SystemExit("Не найдено ни одного сценария для проверки")

    started_at = time.perf_counter()
    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        if not args.quiet:
            print(f"[{index}/{len(cases)}] {case['case_id']}: {case['question']}", flush=True)
        one_started = time.perf_counter()
        response = _post_answer(args.url, case, timeout=args.timeout_seconds)
        elapsed = round(time.perf_counter() - one_started, 4)
        debug = response.get("debug") or {}
        call_policy, understanding = _extract_message_understanding_debug(debug)
        message_guard = _extract_message_guard(debug)
        findings = _findings(case, response, call_policy, understanding)
        result = {
            "case": case,
            "ok": not findings,
            "status_code": response.get("_status_code"),
            "elapsed_seconds": elapsed,
            "answer_mode": response.get("answer_mode"),
            "message_understanding_call_policy": call_policy,
            "message_understanding": understanding,
            "message_guard": message_guard,
            "findings": findings,
            "answer_start": str(response.get("answer_text") or "").replace("\n", " ")[:500],
        }
        results.append(result)
        _write_json(out_dir / f"{case['case_id']}.json", {"request_case": case, "response": response, "result": result})

    summary = _build_summary(results=results, elapsed_seconds=round(time.perf_counter() - started_at, 4), args=args)
    _write_json(out_dir / "summary.json", summary)
    _write_jsonl(out_dir / "results.jsonl", results)
    _write_tsv(out_dir / "summary.tsv", results)
    _write_findings_md(out_dir / "findings.md", summary, results)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.fail_on_findings and summary["cases_with_findings"]:
        raise SystemExit(2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/api/v1/answer")
    parser.add_argument("--cases-file", default=DEFAULT_CASES_FILE)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--fail-on-findings", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _load_cases(path: Path, *, limit: int) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл сценариев: {path}")
    cases: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row_number, row in enumerate(reader, start=2):
            question = (row.get("question") or "").strip()
            if not question:
                continue
            case = {str(k): str(v or "").strip() for k, v in row.items()}
            case["case_id"] = case.get("case_id") or f"row_{row_number}"
            cases.append(case)
            if limit and len(cases) >= limit:
                break
    return cases


def _post_answer(url: str, case: dict[str, str], *, timeout: int) -> dict[str, Any]:
    body = {
        "question_text": case["question"],
        "channel": "test",
        "external_user_id": "second_step_43_probe",
        "external_chat_id": case["case_id"],
        "debug": True,
        "request_metadata_json": {
            "force_message_understanding": True,
            "message_understanding_probe": True,
            "probe_case_id": case["case_id"],
        },
    }
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            payload["_status_code"] = response.status
            return payload
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except Exception:
            payload = {"raw": raw}
        payload["_status_code"] = exc.code
        return payload



def _extract_message_understanding_debug(debug: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract message-understanding diagnostics from API debug payload.

    API debug is intentionally broad and may contain routing data under
    answer_payload_json/runtime_answer_service/debug_payload_json. This helper
    accepts both future top-level fields and the current nested structure.
    """
    if not isinstance(debug, dict):
        return {}, {}

    direct_policy = debug.get("message_understanding_call_policy")
    direct_understanding = debug.get("message_understanding")
    if isinstance(direct_policy, dict) or isinstance(direct_understanding, dict):
        return (
            direct_policy if isinstance(direct_policy, dict) else {},
            direct_understanding if isinstance(direct_understanding, dict) else {},
        )

    policy = _find_first_dict_by_key(debug, "message_understanding_call_policy")
    understanding = _find_first_dict_by_key(debug, "message_understanding")

    return (
        policy if isinstance(policy, dict) else {},
        understanding if isinstance(understanding, dict) else {},
    )


def _find_first_dict_by_key(value: Any, target_key: str) -> dict[str, Any] | None:
    if isinstance(value, dict):
        if target_key in value and isinstance(value[target_key], dict):
            return value[target_key]
        for item in value.values():
            found = _find_first_dict_by_key(item, target_key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_first_dict_by_key(item, target_key)
            if found is not None:
                return found
    return None


def _findings(
    case: dict[str, str],
    response: dict[str, Any],
    call_policy: dict[str, Any],
    understanding: dict[str, Any],
) -> list[str]:
    findings: list[str] = []
    if response.get("_status_code") != 200 or response.get("ok") is not True:
        findings.append("api_not_ok")
        return findings

    guard_payload = _extract_message_guard(response.get("debug") or {})
    if _is_message_guard_blocked(guard_payload):
        return _findings_for_guard_blocked_case(case=case, response=response, guard_payload=guard_payload)

    if not call_policy:
        findings.append("missing_message_understanding_call_policy")
        return findings

    if call_policy.get("should_call") is not True:
        findings.append(f"message_understanding_not_called_reason={call_policy.get('reason')}")
        return findings

    provider_status = str(understanding.get("provider_status") or "")
    if provider_status != "ok":
        findings.append(f"provider_status={provider_status or 'empty'}")
        return findings

    expected_intent = (case.get("expected_intent") or "").strip().lower()
    actual_intent = str(understanding.get("intent") or "").strip().lower()
    if expected_intent and actual_intent != expected_intent:
        findings.append(f"intent_expected_{expected_intent}_got_{actual_intent or 'empty'}")

    expected_discovery = (case.get("expected_needs_service_discovery") or "").strip().lower()
    if expected_discovery in {"true", "false"}:
        expected = expected_discovery == "true"
        actual = bool(understanding.get("needs_service_discovery"))
        if actual != expected:
            findings.append(f"service_discovery_expected_{expected}_got_{actual}")

    expected_territory = (case.get("expected_territory") or "").strip().lower()
    if expected_territory:
        actual_territory = str(understanding.get("territory") or "").strip().lower()
        if expected_territory not in actual_territory:
            findings.append(f"territory_missing_{case.get('expected_territory')}")

    return findings


def _extract_message_guard(debug: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(debug, dict):
        return {}
    direct = debug.get("message_guard")
    if isinstance(direct, dict):
        return direct
    found = _find_first_dict_by_key(debug, "message_guard")
    return found if isinstance(found, dict) else {}


def _is_message_guard_blocked(guard_payload: dict[str, Any]) -> bool:
    if not isinstance(guard_payload, dict):
        return False
    if guard_payload.get("should_run_rag") is False:
        return True
    reason_code = str(guard_payload.get("reason_code") or "").strip().lower()
    message_kind = str(guard_payload.get("message_kind") or "").strip().lower()
    return reason_code in {"technical_ping", "bot_command", "empty_message", "out_of_domain", "unsafe_input"} or message_kind in {"technical_ping", "bot_command", "empty", "out_of_domain", "unsafe"}


def _findings_for_guard_blocked_case(
    *,
    case: dict[str, str],
    response: dict[str, Any],
    guard_payload: dict[str, Any],
) -> list[str]:
    findings: list[str] = []
    expected_domain = (case.get("expected_supported_domain") or "").strip().lower()
    expected_intent = (case.get("expected_intent") or "").strip().lower()

    if expected_domain == "true" and expected_intent != "other":
        findings.append("message_guard_blocked_domain_question")

    if response.get("answer_mode") != "safe_no_answer":
        findings.append(f"guard_answer_mode_expected_safe_no_answer_got_{response.get('answer_mode') or 'empty'}")

    answer_text = str(response.get("answer_text") or "").strip()
    if not answer_text:
        findings.append("guard_answer_text_empty")

    if not guard_payload.get("reason_code"):
        findings.append("message_guard_reason_missing")

    return findings


def _build_summary(*, results: list[dict[str, Any]], elapsed_seconds: float, args: argparse.Namespace) -> dict[str, Any]:
    call_reasons: dict[str, int] = {}
    provider_statuses: dict[str, int] = {}
    for row in results:
        guard_payload = row.get("message_guard") or {}
        if _is_message_guard_blocked(guard_payload):
            reason = f"message_guard:{guard_payload.get('reason_code') or 'blocked'}"
            status = "not_called_message_guard"
        else:
            reason = str((row.get("message_understanding_call_policy") or {}).get("reason") or "empty")
            status = str((row.get("message_understanding") or {}).get("provider_status") or "empty")
        call_reasons[reason] = call_reasons.get(reason, 0) + 1
        provider_statuses[status] = provider_statuses.get(status, 0) + 1
    return {
        "ok": all(bool(row.get("ok")) for row in results),
        "version": VERSION,
        "cases_total": len(results),
        "cases_ok": sum(1 for row in results if row.get("ok")),
        "cases_with_findings": sum(1 for row in results if not row.get("ok")),
        "call_reasons": call_reasons,
        "provider_statuses": provider_statuses,
        "elapsed_seconds": elapsed_seconds,
        "url": args.url,
        "cases_file": args.cases_file,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "case_id",
        "ok",
        "status_code",
        "elapsed_seconds",
        "answer_mode",
        "should_call",
        "call_reason",
        "provider_status",
        "expected_intent",
        "actual_intent",
        "confidence",
        "expected_discovery",
        "actual_discovery",
        "findings",
        "answer_start",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for row in rows:
            case = row["case"]
            policy = row.get("message_understanding_call_policy") or {}
            understanding = row.get("message_understanding") or {}
            guard_payload = row.get("message_guard") or {}
            guard_blocked = _is_message_guard_blocked(guard_payload)
            writer.writerow(
                {
                    "case_id": case.get("case_id"),
                    "ok": row.get("ok"),
                    "status_code": row.get("status_code"),
                    "elapsed_seconds": row.get("elapsed_seconds"),
                    "answer_mode": row.get("answer_mode"),
                    "should_call": False if guard_blocked else policy.get("should_call"),
                    "call_reason": f"message_guard:{guard_payload.get('reason_code') or 'blocked'}" if guard_blocked else policy.get("reason"),
                    "provider_status": "not_called_message_guard" if guard_blocked else understanding.get("provider_status"),
                    "expected_intent": case.get("expected_intent"),
                    "actual_intent": understanding.get("intent"),
                    "confidence": understanding.get("confidence"),
                    "expected_discovery": case.get("expected_needs_service_discovery"),
                    "actual_discovery": understanding.get("needs_service_discovery"),
                    "findings": ";".join(row.get("findings") or []),
                    "answer_start": row.get("answer_start"),
                }
            )


def _write_findings_md(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Message understanding API probe findings",
        "",
        f"version: `{VERSION}`",
        f"cases_total: {summary['cases_total']}",
        f"cases_ok: {summary['cases_ok']}",
        f"cases_with_findings: {summary['cases_with_findings']}",
        "",
    ]
    for row in rows:
        case = row["case"]
        policy = row.get("message_understanding_call_policy") or {}
        understanding = row.get("message_understanding") or {}
        guard_payload = row.get("message_guard") or {}
        guard_blocked = _is_message_guard_blocked(guard_payload)
        findings = row.get("findings") or []
        lines.extend(
            [
                f"## {case.get('case_id')}",
                f"- ok: {row.get('ok')}",
                f"- answer_mode: {row.get('answer_mode')}",
                f"- should_call: {False if guard_blocked else policy.get('should_call')}",
                f"- call_reason: {('message_guard:' + str(guard_payload.get('reason_code') or 'blocked')) if guard_blocked else (policy.get('reason') or '')}",
                f"- provider_status: {'not_called_message_guard' if guard_blocked else (understanding.get('provider_status') or '')}",
                f"- expected_intent: {case.get('expected_intent')}",
                f"- actual_intent: {understanding.get('intent') or ''}",
                f"- confidence: {understanding.get('confidence') or ''}",
                f"- findings: {'; '.join(findings) if findings else 'нет автоматических флагов'}",
                "",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
