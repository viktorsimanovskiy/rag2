#!/usr/bin/env python3
# ============================================================
# File: scripts/run_llm_answer_composer_runtime_assist_probe.py
# Purpose:
#   Probe runtime LLM answer composer in test-only assist mode.
#
# Version:
#   second_step_50_runtime_llm_answer_composer_test_assist_probe_v1
#
# Safety:
#   The probe requests replacement only with channel=test and debug=true.
#   Production channels must not replace answers even when the request metadata
#   contains force_llm_answer_composer_replacement=true.
# ============================================================

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

VERSION = "second_step_50_runtime_llm_answer_composer_test_assist_probe_v1"
DEFAULT_CASES_FILE = "scripts/temp/llm_answer_composer_regression_cases.tsv"


@dataclass(frozen=True)
class ProbeCase:
    case_id: str
    category: str
    question: str
    expected_compose: bool
    channel: str = "test"
    force_replacement: bool = True
    comment: str = ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe runtime LLM answer composer in controlled test-only assist mode."
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000/api/v1/answer")
    parser.add_argument("--cases-file", default=DEFAULT_CASES_FILE)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument(
        "--include-non-test-negative",
        action="store_true",
        help="Also verify that a forced replacement request on a production-like channel is blocked.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = _load_cases(Path(args.cases_file))
    if args.include_non_test_negative and cases:
        base = next((item for item in cases if item.expected_compose), cases[0])
        cases.append(
            ProbeCase(
                case_id=f"{base.case_id}_NON_TEST_NEGATIVE",
                category="non_test_suppression",
                question=base.question,
                expected_compose=False,
                channel="telegram",
                force_replacement=True,
                comment="Forced replacement must be blocked outside test_console channel.",
            )
        )

    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    for case in cases:
        result = _run_one_case(case, url=args.url, timeout=args.timeout)
        results.append(result)
        (out_dir / f"{case.case_id}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary = _build_summary(results, started_at=started, url=args.url, cases_file=args.cases_file)
    _write_outputs(out_dir, results, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["ok"]:
        raise SystemExit(1)


def _load_cases(path: Path) -> list[ProbeCase]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        result: list[ProbeCase] = []
        for row in reader:
            case_id = str(row.get("case_id") or "").strip()
            question = str(row.get("question") or "").strip()
            if not case_id or not question:
                continue
            result.append(
                ProbeCase(
                    case_id=case_id,
                    category=str(row.get("category") or "").strip(),
                    question=question,
                    expected_compose=_parse_bool(row.get("expected_compose")),
                    comment=str(row.get("comment") or "").strip(),
                )
            )
    return result


def _run_one_case(case: ProbeCase, *, url: str, timeout: int) -> dict[str, Any]:
    started = time.perf_counter()
    request_payload = {
        "question_text": case.question,
        "channel": case.channel,
        "external_user_id": "second_step_50_runtime_composer_assist_probe",
        "external_chat_id": "second_step_50_runtime_composer_assist_probe",
        "external_session_id": (
            f"second_step_50_runtime_composer_assist_probe:"
            f"{case.case_id}:{int(time.time() * 1000)}"
        ),
        "debug": True,
        "request_metadata_json": {
            "second_step_50_runtime_llm_answer_composer_assist_probe": True,
            "case_id": case.case_id,
            "force_llm_answer_composer_replacement": bool(case.force_replacement),
        },
    }

    http_status = 0
    response_payload: dict[str, Any] = {}
    error: str | None = None
    try:
        response_payload, http_status = _post_json(url, request_payload, timeout=timeout)
    except urllib.error.HTTPError as exc:
        http_status = int(exc.code or 0)
        try:
            body = exc.read().decode("utf-8")
            response_payload = json.loads(body) if body else {}
        except Exception:
            response_payload = {}
        error = repr(exc)
    except Exception as exc:
        error = repr(exc)

    elapsed = round(time.perf_counter() - started, 4)
    result: dict[str, Any] = {
        "version": VERSION,
        "case": asdict(case),
        "request_payload": request_payload,
        "http_status": http_status,
        "api_ok": bool(response_payload.get("ok")) if isinstance(response_payload, dict) else False,
        "answer_mode": str(response_payload.get("answer_mode") or "") if isinstance(response_payload, dict) else "",
        "answer_text": str(response_payload.get("answer_text") or "") if isinstance(response_payload, dict) else "",
        "answer_text_short": str(response_payload.get("answer_text_short") or "") if isinstance(response_payload, dict) else "",
        "citations_count": len(response_payload.get("citations") or []) if isinstance(response_payload, dict) else 0,
        "error": error,
        "elapsed_seconds": elapsed,
    }

    debug = response_payload.get("debug") if isinstance(response_payload, dict) else None
    if not isinstance(debug, dict):
        debug = {}
    answer_payload_json = debug.get("answer_payload_json") if isinstance(debug.get("answer_payload_json"), dict) else {}
    debug_payload_json = debug.get("debug_payload_json") if isinstance(debug.get("debug_payload_json"), dict) else {}

    composer = answer_payload_json.get("llm_answer_composer")
    if not isinstance(composer, dict):
        composer = debug_payload_json.get("llm_answer_composer") if isinstance(debug_payload_json.get("llm_answer_composer"), dict) else {}

    policy = answer_payload_json.get("llm_answer_composer_call_policy")
    if not isinstance(policy, dict):
        policy = debug_payload_json.get("llm_answer_composer_call_policy") if isinstance(debug_payload_json.get("llm_answer_composer_call_policy"), dict) else {}

    message_guard = debug_payload_json.get("message_guard") if isinstance(debug_payload_json.get("message_guard"), dict) else {}

    result["llm_answer_composer_call_policy"] = policy
    result["llm_answer_composer"] = composer
    result["message_guard"] = message_guard
    result["findings"] = _case_findings(case, result)
    result["ok"] = not result["findings"]
    return result


def _post_json(url: str, payload: dict[str, Any], *, timeout: int) -> tuple[dict[str, Any], int]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response_body = response.read().decode("utf-8")
        parsed = json.loads(response_body) if response_body else {}
        return parsed, int(response.status)


def _case_findings(case: ProbeCase, result: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    if result.get("http_status") != 200 or not result.get("api_ok"):
        findings.append("api_not_ok")
        return findings

    answer_text = str(result.get("answer_text") or "")
    answer_mode = str(result.get("answer_mode") or "")
    composer = result.get("llm_answer_composer") if isinstance(result.get("llm_answer_composer"), dict) else {}
    policy = result.get("llm_answer_composer_call_policy") if isinstance(result.get("llm_answer_composer_call_policy"), dict) else {}
    message_guard = result.get("message_guard") if isinstance(result.get("message_guard"), dict) else {}

    guard_blocked = message_guard and message_guard.get("should_run_rag") is False
    if case.category == "guard":
        if not guard_blocked:
            findings.append("guard_case_not_blocked_before_rag")
        if composer:
            findings.append("guard_case_should_not_have_runtime_composer")
        if not answer_text.strip():
            findings.append("guard_case_empty_answer")
        return findings

    if not policy:
        findings.append("missing_llm_answer_composer_call_policy")
    elif policy.get("enabled") is not True:
        findings.append("runtime_composer_service_not_enabled")

    if not composer:
        findings.append("missing_llm_answer_composer_payload")
        return findings

    status = str(composer.get("status") or "")
    provider_status = str(composer.get("provider_status") or "")
    validation_warnings = [str(item) for item in (composer.get("validation_warnings") or [])]
    violations = [str(item) for item in (composer.get("grounding_violations") or [])]
    replacement_applied = composer.get("runtime_replacement_applied") is True
    replacement_suppressed = composer.get("runtime_replacement_suppressed") is True
    replacement_reason = str(composer.get("runtime_replacement_reason") or "")
    composed = str(composer.get("composed_answer_text") or "")
    original = str(composer.get("original_answer_text") or "")
    final = str(composer.get("final_answer_text") or "")

    if case.channel != "test":
        if replacement_applied:
            findings.append("non_test_channel_replacement_applied")
        if replacement_reason != "blocked_non_test_channel":
            findings.append(f"non_test_channel_unexpected_replacement_reason:{replacement_reason or 'empty'}")
        return findings

    if answer_mode == "safe_no_answer":
        if replacement_applied:
            findings.append("safe_no_answer_replacement_applied")
        if status != "skipped":
            findings.append(f"safe_no_answer_composer_status_{status or 'empty'}")
        return findings

    if any(item.startswith("answer_question_mismatch:") for item in validation_warnings):
        if replacement_applied:
            findings.append("mismatch_answer_replacement_applied")
        if status != "skipped":
            findings.append("mismatch_answer_should_skip_composer")
        return findings

    if not case.expected_compose:
        if replacement_applied:
            findings.append("case_expected_skip_but_replacement_applied")
        if status != "skipped":
            findings.append("case_expected_skip_but_composer_not_skipped")
        return findings

    if provider_status != "ok":
        findings.append(f"composer_provider_status_{provider_status or 'empty'}")
    if status != "ok":
        findings.append(f"composer_status_{status or 'empty'}")
    if violations:
        findings.append("composer_grounding_violations=" + ",".join(violations))
    if not replacement_applied:
        findings.append(f"expected_runtime_replacement_not_applied:{replacement_reason or 'empty'}")
    if not composed.strip():
        findings.append("empty_composed_answer")
    if replacement_applied and answer_text != composed:
        findings.append("api_answer_not_equal_composed_answer")
    if replacement_applied and final != answer_text:
        findings.append("composer_final_answer_not_equal_api_answer")
    if replacement_applied and not original.strip():
        findings.append("missing_original_answer_text")
    if replacement_applied and replacement_suppressed:
        findings.append("replacement_marked_both_applied_and_suppressed")
    return findings


def _build_summary(results: list[dict[str, Any]], *, started_at: float, url: str, cases_file: str) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    provider_statuses: dict[str, int] = {}
    replacements = {"applied": 0, "suppressed": 0, "not_present": 0}
    for row in results:
        composer = row.get("llm_answer_composer") if isinstance(row.get("llm_answer_composer"), dict) else {}
        statuses[str(composer.get("status") or "not_present")] = statuses.get(str(composer.get("status") or "not_present"), 0) + 1
        provider_statuses[str(composer.get("provider_status") or "not_present")] = provider_statuses.get(str(composer.get("provider_status") or "not_present"), 0) + 1
        if not composer:
            replacements["not_present"] += 1
        elif composer.get("runtime_replacement_applied") is True:
            replacements["applied"] += 1
        else:
            replacements["suppressed"] += 1

    with_findings = [row for row in results if row.get("findings")]
    return {
        "ok": not with_findings,
        "version": VERSION,
        "cases_total": len(results),
        "cases_ok": len(results) - len(with_findings),
        "cases_with_findings": len(with_findings),
        "composer_statuses": statuses,
        "provider_statuses": provider_statuses,
        "runtime_replacements": replacements,
        "elapsed_seconds": round(time.perf_counter() - started_at, 4),
        "url": url,
        "cases_file": cases_file,
    }


def _write_outputs(out_dir: Path, results: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (out_dir / "summary.tsv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "case_id",
            "category",
            "channel",
            "api_ok",
            "answer_mode",
            "citations_count",
            "policy_enabled",
            "policy_replacement_enabled",
            "composer_provider_status",
            "composer_status",
            "runtime_replacement_applied",
            "runtime_replacement_reason",
            "validation_warnings",
            "violations",
            "findings",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in results:
            case = row.get("case") or {}
            composer = row.get("llm_answer_composer") if isinstance(row.get("llm_answer_composer"), dict) else {}
            policy = row.get("llm_answer_composer_call_policy") if isinstance(row.get("llm_answer_composer_call_policy"), dict) else {}
            writer.writerow(
                {
                    "case_id": case.get("case_id"),
                    "category": case.get("category"),
                    "channel": case.get("channel"),
                    "api_ok": row.get("api_ok"),
                    "answer_mode": row.get("answer_mode"),
                    "citations_count": row.get("citations_count"),
                    "policy_enabled": policy.get("enabled"),
                    "policy_replacement_enabled": policy.get("replacement_enabled"),
                    "composer_provider_status": composer.get("provider_status"),
                    "composer_status": composer.get("status"),
                    "runtime_replacement_applied": composer.get("runtime_replacement_applied"),
                    "runtime_replacement_reason": composer.get("runtime_replacement_reason"),
                    "validation_warnings": ";".join(str(item) for item in (composer.get("validation_warnings") or [])),
                    "violations": ";".join(str(item) for item in (composer.get("grounding_violations") or [])),
                    "findings": ";".join(str(item) for item in (row.get("findings") or [])),
                }
            )

    lines = [
        "# Runtime LLM answer composer assist probe",
        "",
        f"version: `{summary['version']}`",
        f"ok: `{summary['ok']}`",
        f"cases_total: `{summary['cases_total']}`",
        f"cases_with_findings: `{summary['cases_with_findings']}`",
        f"runtime_replacements: `{json.dumps(summary['runtime_replacements'], ensure_ascii=False)}`",
        "",
    ]
    with_findings = [row for row in results if row.get("findings")]
    if with_findings:
        lines.append("## Findings")
        for row in with_findings:
            case = row.get("case") or {}
            lines.append(f"- `{case.get('case_id')}`: {', '.join(row.get('findings') or [])}")
    else:
        lines.append("Диагностических флагов нет.")
    (out_dir / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "да"}


if __name__ == "__main__":
    main()
