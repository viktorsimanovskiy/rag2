# ============================================================
# File: scripts/run_llm_answer_composer_api_probe.py
# Purpose:
#   Shadow diagnostic runner for optional LLM answer composer.
#
# Version:
#   second_step_47_llm_answer_composer_api_probe_v1
#
# Notes:
#   - Calls current HTTP API to get deterministic grounded answers.
#   - Then calls LLMAnswerComposerService in shadow mode.
#   - Does not change API behavior and does not replace user-facing answers.
# ============================================================

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config.settings import load_settings
from app.integrations.openai.client_factory import OpenAIClientFactory
from app.services.generation.llm_answer_composer import (
    LLMAnswerComposerConfig,
    LLMAnswerComposerService,
    input_from_api_response,
)

VERSION = "second_step_47_llm_answer_composer_api_probe_v1"
DEFAULT_CASES_FILE = "scripts/temp/llm_answer_composer_regression_cases.tsv"


@dataclass(slots=True, frozen=True)
class ComposerCase:
    case_id: str
    category: str
    question: str
    expected_compose: bool
    comment: str


async def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = _load_cases(Path(args.cases_file), limit=args.limit)
    if not cases:
        raise SystemExit("Не найдено ни одного сценария для проверки")

    settings = load_settings()
    client = OpenAIClientFactory(settings.openai).create_async_client()
    composer = LLMAnswerComposerService(
        client,
        config=LLMAnswerComposerConfig(
            enabled=True,
            mode=args.mode,
            model_name=args.model or settings.message_understanding.model_name,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            request_timeout_seconds=args.timeout_seconds,
            max_output_chars=args.max_output_chars,
        ),
    )

    started_at = time.perf_counter()
    results: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        if not args.quiet:
            print(f"[{index}/{len(cases)}] {case.case_id}: {case.question}", flush=True)

        one_started = time.perf_counter()
        api_response = _call_api(args.url, case, timeout_seconds=args.api_timeout_seconds)
        api_elapsed = round(time.perf_counter() - one_started, 4)
        response_payload = api_response.get("response") or {}
        if not isinstance(response_payload, dict):
            response_payload = {}

        composer_payload = input_from_api_response(case.question, response_payload)
        composer_started = time.perf_counter()
        composer_result = await composer.compose(composer_payload)
        composer_elapsed = round(time.perf_counter() - composer_started, 4)

        result = {
            "case": _case_to_payload(case),
            "api_status_code": api_response.get("status_code"),
            "api_ok": bool(response_payload.get("ok")),
            "api_answer_mode": response_payload.get("answer_mode"),
            "api_answer_text": response_payload.get("answer_text"),
            "api_citations_count": len(response_payload.get("citations") or []),
            "api_message_guard": _extract_message_guard(response_payload),
            "composer": composer_result.to_payload(),
            "elapsed_seconds": {
                "api": api_elapsed,
                "composer": composer_elapsed,
            },
        }
        findings = _findings(case, result)
        result["ok"] = not findings
        result["findings"] = findings
        results.append(result)
        _write_json(out_dir / f"{case.case_id}.json", result)

    summary = _build_summary(cases, results, elapsed_seconds=round(time.perf_counter() - started_at, 4), args=args)
    _write_json(out_dir / "summary.json", summary)
    _write_jsonl(out_dir / "results.jsonl", results)
    _write_summary_tsv(out_dir / "summary.tsv", results)
    _write_findings_md(out_dir / "findings.md", summary, results)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.fail_on_findings and summary["cases_with_findings"]:
        raise SystemExit(2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--cases-file", default=DEFAULT_CASES_FILE)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mode", default="shadow", choices=["shadow", "assist", "disabled"])
    parser.add_argument("--model", default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=1000)
    parser.add_argument("--max-output-chars", type=int, default=4500)
    parser.add_argument("--timeout-seconds", type=int, default=35)
    parser.add_argument("--api-timeout-seconds", type=int, default=60)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fail-on-findings", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _load_cases(path: Path, *, limit: int) -> list[ComposerCase]:
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл сценариев: {path}")
    cases: list[ComposerCase] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        required = {"case_id", "category", "question", "expected_compose", "comment"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError("В файле сценариев не хватает колонок: " + ", ".join(missing))
        for row_number, row in enumerate(reader, start=2):
            question = (row.get("question") or "").strip()
            if not question:
                continue
            expected_raw = (row.get("expected_compose") or "").strip().lower()
            cases.append(
                ComposerCase(
                    case_id=(row.get("case_id") or f"row_{row_number}").strip(),
                    category=(row.get("category") or "").strip(),
                    question=question,
                    expected_compose=expected_raw == "true",
                    comment=(row.get("comment") or "").strip(),
                )
            )
            if limit and len(cases) >= limit:
                break
    return cases


def _call_api(url: str, case: ComposerCase, *, timeout_seconds: int) -> dict[str, Any]:
    payload = {
        "question_text": case.question,
        "channel": "test",
        "external_session_id": f"llm_answer_composer_step47:{case.case_id}",
        "external_user_id": "second_step_47_probe",
        "external_chat_id": case.case_id,
        "debug": True,
        "request_metadata_json": {
            "force_message_understanding": True,
            "llm_answer_composer_probe": True,
            "probe_case_id": case.case_id,
        },
    }
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=raw,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
            parsed = json.loads(body) if body else {}
            if isinstance(parsed, dict):
                parsed["_status_code"] = response.status
            return {"status_code": response.status, "response": parsed}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body) if body else {}
        except json.JSONDecodeError:
            parsed = {"raw_body": body}
        if isinstance(parsed, dict):
            parsed["_status_code"] = exc.code
        return {"status_code": exc.code, "response": parsed}


def _findings(case: ComposerCase, result: dict[str, Any]) -> list[str]:
    findings: list[str] = []

    if not result.get("api_ok"):
        findings.append(f"api_not_ok_status_{result.get('api_status_code')}")
        return findings

    message_guard = result.get("api_message_guard") or {}
    guard_should_run = message_guard.get("should_run_rag")
    composer = result.get("composer") or {}
    status = str(composer.get("status") or "")
    provider_status = str(composer.get("provider_status") or "")
    violations = composer.get("grounding_violations") or []

    if guard_should_run is False:
        if provider_status != "not_called" or status != "skipped":
            findings.append("message_guard_case_should_not_call_composer")
        return findings

    validation_warnings = [str(item) for item in (composer.get("validation_warnings") or [])]
    skip_reason = validation_warnings[0] if validation_warnings else ""
    source_not_composable = (
        skip_reason.startswith("safe_no_answer_not_composed")
        or skip_reason.startswith("answer_question_mismatch:")
        or skip_reason in {"no_citations", "empty_deterministic_answer"}
    )

    if case.expected_compose:
        if source_not_composable:
            # This is not a composer failure: the deterministic answer is not a safe source
            # for stylistic rewriting. Keep it visible in validation_warnings.
            pass
        else:
            if provider_status != "ok":
                findings.append(f"composer_provider_status_{provider_status or 'empty'}")
            if status not in {"ok", "rejected"}:
                findings.append(f"composer_unexpected_status_{status or 'empty'}")
            if violations:
                findings.append("composer_grounding_violations=" + ",".join(str(item) for item in violations))
    else:
        if status == "ok":
            findings.append("composer_should_skip_but_status_ok")

    final_answer = str(composer.get("final_answer_text") or "").strip()
    if case.expected_compose and not final_answer:
        findings.append("empty_final_answer")

    return findings


def _extract_message_guard(response_payload: dict[str, Any]) -> dict[str, Any]:
    debug = response_payload.get("debug")
    if not isinstance(debug, dict):
        return {}
    delivery = debug.get("delivery_payload_json")
    if isinstance(delivery, dict) and isinstance(delivery.get("message_guard"), dict):
        return dict(delivery.get("message_guard") or {})
    debug_payload = debug.get("debug_payload_json")
    if isinstance(debug_payload, dict) and isinstance(debug_payload.get("message_guard"), dict):
        return dict(debug_payload.get("message_guard") or {})
    return {}


def _build_summary(cases: list[ComposerCase], results: list[dict[str, Any]], *, elapsed_seconds: float, args: argparse.Namespace) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    provider_statuses: dict[str, int] = {}
    for result in results:
        composer = result.get("composer") or {}
        statuses[str(composer.get("status") or "empty")] = statuses.get(str(composer.get("status") or "empty"), 0) + 1
        provider_statuses[str(composer.get("provider_status") or "empty")] = provider_statuses.get(str(composer.get("provider_status") or "empty"), 0) + 1
    return {
        "ok": not any(result.get("findings") for result in results),
        "version": VERSION,
        "cases_total": len(cases),
        "cases_ok": sum(1 for result in results if not result.get("findings")),
        "cases_with_findings": sum(1 for result in results if result.get("findings")),
        "composer_statuses": statuses,
        "provider_statuses": provider_statuses,
        "elapsed_seconds": elapsed_seconds,
        "url": args.url,
        "mode": args.mode,
        "model": args.model,
        "cases_file": args.cases_file,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_summary_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow([
            "case_id",
            "category",
            "api_ok",
            "api_answer_mode",
            "citations_count",
            "composer_provider_status",
            "composer_status",
            "violations",
            "findings",
        ])
        for row in rows:
            case = row.get("case") or {}
            composer = row.get("composer") or {}
            writer.writerow([
                case.get("case_id"),
                case.get("category"),
                row.get("api_ok"),
                row.get("api_answer_mode"),
                row.get("api_citations_count"),
                composer.get("provider_status"),
                composer.get("status"),
                ",".join(str(item) for item in (composer.get("grounding_violations") or [])),
                ",".join(str(item) for item in (composer.get("validation_warnings") or [])),
                ",".join(str(item) for item in (row.get("findings") or [])),
            ])


def _write_findings_md(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# LLM answer composer probe",
        "",
        f"version: `{summary['version']}`",
        f"ok: `{summary['ok']}`",
        f"cases_total: `{summary['cases_total']}`",
        f"cases_with_findings: `{summary['cases_with_findings']}`",
        "",
    ]
    for row in rows:
        findings = row.get("findings") or []
        if not findings:
            continue
        case = row.get("case") or {}
        lines.extend([
            f"## {case.get('case_id')} — {case.get('category')}",
            "",
            f"question: {case.get('question')}",
            f"api_answer_mode: `{row.get('api_answer_mode')}`",
            f"findings: `{', '.join(str(item) for item in findings)}`",
            "",
        ])
    if summary["cases_with_findings"] == 0:
        lines.append("Диагностических флагов нет.")
    path.write_text("\n".join(lines), encoding="utf-8")


def _case_to_payload(case: ComposerCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "category": case.category,
        "question": case.question,
        "expected_compose": case.expected_compose,
        "comment": case.comment,
    }


if __name__ == "__main__":
    asyncio.run(main())
