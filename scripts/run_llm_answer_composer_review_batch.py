#!/usr/bin/env python3
# ============================================================
# File: scripts/run_llm_answer_composer_review_batch.py
# Purpose:
#   Run a broader human-review batch for runtime LLM answer composer.
#
# Version:
#   second_step_53_llm_answer_composer_review_batch_structure_v1
#
# Safety:
#   - Uses channel=test and debug=true.
#   - Requests force_llm_answer_composer_replacement=true only for the test channel.
#   - Does not enable replacement for Telegram/MAX/web.
#   - Produces review artifacts: original answer, composed answer, final answer,
#     validation warnings, grounding violations and replacement reasons.
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
from typing import Any, Iterable

VERSION = "second_step_53_llm_answer_composer_review_batch_structure_v1"
DEFAULT_QUESTIONS_FILE = "scripts/temp/service_question_bank.tsv"


@dataclass(frozen=True)
class ReviewCase:
    case_id: str
    category: str
    question: str
    service_key: str = ""
    service_name_short: str = ""
    expected_service_hint: str = ""
    source_row: dict[str, Any] | None = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a broader test-channel review batch for runtime LLM answer composer. "
            "The script is intended for manual review before enabling any non-test replacement."
        )
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000/api/v1/answer")
    parser.add_argument("--questions-file", default=DEFAULT_QUESTIONS_FILE)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--category", default="01_как_пишет_человек_в_чате")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--sample-per-service", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--no-force-replacement",
        action="store_true",
        help="Do not request test-only answer replacement; useful for shadow-only comparison.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = _load_cases(
        Path(args.questions_file),
        category=args.category,
        limit=args.limit,
        sample_per_service=args.sample_per_service,
    )
    if not cases:
        raise SystemExit(f"Не найдено вопросов для прогона: {args.questions_file}")

    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        result = _run_one_case(
            case,
            url=args.url,
            timeout=args.timeout,
            force_replacement=not args.no_force_replacement,
            ordinal=index,
        )
        results.append(result)
        safe_name = _safe_file_name(case.case_id or f"case_{index:03d}")
        (out_dir / f"{safe_name}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary = _build_summary(
        results,
        started_at=started,
        url=args.url,
        questions_file=args.questions_file,
        category=args.category,
        limit=args.limit,
        sample_per_service=args.sample_per_service,
        force_replacement=not args.no_force_replacement,
    )
    _write_outputs(out_dir, results, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["ok"]:
        raise SystemExit(1)


def _load_cases(path: Path, *, category: str, limit: int, sample_per_service: int) -> list[ReviewCase]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        raw_rows = [dict(row) for row in reader]

    filtered: list[dict[str, Any]] = []
    for row in raw_rows:
        row_category = str(row.get("category") or "").strip()
        if category and row_category != category:
            continue
        question = str(row.get("question_text") or row.get("question") or "").strip()
        if not question:
            continue
        filtered.append(row)

    if sample_per_service > 0:
        per_service: dict[str, int] = {}
        sampled: list[dict[str, Any]] = []
        for row in filtered:
            service_key = str(row.get("service_key") or row.get("expected_service_key") or "").strip()
            bucket = service_key or "__no_service_key__"
            current = per_service.get(bucket, 0)
            if current >= sample_per_service:
                continue
            sampled.append(row)
            per_service[bucket] = current + 1
        filtered = sampled

    if limit and limit > 0:
        filtered = filtered[:limit]

    cases: list[ReviewCase] = []
    for index, row in enumerate(filtered, start=1):
        case_id = str(row.get("question_id") or row.get("case_id") or f"BQ{index:03d}").strip()
        cases.append(
            ReviewCase(
                case_id=case_id,
                category=str(row.get("category") or "").strip(),
                question=str(row.get("question_text") or row.get("question") or "").strip(),
                service_key=str(row.get("service_key") or row.get("expected_service_key") or "").strip(),
                service_name_short=str(row.get("service_name_short") or "").strip(),
                expected_service_hint=str(row.get("expected_service_hint") or "").strip(),
                source_row=row,
            )
        )
    return cases


def _run_one_case(
    case: ReviewCase,
    *,
    url: str,
    timeout: int,
    force_replacement: bool,
    ordinal: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    request_payload = {
        "question_text": case.question,
        "channel": "test",
        "external_user_id": "second_step_52_composer_review_batch",
        "external_chat_id": "second_step_52_composer_review_batch",
        "external_session_id": (
            "second_step_52_composer_review_batch:"
            f"{case.case_id}:{ordinal}:{int(time.time() * 1000)}"
        ),
        "debug": True,
        "request_metadata_json": {
            "second_step_51_llm_answer_composer_review_batch": True,
            "case_id": case.case_id,
            "expected_service_key": case.service_key,
            "force_llm_answer_composer_replacement": bool(force_replacement),
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
        "case": asdict(case) | {"source_row": case.source_row or {}},
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

    service_resolution = response_payload.get("service_resolution") if isinstance(response_payload.get("service_resolution"), dict) else {}
    message_guard = debug_payload_json.get("message_guard") if isinstance(debug_payload_json.get("message_guard"), dict) else {}

    result["service_resolution"] = service_resolution
    result["llm_answer_composer_call_policy"] = policy
    result["llm_answer_composer"] = composer
    result["message_guard"] = message_guard
    result["review"] = _build_review_fields(result)
    result["hard_findings"] = _hard_findings(result)
    result["soft_notes"] = _soft_notes(result)
    result["ok"] = not result["hard_findings"]
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


def _build_review_fields(result: dict[str, Any]) -> dict[str, Any]:
    composer = result.get("llm_answer_composer") if isinstance(result.get("llm_answer_composer"), dict) else {}
    original = str(composer.get("original_answer_text") or "")
    composed = str(composer.get("composed_answer_text") or "")
    final = str(composer.get("final_answer_text") or result.get("answer_text") or "")
    api_answer = str(result.get("answer_text") or "")
    original_bullets = _count_list_markers(original)
    final_bullets = _count_list_markers(final)
    return {
        "original_answer_text": original,
        "composed_answer_text": composed,
        "final_answer_text": final,
        "api_answer_text": api_answer,
        "has_change": bool(original and final and _normalize_text(original) != _normalize_text(final)),
        "original_length": len(original),
        "final_length": len(final),
        "length_delta": len(final) - len(original),
        "original_bullet_count": original_bullets,
        "final_bullet_count": final_bullets,
        "list_structure_lost": bool(original_bullets >= 5 and final_bullets < 2),
    }


def _hard_findings(result: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    if result.get("http_status") != 200 or not result.get("api_ok"):
        findings.append("api_not_ok")
        return findings

    composer = result.get("llm_answer_composer") if isinstance(result.get("llm_answer_composer"), dict) else {}
    policy = result.get("llm_answer_composer_call_policy") if isinstance(result.get("llm_answer_composer_call_policy"), dict) else {}
    message_guard = result.get("message_guard") if isinstance(result.get("message_guard"), dict) else {}

    if message_guard and message_guard.get("should_run_rag") is False:
        if composer:
            findings.append("guard_case_should_not_have_composer")
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
    warnings = [str(item) for item in (composer.get("validation_warnings") or [])]
    violations = [str(item) for item in (composer.get("grounding_violations") or [])]
    replacement_applied = composer.get("runtime_replacement_applied") is True
    composed = str(composer.get("composed_answer_text") or "")
    answer_text = str(result.get("answer_text") or "")

    if violations:
        findings.append("composer_grounding_violations=" + ",".join(violations))
    if replacement_applied and violations:
        findings.append("replacement_applied_with_grounding_violations")
    if replacement_applied and any(item.startswith("answer_question_mismatch:") for item in warnings):
        findings.append("replacement_applied_after_answer_question_mismatch")
    service_resolution = result.get("service_resolution") if isinstance(result.get("service_resolution"), dict) else {}
    resolution_status = str(service_resolution.get("resolution_status") or "")

    if replacement_applied and result.get("answer_mode") == "safe_no_answer":
        findings.append("safe_no_answer_replacement_applied")
    if replacement_applied and resolution_status == "service_discovery":
        findings.append("service_discovery_replacement_applied")
    if replacement_applied and _looks_like_broad_or_clarification_answer(
        str(composer.get("original_answer_text") or result.get("answer_text") or "")
    ):
        findings.append("broad_or_clarification_replacement_applied")
    review = result.get("review") if isinstance(result.get("review"), dict) else {}
    if replacement_applied and not composed.strip():
        findings.append("replacement_applied_without_composed_answer")
    if replacement_applied and review.get("list_structure_lost") is True:
        findings.append("replacement_applied_with_lost_list_structure")
    if replacement_applied and composed.strip() and _normalize_text(answer_text) != _normalize_text(composed):
        findings.append("api_answer_not_equal_composed_answer")
    if provider_status not in {"ok", "not_called"}:
        findings.append(f"unexpected_provider_status:{provider_status or 'empty'}")
    if status not in {"ok", "skipped"}:
        findings.append(f"unexpected_composer_status:{status or 'empty'}")
    return findings


def _soft_notes(result: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    composer = result.get("llm_answer_composer") if isinstance(result.get("llm_answer_composer"), dict) else {}
    if not composer:
        return notes
    status = str(composer.get("status") or "")
    provider_status = str(composer.get("provider_status") or "")
    replacement_reason = str(composer.get("runtime_replacement_reason") or "")
    warnings = [str(item) for item in (composer.get("validation_warnings") or [])]
    service_resolution = result.get("service_resolution") if isinstance(result.get("service_resolution"), dict) else {}
    resolution_status = str(service_resolution.get("resolution_status") or "")
    if resolution_status:
        notes.append("resolution_status:" + resolution_status)
    if status == "skipped":
        notes.append("composer_skipped")
    if provider_status == "not_called":
        notes.append("provider_not_called")
    if replacement_reason:
        notes.append("replacement_reason:" + replacement_reason)
    for item in warnings:
        notes.append("validation_warning:" + item)
    return notes


def _build_summary(
    results: list[dict[str, Any]],
    *,
    started_at: float,
    url: str,
    questions_file: str,
    category: str,
    limit: int,
    sample_per_service: int,
    force_replacement: bool,
) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    provider_statuses: dict[str, int] = {}
    replacements = {"applied": 0, "suppressed": 0, "not_present": 0}
    answer_modes: dict[str, int] = {}
    resolution_statuses: dict[str, int] = {}
    replacement_reasons: dict[str, int] = {}
    changed = 0
    list_structure_lost_count = 0
    for row in results:
        answer_mode = str(row.get("answer_mode") or "empty")
        answer_modes[answer_mode] = answer_modes.get(answer_mode, 0) + 1
        service_resolution = row.get("service_resolution") if isinstance(row.get("service_resolution"), dict) else {}
        resolution_status = str(service_resolution.get("resolution_status") or "empty")
        resolution_statuses[resolution_status] = resolution_statuses.get(resolution_status, 0) + 1
        composer = row.get("llm_answer_composer") if isinstance(row.get("llm_answer_composer"), dict) else {}
        statuses[str(composer.get("status") or "not_present")] = statuses.get(str(composer.get("status") or "not_present"), 0) + 1
        provider_statuses[str(composer.get("provider_status") or "not_present")] = provider_statuses.get(str(composer.get("provider_status") or "not_present"), 0) + 1
        if not composer:
            replacements["not_present"] += 1
        elif composer.get("runtime_replacement_applied") is True:
            replacements["applied"] += 1
        else:
            replacements["suppressed"] += 1
        replacement_reason = str(composer.get("runtime_replacement_reason") or "not_present")
        replacement_reasons[replacement_reason] = replacement_reasons.get(replacement_reason, 0) + 1
        review = row.get("review") if isinstance(row.get("review"), dict) else {}
        if review.get("has_change"):
            changed += 1
        if review.get("list_structure_lost"):
            list_structure_lost_count += 1

    hard = [row for row in results if row.get("hard_findings")]
    return {
        "ok": not hard,
        "version": VERSION,
        "cases_total": len(results),
        "cases_ok": len(results) - len(hard),
        "cases_with_hard_findings": len(hard),
        "composer_statuses": statuses,
        "provider_statuses": provider_statuses,
        "runtime_replacements": replacements,
        "answer_modes": answer_modes,
        "resolution_statuses": resolution_statuses,
        "replacement_reasons": replacement_reasons,
        "changed_answers": changed,
        "list_structure_lost_count": list_structure_lost_count,
        "elapsed_seconds": round(time.perf_counter() - started_at, 4),
        "url": url,
        "questions_file": questions_file,
        "category": category,
        "limit": limit,
        "sample_per_service": sample_per_service,
        "force_replacement": force_replacement,
    }


def _write_outputs(out_dir: Path, results: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    _write_summary_tsv(out_dir / "summary.tsv", results)
    _write_review_tsv(out_dir / "manual_review.tsv", results)
    _write_findings_md(out_dir / "findings.md", results, summary)
    _write_manual_review_md(out_dir / "manual_review.md", results, summary)


def _write_summary_tsv(path: Path, results: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "case_id",
            "service_key",
            "service_name_short",
            "api_ok",
            "answer_mode",
            "citations_count",
            "resolved_service_key",
            "resolved_service_name",
            "resolution_status",
            "composer_provider_status",
            "composer_status",
            "runtime_replacement_applied",
            "runtime_replacement_reason",
            "validation_warnings",
            "violations",
            "has_change",
            "length_delta",
            "original_bullet_count",
            "final_bullet_count",
            "list_structure_lost",
            "hard_findings",
            "soft_notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in results:
            case = row.get("case") or {}
            composer = row.get("llm_answer_composer") if isinstance(row.get("llm_answer_composer"), dict) else {}
            service_resolution = row.get("service_resolution") if isinstance(row.get("service_resolution"), dict) else {}
            review = row.get("review") if isinstance(row.get("review"), dict) else {}
            writer.writerow(
                {
                    "case_id": case.get("case_id"),
                    "service_key": case.get("service_key"),
                    "service_name_short": case.get("service_name_short"),
                    "api_ok": row.get("api_ok"),
                    "answer_mode": row.get("answer_mode"),
                    "citations_count": row.get("citations_count"),
                    "resolved_service_key": service_resolution.get("service_key"),
                    "resolved_service_name": service_resolution.get("service_name"),
                    "resolution_status": service_resolution.get("resolution_status"),
                    "composer_provider_status": composer.get("provider_status"),
                    "composer_status": composer.get("status"),
                    "runtime_replacement_applied": composer.get("runtime_replacement_applied"),
                    "runtime_replacement_reason": composer.get("runtime_replacement_reason"),
                    "validation_warnings": ";".join(str(item) for item in (composer.get("validation_warnings") or [])),
                    "violations": ";".join(str(item) for item in (composer.get("grounding_violations") or [])),
                    "has_change": review.get("has_change"),
                    "length_delta": review.get("length_delta"),
                    "original_bullet_count": review.get("original_bullet_count"),
                    "final_bullet_count": review.get("final_bullet_count"),
                    "list_structure_lost": review.get("list_structure_lost"),
                    "hard_findings": ";".join(str(item) for item in (row.get("hard_findings") or [])),
                    "soft_notes": ";".join(str(item) for item in (row.get("soft_notes") or [])),
                }
            )


def _write_review_tsv(path: Path, results: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "case_id",
            "question",
            "answer_mode",
            "runtime_replacement_applied",
            "runtime_replacement_reason",
            "validation_warnings",
            "hard_findings",
            "original_answer_text",
            "composed_answer_text",
            "final_answer_text",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in results:
            case = row.get("case") or {}
            composer = row.get("llm_answer_composer") if isinstance(row.get("llm_answer_composer"), dict) else {}
            review = row.get("review") if isinstance(row.get("review"), dict) else {}
            writer.writerow(
                {
                    "case_id": case.get("case_id"),
                    "question": case.get("question"),
                    "answer_mode": row.get("answer_mode"),
                    "runtime_replacement_applied": composer.get("runtime_replacement_applied"),
                    "runtime_replacement_reason": composer.get("runtime_replacement_reason"),
                    "validation_warnings": ";".join(str(item) for item in (composer.get("validation_warnings") or [])),
                    "hard_findings": ";".join(str(item) for item in (row.get("hard_findings") or [])),
                    "original_answer_text": review.get("original_answer_text"),
                    "composed_answer_text": review.get("composed_answer_text"),
                    "final_answer_text": review.get("final_answer_text"),
                }
            )


def _write_findings_md(path: Path, results: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# LLM answer composer review batch",
        "",
        f"version: `{summary['version']}`",
        f"ok: `{summary['ok']}`",
        f"cases_total: `{summary['cases_total']}`",
        f"cases_with_hard_findings: `{summary['cases_with_hard_findings']}`",
        f"runtime_replacements: `{json.dumps(summary['runtime_replacements'], ensure_ascii=False)}`",
        f"replacement_reasons: `{json.dumps(summary.get('replacement_reasons', {}), ensure_ascii=False)}`",
        f"resolution_statuses: `{json.dumps(summary.get('resolution_statuses', {}), ensure_ascii=False)}`",
        f"changed_answers: `{summary['changed_answers']}`",
        f"list_structure_lost_count: `{summary.get('list_structure_lost_count')}`",
        "",
    ]
    hard = [row for row in results if row.get("hard_findings")]
    if hard:
        lines.append("## Hard findings")
        for row in hard:
            case = row.get("case") or {}
            lines.append(f"- `{case.get('case_id')}`: {', '.join(row.get('hard_findings') or [])}")
    else:
        lines.append("Блокирующих флагов нет.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manual_review_md(path: Path, results: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# Manual review: LLM answer composer",
        "",
        "Этот файл предназначен для ручной оценки качества переписанного ответа.",
        "В пользовательский канал эти ответы включать нельзя без отдельного решения.",
        "",
        f"Всего случаев: {summary['cases_total']}",
        f"Замен применено в test-режиме: {summary['runtime_replacements'].get('applied')}",
        "",
    ]
    shown = 0
    for row in results:
        composer = row.get("llm_answer_composer") if isinstance(row.get("llm_answer_composer"), dict) else {}
        if composer.get("runtime_replacement_applied") is not True:
            continue
        case = row.get("case") or {}
        review = row.get("review") if isinstance(row.get("review"), dict) else {}
        shown += 1
        lines.append(f"## {case.get('case_id')} — {case.get('service_name_short') or case.get('category')}")
        lines.append("")
        lines.append(f"**Вопрос:** {case.get('question')}")
        lines.append("")
        lines.append("**Исходный ответ:**")
        lines.append("")
        lines.append(_quote_block(_clip(str(review.get("original_answer_text") or ""), 1600)))
        lines.append("")
        lines.append("**LLM-версия:**")
        lines.append("")
        lines.append(_quote_block(_clip(str(review.get("final_answer_text") or ""), 1600)))
        lines.append("")
    if shown == 0:
        lines.append("Нет применённых test-замен для ручной оценки.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _clip(text: str, limit: int) -> str:
    clean = text.strip()
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + "\n...[обрезано]"


def _quote_block(text: str) -> str:
    if not text.strip():
        return "> [пусто]"
    return "\n".join("> " + line for line in text.splitlines())


def _looks_like_broad_or_clarification_answer(answer_text: str) -> bool:
    normalized = _normalize_text(answer_text).lower()
    markers = (
        "нашёл несколько мер",
        "найдено несколько мер",
        "несколько мер социальной поддержки",
        "могут быть релевантны",
        "это не означает, что право",
        "нельзя надёжно выбрать одну",
        "нужно уточнить",
        "что нужно уточнить",
        "после уточнения можно проверять",
    )
    return any(marker in normalized for marker in markers)



def _count_list_markers(text: str) -> int:
    value = str(text or "")
    count = value.count("•")
    count += len(__import__("re").findall(r"(?:^|\n)\s*[-–]\s+\S", value))
    count += len(__import__("re").findall(r"(?:^|\n)\s*\d{1,2}[.)]\s+\S", value))
    return count


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").replace("\xa0", " ").split())


def _safe_file_name(value: str) -> str:
    result = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_"}:
            result.append(ch)
        else:
            result.append("_")
    return "".join(result).strip("_")[:120] or "case"


if __name__ == "__main__":
    main()
