# ============================================================
# File: scripts/run_message_understanding_regression_cases.py
# Purpose:
#   Regression runner for optional LLM message understanding layer.
#
# Version:
#   second_step_44_message_understanding_regression_v3
#
# Notes:
#   - Calls only MessageUnderstandingService directly.
#   - Does not call retrieval and does not change API behavior.
#   - Intended for shadow/diagnostic evaluation before enabling assist mode.
# ============================================================

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config.settings import load_settings
from app.integrations.openai.client_factory import OpenAIClientFactory
from app.services.answers.intent_classifier import RuleBasedIntentClassifier
from app.services.answers.message_understanding import (
    LLMMessageUnderstandingService,
    MessageUnderstandingConfig,
)

VERSION = "second_step_44_message_understanding_regression_v3"
DEFAULT_CASES_FILE = "scripts/temp/message_understanding_regression_cases.tsv"


@dataclass(slots=True, frozen=True)
class RegressionCase:
    case_id: str
    category: str
    question: str
    expected_supported_domain: str
    expected_intent: str
    expected_needs_service_discovery: str
    expected_territory: str
    must_have_any_fact: list[str]
    must_have_any_need: list[str]
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
    classifier = RuleBasedIntentClassifier()

    service = LLMMessageUnderstandingService(
        client,
        config=MessageUnderstandingConfig(
            enabled=True,
            mode=args.mode,
            model_name=args.model or settings.message_understanding.model_name,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            min_confidence_to_apply=args.min_confidence,
            request_timeout_seconds=args.timeout_seconds,
        ),
    )

    started_at = time.perf_counter()
    results: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        if not args.quiet:
            print(f"[{index}/{len(cases)}] {case.case_id}: {case.question}", flush=True)

        deterministic = await classifier.classify(case.question)
        one_started = time.perf_counter()
        understanding = await service.understand(
            case.question,
            deterministic_classification=deterministic,
            channel_code=args.channel,
        )
        elapsed = round(time.perf_counter() - one_started, 4)
        payload = understanding.to_payload()
        findings = _findings(case, payload)
        results.append(
            {
                "case": _case_to_payload(case),
                "deterministic_classification": _jsonable(deterministic),
                "message_understanding": payload,
                "elapsed_seconds": elapsed,
                "ok": not findings,
                "findings": findings,
            }
        )

    summary = _build_summary(
        cases=cases,
        results=results,
        elapsed_seconds=round(time.perf_counter() - started_at, 4),
        args=args,
    )

    _write_json(out_dir / "summary.json", summary)
    _write_json(out_dir / "results.json", {"summary": summary, "results": results})
    _write_jsonl(out_dir / "results.jsonl", results)
    _write_summary_tsv(out_dir / "summary.tsv", results)
    _write_findings_tsv(out_dir / "findings.tsv", results)
    _write_findings_md(out_dir / "findings.md", summary, results)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    if args.fail_on_findings and summary["cases_with_findings"]:
        raise SystemExit(2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-file", default=DEFAULT_CASES_FILE)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--channel", default="test")
    parser.add_argument("--model", default="")
    parser.add_argument("--mode", default="shadow")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=700)
    parser.add_argument("--min-confidence", type=float, default=0.72)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--fail-on-findings", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _load_cases(path: Path, *, limit: int) -> list[RegressionCase]:
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл сценариев: {path}")
    cases: list[RegressionCase] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        required = {
            "case_id",
            "category",
            "question",
            "expected_supported_domain",
            "expected_intent",
            "expected_needs_service_discovery",
            "expected_territory",
            "must_have_any_fact",
            "must_have_any_need",
            "comment",
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"В файле сценариев не хватает колонок: {', '.join(missing)}")
        for row_number, row in enumerate(reader, start=2):
            question = (row.get("question") or "").strip()
            if not question:
                continue
            cases.append(
                RegressionCase(
                    case_id=(row.get("case_id") or f"row_{row_number}").strip(),
                    category=(row.get("category") or "").strip(),
                    question=question,
                    expected_supported_domain=(row.get("expected_supported_domain") or "").strip().lower(),
                    expected_intent=(row.get("expected_intent") or "").strip().lower(),
                    expected_needs_service_discovery=(row.get("expected_needs_service_discovery") or "").strip().lower(),
                    expected_territory=(row.get("expected_territory") or "").strip(),
                    must_have_any_fact=_split_terms(row.get("must_have_any_fact") or ""),
                    must_have_any_need=_split_terms(row.get("must_have_any_need") or ""),
                    comment=(row.get("comment") or "").strip(),
                )
            )
            if limit and len(cases) >= limit:
                break
    return cases


def _split_terms(raw: str) -> list[str]:
    result: list[str] = []
    for item in str(raw or "").split("|"):
        value = " ".join(item.split()).strip()
        if value:
            result.append(value)
    return result


def _findings(case: RegressionCase, payload: dict[str, Any]) -> list[str]:
    findings: list[str] = []

    provider_status = str(payload.get("provider_status") or "")
    if provider_status != "ok":
        findings.append(f"provider_status={provider_status or 'empty'}")
        return findings

    if case.expected_supported_domain in {"true", "false"}:
        expected = case.expected_supported_domain == "true"
        actual = bool(payload.get("is_supported_domain"))
        if actual != expected:
            findings.append(f"supported_domain_expected_{expected}_got_{actual}")

    expected_intent = case.expected_intent.strip().lower()
    if expected_intent:
        actual_intent = str(payload.get("intent") or "").strip().lower()
        if actual_intent != expected_intent:
            findings.append(f"intent_expected_{expected_intent}_got_{actual_intent or 'empty'}")

    if case.expected_needs_service_discovery in {"true", "false"}:
        expected = case.expected_needs_service_discovery == "true"
        actual = bool(payload.get("needs_service_discovery"))
        if actual != expected:
            findings.append(f"service_discovery_expected_{expected}_got_{actual}")

    expected_territory = case.expected_territory.strip().lower()
    if expected_territory:
        actual_territory = str(payload.get("territory") or "").strip().lower()
        if expected_territory not in actual_territory:
            findings.append(f"territory_missing_{case.expected_territory}")

    if case.must_have_any_fact:
        haystack = _joined_lower(payload.get("applicant_facts"), payload.get("topic"))
        if not _contains_any(haystack, case.must_have_any_fact):
            findings.append("missing_expected_fact")

    if case.must_have_any_need:
        haystack = _joined_lower(payload.get("user_needs"), payload.get("topic"), payload.get("service_hint"))
        if not _contains_any(haystack, case.must_have_any_need):
            findings.append("missing_expected_need")

    if payload.get("needs_clarification") and not str(payload.get("clarification_question") or "").strip():
        findings.append("clarification_flag_without_question")

    safety_flags = payload.get("safety_flags") or []
    if safety_flags and case.expected_supported_domain == "true":
        allowed = {"privacy"}
        unexpected = [str(item) for item in safety_flags if str(item) not in allowed]
        if unexpected:
            findings.append("unexpected_safety_flags=" + ",".join(unexpected))

    return findings


def _contains_any(haystack: str, terms: list[str]) -> bool:
    compact_haystack = _normalize_for_match(haystack)
    for term in terms:
        compact_term = _normalize_for_match(term)
        if not compact_term:
            continue
        if compact_term in compact_haystack:
            return True
        if _token_overlap_match(compact_haystack, compact_term):
            return True
    return False


def _token_overlap_match(haystack: str, term: str) -> bool:
    """Loose semantic matcher for diagnostics only.

    It prevents false findings when the LLM returns "ветеран труда
    Красноярского края" and the scenario says "ветеран труда края".
    This does not affect runtime behavior.
    """
    hay_tokens = [tok for tok in haystack.split() if len(tok) >= 3]
    term_tokens = [tok for tok in term.split() if len(tok) >= 3]
    if not hay_tokens or not term_tokens:
        return False
    matched = 0
    for term_token in term_tokens:
        if any(_same_stemish_token(term_token, hay_token) for hay_token in hay_tokens):
            matched += 1
    return matched >= max(1, min(len(term_tokens), 2))


def _same_stemish_token(left: str, right: str) -> bool:
    if left == right:
        return True
    if len(left) >= 5 and len(right) >= 5 and (left.startswith(right[:5]) or right.startswith(left[:5])):
        return True
    if len(left) >= 4 and len(right) >= 4 and (left.startswith(right[:4]) or right.startswith(left[:4])):
        return True
    return False


def _normalize_for_match(value: str) -> str:
    text = str(value or "").replace("ё", "е").lower()
    text = text.replace("-", " ")
    return " ".join(text.split())


def _joined_lower(*values: Any) -> str:
    parts: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        else:
            parts.append(str(value))
    return _normalize_for_match(" ".join(parts))


def _case_to_payload(case: RegressionCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "category": case.category,
        "question": case.question,
        "expected_supported_domain": case.expected_supported_domain,
        "expected_intent": case.expected_intent,
        "expected_needs_service_discovery": case.expected_needs_service_discovery,
        "expected_territory": case.expected_territory,
        "must_have_any_fact": list(case.must_have_any_fact),
        "must_have_any_need": list(case.must_have_any_need),
        "comment": case.comment,
    }


def _build_summary(
    *,
    cases: list[RegressionCase],
    results: list[dict[str, Any]],
    elapsed_seconds: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    provider_statuses: dict[str, int] = {}
    categories: dict[str, dict[str, int]] = {}
    for row in results:
        payload = row.get("message_understanding") or {}
        status = str(payload.get("provider_status") or "empty")
        provider_statuses[status] = provider_statuses.get(status, 0) + 1
        category = str((row.get("case") or {}).get("category") or "")
        bucket = categories.setdefault(category, {"total": 0, "ok": 0, "findings": 0})
        bucket["total"] += 1
        if row.get("ok"):
            bucket["ok"] += 1
        else:
            bucket["findings"] += 1

    return {
        "ok": all(bool(row.get("ok")) for row in results),
        "version": VERSION,
        "cases_total": len(cases),
        "cases_ok": sum(1 for row in results if row.get("ok")),
        "cases_with_findings": sum(1 for row in results if not row.get("ok")),
        "provider_statuses": provider_statuses,
        "categories": categories,
        "elapsed_seconds": elapsed_seconds,
        "model": args.model or "settings.message_understanding.model_name",
        "mode": args.mode,
        "cases_file": args.cases_file,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _write_summary_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "case_id",
        "ok",
        "provider_status",
        "elapsed_seconds",
        "expected_intent",
        "actual_intent",
        "confidence",
        "expected_discovery",
        "actual_discovery",
        "expected_territory",
        "actual_territory",
        "topic",
        "service_hint",
        "applicant_facts",
        "user_needs",
        "findings",
        "question",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for row in rows:
            case = row["case"]
            payload = row["message_understanding"]
            writer.writerow(
                {
                    "case_id": case["case_id"],
                    "ok": row.get("ok"),
                    "provider_status": payload.get("provider_status"),
                    "elapsed_seconds": row.get("elapsed_seconds"),
                    "expected_intent": case.get("expected_intent"),
                    "actual_intent": payload.get("intent"),
                    "confidence": payload.get("confidence"),
                    "expected_discovery": case.get("expected_needs_service_discovery"),
                    "actual_discovery": payload.get("needs_service_discovery"),
                    "expected_territory": case.get("expected_territory"),
                    "actual_territory": payload.get("territory"),
                    "topic": payload.get("topic"),
                    "service_hint": payload.get("service_hint"),
                    "applicant_facts": " | ".join(str(item) for item in payload.get("applicant_facts") or []),
                    "user_needs": " | ".join(str(item) for item in payload.get("user_needs") or []),
                    "findings": ";".join(row.get("findings") or []),
                    "question": case.get("question"),
                }
            )


def _write_findings_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = ["case_id", "category", "finding", "question"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for row in rows:
            case = row["case"]
            for finding in row.get("findings") or []:
                writer.writerow(
                    {
                        "case_id": case.get("case_id"),
                        "category": case.get("category"),
                        "finding": finding,
                        "question": case.get("question"),
                    }
                )


def _write_findings_md(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Message understanding regression findings",
        "",
        f"version: `{VERSION}`",
        f"cases_total: {summary['cases_total']}",
        f"cases_ok: {summary['cases_ok']}",
        f"cases_with_findings: {summary['cases_with_findings']}",
        "",
    ]
    for row in rows:
        case = row["case"]
        payload = row["message_understanding"]
        findings = row.get("findings") or []
        lines.extend(
            [
                f"## {case['case_id']} — {case['category']}",
                f"- ok: {row.get('ok')}",
                f"- provider_status: {payload.get('provider_status')}",
                f"- expected_intent: {case.get('expected_intent')}",
                f"- actual_intent: {payload.get('intent')}",
                f"- confidence: {payload.get('confidence')}",
                f"- expected_discovery: {case.get('expected_needs_service_discovery')}",
                f"- actual_discovery: {payload.get('needs_service_discovery')}",
                f"- topic: {payload.get('topic') or ''}",
                f"- service_hint: {payload.get('service_hint') or ''}",
                f"- applicant_facts: {' | '.join(str(item) for item in payload.get('applicant_facts') or [])}",
                f"- user_needs: {' | '.join(str(item) for item in payload.get('user_needs') or [])}",
                f"- findings: {'; '.join(findings) if findings else 'нет автоматических флагов'}",
                "",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if hasattr(value, "value"):
        return value.value
    return value


if __name__ == "__main__":
    asyncio.run(main())
