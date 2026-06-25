# ============================================================
# File: scripts/compare_message_understanding_models.py
# Purpose:
#   Compare two or more LLM models for optional message understanding layer.
#
# Version:
#   second_step_06_compare_message_understanding_models_v1
#
# Notes:
#   - Does not change API behavior.
#   - Calls only MessageUnderstandingService in shadow-like mode.
#   - Writes summary reports for manual decision: e.g. gpt-4.1-mini vs gpt-4.1-nano.
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
from typing import Any, Iterable

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

VERSION = "second_step_06_compare_message_understanding_models_v1"
DEFAULT_QUESTIONS_FILE = "scripts/temp/message_understanding_compare_questions.tsv"
DEFAULT_MODELS = ["gpt-4.1-mini", "gpt-4.1-nano"]


@dataclass(slots=True, frozen=True)
class CompareCase:
    case_id: str
    category: str
    question: str
    expected_intent: str
    expected_needs_service_discovery: str
    expected_territory: str
    must_have_any_fact: list[str]
    comment: str


async def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = _load_cases(Path(args.questions_file), limit=args.limit)
    models = _parse_models(args.models)
    if not models:
        raise SystemExit("Не задан список моделей для сравнения")

    settings = load_settings()
    client = OpenAIClientFactory(settings.openai).create_async_client()
    classifier = RuleBasedIntentClassifier()

    services = {
        model: LLMMessageUnderstandingService(
            client,
            config=MessageUnderstandingConfig(
                enabled=True,
                mode="shadow",
                model_name=model,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                min_confidence_to_apply=args.min_confidence,
                request_timeout_seconds=args.timeout_seconds,
            ),
        )
        for model in models
    }

    started_at = time.perf_counter()
    results: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        if not args.quiet:
            print(f"[{index}/{len(cases)}] {case.case_id}: {case.question}", flush=True)

        deterministic = await classifier.classify(case.question)

        model_payloads: dict[str, dict[str, Any]] = {}
        for model, service in services.items():
            one_started = time.perf_counter()
            understanding = await service.understand(
                case.question,
                deterministic_classification=deterministic,
                channel_code=args.channel,
            )
            elapsed = round(time.perf_counter() - one_started, 4)
            payload = understanding.to_payload()
            payload["elapsed_seconds"] = elapsed
            payload["score"] = _score_payload(case, payload)
            model_payloads[model] = payload

        results.append(
            {
                "case": _case_to_payload(case),
                "deterministic_classification": _jsonable(deterministic),
                "models": model_payloads,
                "pairwise": _pairwise(models, model_payloads),
            }
        )

    summary = _build_summary(
        cases=cases,
        models=models,
        results=results,
        elapsed_seconds=round(time.perf_counter() - started_at, 4),
        args=args,
    )

    _write_json(out_dir / "results.json", {"summary": summary, "results": results})
    _write_jsonl(out_dir / "results.jsonl", results)
    _write_tsv(out_dir / "results.tsv", cases, models, results)
    _write_disagreements_tsv(out_dir / "disagreements.tsv", models, results)
    _write_errors_tsv(out_dir / "errors.tsv", models, results)
    _write_summary_md(out_dir / "summary.md", summary, models, results)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


# ============================================================
# Loading
# ============================================================


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions-file", default=DEFAULT_QUESTIONS_FILE)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--channel", default="model_compare")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=700)
    parser.add_argument("--min-confidence", type=float, default=0.72)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _parse_models(raw: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in str(raw or "").split(","):
        model = item.strip()
        if not model or model in seen:
            continue
        seen.add(model)
        result.append(model)
    return result


def _load_cases(path: Path, *, limit: int = 0) -> list[CompareCase]:
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл вопросов: {path}")

    cases: list[CompareCase] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row_number, row in enumerate(reader, start=2):
            case = CompareCase(
                case_id=(row.get("case_id") or f"row_{row_number}").strip(),
                category=(row.get("category") or "").strip(),
                question=(row.get("question") or "").strip(),
                expected_intent=(row.get("expected_intent") or "").strip().lower(),
                expected_needs_service_discovery=(row.get("expected_needs_service_discovery") or "").strip().lower(),
                expected_territory=(row.get("expected_territory") or "").strip(),
                must_have_any_fact=_split_terms(row.get("must_have_any_fact") or ""),
                comment=(row.get("comment") or "").strip(),
            )
            if not case.question:
                continue
            cases.append(case)
            if limit > 0 and len(cases) >= limit:
                break
    return cases


# ============================================================
# Scoring
# ============================================================


def _score_payload(case: CompareCase, payload: dict[str, Any]) -> dict[str, Any]:
    provider_status = str(payload.get("provider_status") or "")
    intent = str(payload.get("intent") or "").lower()
    facts = _lower_joined(payload.get("applicant_facts") or [])
    topic = _lower_joined([payload.get("topic"), payload.get("service_hint")])
    territory = str(payload.get("territory") or "").lower()

    expected_intent_ok: bool | None = None
    if case.expected_intent:
        expected_intent_ok = intent == case.expected_intent

    expected_discovery_ok: bool | None = None
    if case.expected_needs_service_discovery in {"true", "false"}:
        expected_value = case.expected_needs_service_discovery == "true"
        expected_discovery_ok = bool(payload.get("needs_service_discovery")) == expected_value

    expected_territory_ok: bool | None = None
    if case.expected_territory:
        expected_territory_ok = case.expected_territory.lower() in territory or case.expected_territory.lower() in topic or case.expected_territory.lower() in facts

    must_have_fact_ok: bool | None = None
    if case.must_have_any_fact:
        must_have_fact_ok = any(term.lower() in facts or term.lower() in topic or term.lower() in territory for term in case.must_have_any_fact)

    checks = [
        value
        for value in [expected_intent_ok, expected_discovery_ok, expected_territory_ok, must_have_fact_ok]
        if value is not None
    ]

    return {
        "provider_ok": provider_status == "ok",
        "expected_intent_ok": expected_intent_ok,
        "expected_discovery_ok": expected_discovery_ok,
        "expected_territory_ok": expected_territory_ok,
        "must_have_fact_ok": must_have_fact_ok,
        "checks_total": len(checks),
        "checks_ok": sum(1 for value in checks if value is True),
        "checks_failed": sum(1 for value in checks if value is False),
    }


def _pairwise(models: list[str], payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if len(models) < 2:
        return result
    base = models[0]
    for other in models[1:]:
        a = payloads.get(base) or {}
        b = payloads.get(other) or {}
        key = f"{base}__vs__{other}"
        result[key] = {
            "same_provider_status": a.get("provider_status") == b.get("provider_status"),
            "same_intent": a.get("intent") == b.get("intent"),
            "same_supported_domain": a.get("is_supported_domain") == b.get("is_supported_domain"),
            "same_service_discovery": a.get("needs_service_discovery") == b.get("needs_service_discovery"),
            "confidence_delta": round(float(a.get("confidence") or 0) - float(b.get("confidence") or 0), 3),
            "elapsed_delta_seconds": round(float(a.get("elapsed_seconds") or 0) - float(b.get("elapsed_seconds") or 0), 4),
        }
    return result


def _build_summary(
    *,
    cases: list[CompareCase],
    models: list[str],
    results: list[dict[str, Any]],
    elapsed_seconds: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    by_model: dict[str, Any] = {}
    for model in models:
        payloads = [(row.get("models") or {}).get(model) or {} for row in results]
        scores = [payload.get("score") or {} for payload in payloads]
        checks_total = sum(int(score.get("checks_total") or 0) for score in scores)
        checks_ok = sum(int(score.get("checks_ok") or 0) for score in scores)
        by_model[model] = {
            "provider_ok": sum(1 for payload in payloads if payload.get("provider_status") == "ok"),
            "provider_error": sum(1 for payload in payloads if payload.get("provider_status") != "ok"),
            "avg_confidence": _avg([float(payload.get("confidence") or 0) for payload in payloads]),
            "avg_elapsed_seconds": _avg([float(payload.get("elapsed_seconds") or 0) for payload in payloads]),
            "checks_total": checks_total,
            "checks_ok": checks_ok,
            "checks_failed": checks_total - checks_ok,
            "checks_ok_rate": round(checks_ok / checks_total, 4) if checks_total else None,
            "intent_matches_expected": sum(1 for score in scores if score.get("expected_intent_ok") is True),
            "intent_mismatches_expected": sum(1 for score in scores if score.get("expected_intent_ok") is False),
        }

    pairwise_summary: dict[str, Any] = {}
    if len(models) >= 2:
        base = models[0]
        for other in models[1:]:
            key = f"{base}__vs__{other}"
            pair_rows = [(row.get("pairwise") or {}).get(key) or {} for row in results]
            pairwise_summary[key] = {
                "same_intent": sum(1 for row in pair_rows if row.get("same_intent") is True),
                "different_intent": sum(1 for row in pair_rows if row.get("same_intent") is False),
                "same_service_discovery": sum(1 for row in pair_rows if row.get("same_service_discovery") is True),
                "different_service_discovery": sum(1 for row in pair_rows if row.get("same_service_discovery") is False),
                "avg_elapsed_delta_seconds": _avg([float(row.get("elapsed_delta_seconds") or 0) for row in pair_rows]),
            }

    return {
        "version": VERSION,
        "questions_file": args.questions_file,
        "models": models,
        "total_cases": len(cases),
        "elapsed_seconds_total": elapsed_seconds,
        "by_model": by_model,
        "pairwise": pairwise_summary,
    }


# ============================================================
# Writers
# ============================================================


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _write_tsv(path: Path, cases: list[CompareCase], models: list[str], results: list[dict[str, Any]]) -> None:
    headers = [
        "case_id",
        "category",
        "question",
        "expected_intent",
        "expected_needs_service_discovery",
        "expected_territory",
        "must_have_any_fact",
        "comment",
    ]
    for model in models:
        prefix = _safe_col(model)
        headers.extend(
            [
                f"{prefix}_provider_status",
                f"{prefix}_intent",
                f"{prefix}_confidence",
                f"{prefix}_service_hint",
                f"{prefix}_topic",
                f"{prefix}_applicant_facts",
                f"{prefix}_territory",
                f"{prefix}_needs_service_discovery",
                f"{prefix}_needs_clarification",
                f"{prefix}_elapsed_seconds",
                f"{prefix}_checks_ok",
                f"{prefix}_checks_failed",
                f"{prefix}_error",
            ]
        )

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, delimiter="\t", fieldnames=headers)
        writer.writeheader()
        for row in results:
            case = row["case"]
            out: dict[str, Any] = {
                "case_id": case["case_id"],
                "category": case["category"],
                "question": case["question"],
                "expected_intent": case["expected_intent"],
                "expected_needs_service_discovery": case["expected_needs_service_discovery"],
                "expected_territory": case["expected_territory"],
                "must_have_any_fact": " | ".join(case["must_have_any_fact"]),
                "comment": case["comment"],
            }
            for model in models:
                payload = (row.get("models") or {}).get(model) or {}
                score = payload.get("score") or {}
                prefix = _safe_col(model)
                out.update(
                    {
                        f"{prefix}_provider_status": payload.get("provider_status"),
                        f"{prefix}_intent": payload.get("intent"),
                        f"{prefix}_confidence": payload.get("confidence"),
                        f"{prefix}_service_hint": payload.get("service_hint"),
                        f"{prefix}_topic": payload.get("topic"),
                        f"{prefix}_applicant_facts": " | ".join(payload.get("applicant_facts") or []),
                        f"{prefix}_territory": payload.get("territory"),
                        f"{prefix}_needs_service_discovery": payload.get("needs_service_discovery"),
                        f"{prefix}_needs_clarification": payload.get("needs_clarification"),
                        f"{prefix}_elapsed_seconds": payload.get("elapsed_seconds"),
                        f"{prefix}_checks_ok": score.get("checks_ok"),
                        f"{prefix}_checks_failed": score.get("checks_failed"),
                        f"{prefix}_error": payload.get("error"),
                    }
                )
            writer.writerow(out)


def _write_disagreements_tsv(path: Path, models: list[str], results: list[dict[str, Any]]) -> None:
    headers = [
        "case_id",
        "question",
        "field",
    ]
    for model in models:
        prefix = _safe_col(model)
        headers.extend([f"{prefix}_intent", f"{prefix}_service_hint", f"{prefix}_facts", f"{prefix}_needs_discovery", f"{prefix}_confidence"])

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, delimiter="\t", fieldnames=headers)
        writer.writeheader()
        for row in results:
            payloads = row.get("models") or {}
            case = row.get("case") or {}
            intents = {str((payloads.get(model) or {}).get("intent")) for model in models}
            discoveries = {str((payloads.get(model) or {}).get("needs_service_discovery")) for model in models}
            statuses = {str((payloads.get(model) or {}).get("provider_status")) for model in models}
            fields: list[str] = []
            if len(statuses) > 1:
                fields.append("provider_status")
            if len(intents) > 1:
                fields.append("intent")
            if len(discoveries) > 1:
                fields.append("needs_service_discovery")
            if not fields:
                continue
            out = {"case_id": case.get("case_id"), "question": case.get("question"), "field": ",".join(fields)}
            for model in models:
                payload = payloads.get(model) or {}
                prefix = _safe_col(model)
                out.update(
                    {
                        f"{prefix}_intent": payload.get("intent"),
                        f"{prefix}_service_hint": payload.get("service_hint"),
                        f"{prefix}_facts": " | ".join(payload.get("applicant_facts") or []),
                        f"{prefix}_needs_discovery": payload.get("needs_service_discovery"),
                        f"{prefix}_confidence": payload.get("confidence"),
                    }
                )
            writer.writerow(out)


def _write_errors_tsv(path: Path, models: list[str], results: list[dict[str, Any]]) -> None:
    headers = ["case_id", "question", "model", "provider_status", "error"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, delimiter="\t", fieldnames=headers)
        writer.writeheader()
        for row in results:
            case = row.get("case") or {}
            for model in models:
                payload = (row.get("models") or {}).get(model) or {}
                if payload.get("provider_status") == "ok":
                    continue
                writer.writerow(
                    {
                        "case_id": case.get("case_id"),
                        "question": case.get("question"),
                        "model": model,
                        "provider_status": payload.get("provider_status"),
                        "error": payload.get("error"),
                    }
                )


def _write_summary_md(path: Path, summary: dict[str, Any], models: list[str], results: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("# Сравнение моделей для ИИ-диспетчеризации")
    lines.append("")
    lines.append(f"Версия скрипта: `{summary['version']}`")
    lines.append(f"Всего вопросов: **{summary['total_cases']}**")
    lines.append(f"Модели: {', '.join(f'`{m}`' for m in models)}")
    lines.append(f"Общее время: **{summary['elapsed_seconds_total']} сек.**")
    lines.append("")
    lines.append("## Сводка по моделям")
    lines.append("")
    lines.append("| Модель | OK | Ошибки | Средняя уверенность | Среднее время, сек. | Проверок OK | Проверок с ошибкой | Доля OK |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for model in models:
        row = summary["by_model"][model]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{model}`",
                    str(row["provider_ok"]),
                    str(row["provider_error"]),
                    str(row["avg_confidence"]),
                    str(row["avg_elapsed_seconds"]),
                    str(row["checks_ok"]),
                    str(row["checks_failed"]),
                    str(row["checks_ok_rate"]),
                ]
            )
            + " |"
        )
    lines.append("")

    if summary.get("pairwise"):
        lines.append("## Расхождения")
        lines.append("")
        lines.append("| Пара | Разный intent | Разный service_discovery | Средняя разница времени, сек. |")
        lines.append("|---|---:|---:|---:|")
        for key, row in summary["pairwise"].items():
            lines.append(
                f"| `{key}` | {row['different_intent']} | {row['different_service_discovery']} | {row['avg_elapsed_delta_seconds']} |"
            )
        lines.append("")

    worst_rows = _worst_rows(models, results, limit=15)
    if worst_rows:
        lines.append("## Вопросы, которые нужно посмотреть вручную")
        lines.append("")
        lines.append("| ID | Вопрос | Причина |")
        lines.append("|---|---|---|")
        for case_id, question, reason in worst_rows:
            lines.append(f"| {case_id} | {question} | {reason} |")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# Helpers
# ============================================================


def _worst_rows(models: list[str], results: list[dict[str, Any]], *, limit: int) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for row in results:
        case = row.get("case") or {}
        reasons: list[str] = []
        payloads = row.get("models") or {}
        for model in models:
            payload = payloads.get(model) or {}
            score = payload.get("score") or {}
            if payload.get("provider_status") != "ok":
                reasons.append(f"{model}: provider_status={payload.get('provider_status')}")
            elif score.get("checks_failed", 0):
                reasons.append(f"{model}: failed_checks={score.get('checks_failed')}")
        pairwise = row.get("pairwise") or {}
        for key, pair in pairwise.items():
            if pair.get("same_intent") is False:
                reasons.append(f"{key}: different_intent")
            if pair.get("same_service_discovery") is False:
                reasons.append(f"{key}: different_service_discovery")
        if reasons:
            rows.append((str(case.get("case_id")), str(case.get("question")), "; ".join(reasons)))
        if len(rows) >= limit:
            break
    return rows


def _case_to_payload(case: CompareCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "category": case.category,
        "question": case.question,
        "expected_intent": case.expected_intent,
        "expected_needs_service_discovery": case.expected_needs_service_discovery,
        "expected_territory": case.expected_territory,
        "must_have_any_fact": list(case.must_have_any_fact),
        "comment": case.comment,
    }


def _split_terms(raw: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for item in str(raw or "").split("|"):
        text = item.strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        terms.append(text)
    return terms


def _lower_joined(values: Any) -> str:
    if isinstance(values, list):
        return " ".join(str(v or "") for v in values).lower()
    return str(values or "").lower()


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _safe_col(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


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
