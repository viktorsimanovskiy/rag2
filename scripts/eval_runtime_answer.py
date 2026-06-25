from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config.settings import load_settings
from app.db.models.enums import QuestionIntentEnum
from app.runtime.app_runtime import AppRuntime, AppRuntimeConfig
from app.services.answers.runtime_answer_service import RuntimeAnswerInput


# ============================================================
# Eval case model
# ============================================================

@dataclass(slots=True)
class EvalCase:
    """
    One evaluation case for runtime answer path.

    Why this exists:
    - gives us a stable, repeatable benchmark instead of checking JSON by eye
    - keeps evaluation independent from Telegram/n8n wiring
    - lets us compare retrieval/generation changes against the same questions
    """

    case_id: str
    intent: str
    question: str
    expected_substrings: list[str] = field(default_factory=list)
    forbidden_substrings: list[str] = field(default_factory=list)
    expected_answer_mode: str | None = None
    expected_evidence_quality: str | None = None
    notes: str | None = None


@dataclass(slots=True)
class EvalCaseResult:
    """
    Normalized evaluation output for one case.
    """

    case_id: str
    passed: bool
    question: str
    intent: str
    answer_mode: str | None
    confidence_score: float | None
    evidence_quality: str | None
    answer_text: str
    missing_expected_substrings: list[str] = field(default_factory=list)
    present_forbidden_substrings: list[str] = field(default_factory=list)
    failed_checks: list[str] = field(default_factory=list)


# ============================================================
# Parsing helpers
# ============================================================

def _parse_intent(value: str) -> QuestionIntentEnum:
    normalized = (value or "").strip().lower()

    mapping = {
        "documents": QuestionIntentEnum.DOCUMENTS_QUESTION,
        "documents_question": QuestionIntentEnum.DOCUMENTS_QUESTION,
        "docs": QuestionIntentEnum.DOCUMENTS_QUESTION,
        "deadline": QuestionIntentEnum.DEADLINE_QUESTION,
        "deadlines": QuestionIntentEnum.DEADLINE_QUESTION,
        "deadline_question": QuestionIntentEnum.DEADLINE_QUESTION,
        "procedure": QuestionIntentEnum.PROCEDURE_QUESTION,
        "procedure_question": QuestionIntentEnum.PROCEDURE_QUESTION,
        "refusal": QuestionIntentEnum.REJECTION_QUESTION,
        "refusal_reasons": QuestionIntentEnum.REJECTION_QUESTION,
        "refusal_reasons_question": QuestionIntentEnum.REJECTION_QUESTION,
    }

    if normalized not in mapping:
        supported = ", ".join(sorted(mapping.keys()))
        raise ValueError(
            f"Unsupported intent '{value}'. Supported values: {supported}"
        )

    return mapping[normalized]


def _safe_enum_value(value: Any) -> str | None:
    if value is None:
        return None
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, str) and enum_value.strip():
        return enum_value
    value_str = str(value).strip()
    return value_str or None


# ============================================================
# Runtime execution
# ============================================================

async def _run_one_case(
    runtime: AppRuntime,
    *,
    case: EvalCase,
) -> EvalCaseResult:
    """
    Run one case through the real runtime answer service.

    Important:
    this is not a unit test. It intentionally executes the same runtime path
    that we use in smoke testing:
        retrieval -> generation -> runtime result
    """

    intent = _parse_intent(case.intent)
    normalized_question = " ".join(case.question.strip().lower().split())

    async with runtime.session_scope() as session:
        service_factory = runtime.build_service_factory(session)
        service = service_factory.get_runtime_answer_service()

        result = await service.build_answer(
            RuntimeAnswerInput(
                session_id=uuid4(),
                question_event_id=uuid4(),
                channel_code="CLI_EVAL",
                question_text_raw=case.question,
                question_text_normalized=normalized_question,
                language_code="ru",
                intent_type=intent,
            )
        )

    generation_result = result.generation_result
    answer_text = (generation_result.answer_text or "").strip()
    answer_text_norm = answer_text.lower()

    answer_mode = _safe_enum_value(getattr(generation_result, "answer_mode", None))
    confidence_score = getattr(generation_result, "confidence_score", None)

    answer_payload_json = getattr(generation_result, "answer_payload_json", {}) or {}
    evidence_quality_raw = answer_payload_json.get("evidence_quality")
    evidence_quality = None
    if isinstance(evidence_quality_raw, str) and evidence_quality_raw.strip():
        evidence_quality = evidence_quality_raw.strip().lower()

    missing_expected_substrings: list[str] = []
    present_forbidden_substrings: list[str] = []
    failed_checks: list[str] = []

    for needle in case.expected_substrings:
        if needle.strip().lower() not in answer_text_norm:
            missing_expected_substrings.append(needle)

    for needle in case.forbidden_substrings:
        if needle.strip().lower() in answer_text_norm:
            present_forbidden_substrings.append(needle)

    if missing_expected_substrings:
        failed_checks.append("missing_expected_substrings")

    if present_forbidden_substrings:
        failed_checks.append("present_forbidden_substrings")

    if case.expected_answer_mode:
        expected_mode = case.expected_answer_mode.strip().lower()
        actual_mode = (answer_mode or "").strip().lower()
        if expected_mode != actual_mode:
            failed_checks.append("unexpected_answer_mode")

    if case.expected_evidence_quality:
        expected_quality = case.expected_evidence_quality.strip().lower()
        actual_quality = (evidence_quality or "").strip().lower()
        if expected_quality != actual_quality:
            failed_checks.append("unexpected_evidence_quality")

    return EvalCaseResult(
        case_id=case.case_id,
        passed=not failed_checks,
        question=case.question,
        intent=case.intent,
        answer_mode=answer_mode,
        confidence_score=confidence_score,
        evidence_quality=evidence_quality,
        answer_text=answer_text,
        missing_expected_substrings=missing_expected_substrings,
        present_forbidden_substrings=present_forbidden_substrings,
        failed_checks=failed_checks,
    )


async def _run_eval(cases: list[EvalCase]) -> dict[str, Any]:
    settings = load_settings()
    runtime = AppRuntime(
        config=AppRuntimeConfig(
            database=settings.database,
        )
    )

    await runtime.startup()
    try:
        results: list[EvalCaseResult] = []
        for case in cases:
            results.append(await _run_one_case(runtime, case=case))
    finally:
        await runtime.shutdown()

    passed_count = sum(1 for item in results if item.passed)
    failed_count = len(results) - passed_count

    return {
        "summary": {
            "total": len(results),
            "passed": passed_count,
            "failed": failed_count,
            "pass_rate": round((passed_count / len(results)) * 100, 2) if results else 0.0,
        },
        "results": [asdict(item) for item in results],
    }


# ============================================================
# IO helpers
# ============================================================

def _load_cases(path: Path) -> list[EvalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Eval file must contain a JSON array")

    cases: list[EvalCase] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Case #{index} must be an object")
        cases.append(
            EvalCase(
                case_id=str(item.get("case_id") or f"case_{index:03d}"),
                intent=str(item.get("intent") or "").strip(),
                question=str(item.get("question") or "").strip(),
                expected_substrings=[str(v) for v in item.get("expected_substrings") or []],
                forbidden_substrings=[str(v) for v in item.get("forbidden_substrings") or []],
                expected_answer_mode=(
                    str(item.get("expected_answer_mode")).strip()
                    if item.get("expected_answer_mode") is not None
                    else None
                ),
                expected_evidence_quality=(
                    str(item.get("expected_evidence_quality")).strip()
                    if item.get("expected_evidence_quality") is not None
                    else None
                ),
                notes=(
                    str(item.get("notes")).strip()
                    if item.get("notes") is not None
                    else None
                ),
            )
        )

    for case in cases:
        if not case.intent:
            raise ValueError(f"Case '{case.case_id}' has empty intent")
        if not case.question:
            raise ValueError(f"Case '{case.case_id}' has empty question")

    return cases


def _print_human_report(report: dict[str, Any]) -> None:
    summary = report.get("summary", {}) or {}
    print("=" * 100)
    print("RUNTIME EVAL SUMMARY")
    print("- total:", summary.get("total"))
    print("- passed:", summary.get("passed"))
    print("- failed:", summary.get("failed"))
    print("- pass_rate:", summary.get("pass_rate"))
    print("=" * 100)

    for item in report.get("results", []) or []:
        print(f"CASE: {item.get('case_id')}")
        print(f"PASS: {item.get('passed')}")
        print(f"INTENT: {item.get('intent')}")
        print(f"QUESTION: {item.get('question')}")
        print(f"ANSWER MODE: {item.get('answer_mode')}")
        print(f"EVIDENCE QUALITY: {item.get('evidence_quality')}")
        print(f"CONFIDENCE: {item.get('confidence_score')}")

        failed_checks = item.get("failed_checks") or []
        if failed_checks:
            print("FAILED CHECKS:", ", ".join(failed_checks))

        missing = item.get("missing_expected_substrings") or []
        if missing:
            print("MISSING EXPECTED:", ", ".join(missing))

        forbidden = item.get("present_forbidden_substrings") or []
        if forbidden:
            print("PRESENT FORBIDDEN:", ", ".join(forbidden))

        print("ANSWER TEXT:")
        print(item.get("answer_text") or "")
        print("-" * 100)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run repeatable runtime evaluation over a JSON cases file"
    )
    parser.add_argument(
        "--cases",
        required=True,
        help="Path to JSON file with evaluation cases",
    )
    parser.add_argument(
        "--output",
        required=False,
        help="Optional path to save JSON report",
    )
    args = parser.parse_args()

    cases_path = Path(args.cases).resolve()
    cases = _load_cases(cases_path)

    report = asyncio.run(_run_eval(cases))
    _print_human_report(report)

    if args.output:
        output_path = Path(args.output).resolve()
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"JSON report saved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
