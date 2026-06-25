# ============================================================
# File: scripts/run_question_bank_api.py
# Purpose:
#   Массовый прогон банка вопросов через работающий HTTP API RAG2
#   с диагностикой маршрутизации, формы ответа и проблем качества.
#
# Typical usage:
#   python scripts/run_question_bank_api.py \
#     --questions-file scripts/temp/service_question_bank.tsv \
#     --url http://127.0.0.1:8000/api/v1/answer \
#     --out-dir /home/logs/question_bank/step47_sample \
#     --category 01_как_пишет_человек_в_чате \
#     --sample-per-service 2 \
#     --debug
#
# Analyze existing responses without repeating API calls:
#   python scripts/run_question_bank_api.py \
#     --analyze-only \
#     --responses-file /home/logs/question_bank/step46_plain_sample/responses.jsonl \
#     --out-dir /home/logs/question_bank/step47_reanalysis
#
# Notes:
#   - the script uses only the Python standard library;
#   - it appends raw responses to responses.jsonl during live run;
#   - it creates detailed diagnostic files after the run;
#   - for n8n/Telegram-compatible behaviour channel defaults to "telegram".
# ============================================================

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_QUESTIONS_FILE = "scripts/temp/service_question_bank.tsv"
DEFAULT_URL = "http://127.0.0.1:8000/api/v1/answer"
DEFAULT_OUT_DIR = "/home/logs/question_bank"

MAX_PREVIEW_LENGTH = 900
VERY_LONG_ANSWER_LIMIT = 6000
EXTREMELY_LONG_ANSWER_LIMIT = 10000


@dataclass(slots=True)
class QuestionCase:
    question_id: str
    service_index: int
    service_key: str
    service_name_short: str
    service_name_full: str
    category: str
    question_text: str
    expected_service_hint: str
    aliases_hint: str
    question_profile: str = ""
    negative_terms: str = ""


@dataclass(slots=True)
class CandidateInfo:
    rank: int
    service_key: str
    service_name_short: str
    score: float | None
    confidence: str
    matched_terms: list[str]
    matched_aliases: list[str]
    matches_expected: bool


@dataclass(slots=True)
class ResolverInfo:
    status: str
    resolved_service_key: str
    resolved_service_name_short: str
    resolved_service_name_full: str
    resolved_score: float | None
    resolved_confidence: str
    candidates: list[CandidateInfo]
    top_candidate_name: str
    top_candidate_key: str
    top_candidate_score: float | None
    top_candidate_confidence: str
    second_candidate_score: float | None
    top_score_gap: float | None
    expected_in_candidates: bool
    expected_candidate_rank: int | None
    expected_candidate_score: float | None
    expected_candidate_name: str
    resolved_matches_expected: bool


@dataclass(slots=True)
class DiagnosticResult:
    flags: list[str]
    notes: list[str]
    primary_issue_class: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Массовый прогон банка вопросов через RAG2 HTTP API."
    )
    parser.add_argument("--questions-file", default=DEFAULT_QUESTIONS_FILE)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--channel", default="telegram")
    parser.add_argument("--external-user-prefix", default="question_bank_user")
    parser.add_argument("--external-chat-prefix", default="question_bank_chat")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--debug", action="store_true", help="Передавать debug=true в API.")
    parser.add_argument("--limit", type=int, default=0, help="Ограничить число вопросов.")
    parser.add_argument("--service-key", default="", help="Оставить только одну услугу.")
    parser.add_argument("--service-index", type=int, default=0, help="Оставить только одну услугу по номеру.")
    parser.add_argument("--category", default="", help="Оставить только одну категорию.")
    parser.add_argument("--question-id", default="", help="Запустить один конкретный вопрос.")
    parser.add_argument(
        "--sample-per-service",
        type=int,
        default=0,
        help="Взять не более N вопросов на каждую услугу.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Не повторять question_id, уже записанные в responses.jsonl в out-dir.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Остановиться после первой HTTP/API ошибки.",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Не выполнять запросы к API, а только построить отчёты по существующему responses.jsonl.",
    )
    parser.add_argument(
        "--responses-file",
        default="",
        help="Путь к responses.jsonl для режима --analyze-only. Если не задан, берётся out-dir/responses.jsonl.",
    )
    parser.add_argument(
        "--copy-responses-on-analyze",
        action="store_true",
        help="В режиме --analyze-only скопировать responses.jsonl в out-dir перед построением отчётов.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.analyze_only:
        return analyze_existing_responses(args, out_dir)

    questions = load_questions(Path(args.questions_file))
    questions = filter_questions(questions, args)

    if args.resume:
        done_ids = read_done_question_ids(out_dir / "responses.jsonl")
        questions = [q for q in questions if q.question_id not in done_ids]

    if args.limit > 0:
        questions = questions[: args.limit]

    run_meta = {
        "started_at": now_iso(),
        "script_version": "step47_diagnostic_v1",
        "questions_file": str(Path(args.questions_file).resolve()),
        "url": args.url,
        "out_dir": str(out_dir.resolve()),
        "channel": args.channel,
        "debug": bool(args.debug),
        "question_count": len(questions),
        "filters": {
            "service_key": args.service_key,
            "service_index": args.service_index,
            "category": args.category,
            "question_id": args.question_id,
            "sample_per_service": args.sample_per_service,
            "limit": args.limit,
            "resume": bool(args.resume),
        },
    }
    write_json(out_dir / "run_meta.json", run_meta)

    raw_path = out_dir / "responses.jsonl"
    completed = 0
    failed = 0

    print(f"Вопросов к прогону: {len(questions)}")
    print(f"Логи: {out_dir}")

    with raw_path.open("a", encoding="utf-8") as raw_file:
        for index, question in enumerate(questions, start=1):
            result = run_one_question(question, args, ordinal=index, total=len(questions))
            raw_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            raw_file.flush()

            completed += 1
            if has_technical_failure(result):
                failed += 1

            print_progress(index, len(questions), question, result)

            if args.fail_fast and has_technical_failure(result):
                break

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    rows = read_jsonl(raw_path)
    analysis = build_analysis(rows)
    write_analysis(out_dir, analysis, rows)

    run_meta["finished_at"] = now_iso()
    run_meta["completed"] = completed
    run_meta["failed_technical"] = failed
    write_json(out_dir / "run_meta.json", run_meta)

    print_finished(out_dir)
    return 0


def analyze_existing_responses(args: argparse.Namespace, out_dir: Path) -> int:
    source_path = Path(args.responses_file) if args.responses_file else out_dir / "responses.jsonl"
    if not source_path.exists():
        raise SystemExit(f"responses.jsonl не найден: {source_path}")

    target_path = out_dir / "responses.jsonl"
    if source_path.resolve() != target_path.resolve() and args.copy_responses_on_analyze:
        shutil.copyfile(source_path, target_path)
        read_path = target_path
    else:
        read_path = source_path

    rows = read_jsonl(read_path)
    analysis = build_analysis(rows)
    write_analysis(out_dir, analysis, rows)

    run_meta = {
        "started_at": now_iso(),
        "finished_at": now_iso(),
        "script_version": "step47_diagnostic_v1",
        "mode": "analyze_only",
        "source_responses_file": str(source_path.resolve()),
        "out_dir": str(out_dir.resolve()),
        "rows_count": len(rows),
    }
    write_json(out_dir / "run_meta.json", run_meta)
    print_finished(out_dir)
    return 0


def load_questions(path: Path) -> list[QuestionCase]:
    if not path.exists():
        raise SystemExit(f"Файл вопросов не найден: {path}")

    questions: list[QuestionCase] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        required = {
            "question_id",
            "service_index",
            "service_key",
            "service_name_short",
            "service_name_full",
            "category",
            "question_text",
            "expected_service_hint",
            "aliases_hint",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit("В файле вопросов нет колонок: " + ", ".join(sorted(missing)))

        for row in reader:
            questions.append(
                QuestionCase(
                    question_id=str(row["question_id"]).strip(),
                    service_index=int(row["service_index"]),
                    service_key=str(row["service_key"]).strip(),
                    service_name_short=str(row["service_name_short"]).strip(),
                    service_name_full=str(row["service_name_full"]).strip(),
                    category=str(row["category"]).strip(),
                    question_text=str(row["question_text"]).strip(),
                    expected_service_hint=str(row["expected_service_hint"]).strip(),
                    aliases_hint=str(row["aliases_hint"]).strip(),
                    question_profile=str(row.get("question_profile", "")).strip(),
                    negative_terms=str(row.get("negative_terms", "")).strip(),
                )
            )
    return questions


def filter_questions(questions: list[QuestionCase], args: argparse.Namespace) -> list[QuestionCase]:
    result = list(questions)

    if args.service_key:
        result = [q for q in result if q.service_key == args.service_key]

    if args.service_index > 0:
        result = [q for q in result if q.service_index == args.service_index]

    if args.category:
        result = [q for q in result if q.category == args.category]

    if args.question_id:
        result = [q for q in result if q.question_id == args.question_id]

    if args.sample_per_service > 0:
        counters: dict[str, int] = defaultdict(int)
        sampled: list[QuestionCase] = []
        for question in result:
            key = question.service_key or f"service_index_{question.service_index}"
            if counters[key] < args.sample_per_service:
                sampled.append(question)
                counters[key] += 1
        result = sampled

    return result


def read_done_question_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done

    for item in read_jsonl(path):
        question_id = str(item.get("question_id") or "").strip()
        if question_id:
            done.add(question_id)
    return done


def run_one_question(
    question: QuestionCase,
    args: argparse.Namespace,
    *,
    ordinal: int,
    total: int,
) -> dict[str, Any]:
    started = time.perf_counter()

    payload = {
        "question_text": question.question_text,
        "channel": args.channel,
        "external_user_id": f"{args.external_user_prefix}_{question.question_id}",
        "external_chat_id": f"{args.external_chat_prefix}_{question.question_id}",
        "debug": bool(args.debug),
    }

    result: dict[str, Any] = {
        "question_id": question.question_id,
        "ordinal": ordinal,
        "total": total,
        "service_index": question.service_index,
        "service_key": question.service_key,
        "service_name_short": question.service_name_short,
        "service_name_full": question.service_name_full,
        "category": question.category,
        "question_profile": question.question_profile,
        "negative_terms": question.negative_terms,
        "question_text": question.question_text,
        "expected_service_hint": question.expected_service_hint,
        "aliases_hint": question.aliases_hint,
        "started_at": now_iso(),
        "elapsed_seconds": None,
        "request": payload,
        "status_code": None,
        "ok": False,
        "answer_mode": None,
        "answer_text": "",
        "answer_text_short": None,
        "service_resolution": None,
        "warnings": [],
        "citations_count": 0,
        "error": None,
        "issue_flags": [],
        "quality_notes": [],
        "diagnostics": {},
    }

    try:
        request = Request(
            args.url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
            },
        )
        with urlopen(request, timeout=args.timeout_seconds) as response:
            response_text = response.read().decode("utf-8", errors="replace")
            response_json = json.loads(response_text)
            result["status_code"] = int(response.status)
            result["raw_response"] = response_json
            result["ok"] = bool(response_json.get("ok"))
            result["answer_mode"] = response_json.get("answer_mode")
            result["answer_text"] = str(response_json.get("answer_text") or "")
            result["answer_text_short"] = response_json.get("answer_text_short")
            result["service_resolution"] = response_json.get("service_resolution") or {}
            result["warnings"] = response_json.get("warnings") or []
            result["citations_count"] = len(response_json.get("citations") or [])

    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        result["status_code"] = int(exc.code)
        result["error"] = body
    except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        result["error"] = repr(exc)

    result["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    enrich_result_diagnostics(result)
    return result


def enrich_result_diagnostics(row: dict[str, Any]) -> None:
    resolver = extract_resolver_info(row)
    diagnostic = evaluate_result(row, resolver)

    row["diagnostics"] = {
        "primary_issue_class": diagnostic.primary_issue_class,
        "resolver_status": resolver.status,
        "resolved_service_key": resolver.resolved_service_key,
        "resolved_service_name_short": resolver.resolved_service_name_short,
        "resolved_service_name_full": resolver.resolved_service_name_full,
        "resolved_score": resolver.resolved_score,
        "resolved_confidence": resolver.resolved_confidence,
        "candidates_count": len(resolver.candidates),
        "top_candidate_key": resolver.top_candidate_key,
        "top_candidate_name": resolver.top_candidate_name,
        "top_candidate_score": resolver.top_candidate_score,
        "top_candidate_confidence": resolver.top_candidate_confidence,
        "second_candidate_score": resolver.second_candidate_score,
        "top_score_gap": resolver.top_score_gap,
        "expected_in_candidates": resolver.expected_in_candidates,
        "expected_candidate_rank": resolver.expected_candidate_rank,
        "expected_candidate_score": resolver.expected_candidate_score,
        "expected_candidate_name": resolver.expected_candidate_name,
        "resolved_matches_expected": resolver.resolved_matches_expected,
        "answer_length": len(str(row.get("answer_text") or "")),
        "answer_line_count": count_lines(row.get("answer_text") or ""),
        "technical_dump_score": technical_dump_score(row.get("answer_text") or ""),
        "target_terms_found": target_terms_found(row),
    }
    row["issue_flags"] = diagnostic.flags
    row["quality_notes"] = diagnostic.notes


def extract_resolver_info(row: dict[str, Any]) -> ResolverInfo:
    service_resolution = row.get("service_resolution") or {}
    if not isinstance(service_resolution, dict):
        service_resolution = {}

    status = normalize_status(
        service_resolution.get("status")
        or service_resolution.get("resolution_status")
        or service_resolution.get("resolver_status")
        or ""
    )
    raw_candidates = service_resolution.get("candidates") or []
    candidates: list[CandidateInfo] = []
    for index, raw_candidate in enumerate(raw_candidates, start=1):
        if not isinstance(raw_candidate, dict):
            continue
        candidate = CandidateInfo(
            rank=index,
            service_key=str(raw_candidate.get("service_key") or ""),
            service_name_short=str(raw_candidate.get("service_name_short") or ""),
            score=to_float_or_none(raw_candidate.get("score")),
            confidence=str(raw_candidate.get("confidence") or ""),
            matched_terms=list_of_strings(raw_candidate.get("matched_terms") or []),
            matched_aliases=list_of_strings(raw_candidate.get("matched_aliases") or []),
            matches_expected=False,
        )
        candidates.append(candidate)

    candidates = [
        CandidateInfo(
            rank=c.rank,
            service_key=c.service_key,
            service_name_short=c.service_name_short,
            score=c.score,
            confidence=c.confidence,
            matched_terms=c.matched_terms,
            matched_aliases=c.matched_aliases,
            matches_expected=candidate_matches_expected(row, c),
        )
        for c in candidates
    ]

    expected_candidates = [c for c in candidates if c.matches_expected]
    expected = expected_candidates[0] if expected_candidates else None
    top = candidates[0] if candidates else None
    second = candidates[1] if len(candidates) > 1 else None
    top_gap = None
    if top and second and top.score is not None and second.score is not None:
        top_gap = round(top.score - second.score, 4)

    resolved_name_short = str(service_resolution.get("service_name_short") or "")
    resolved_name_full = str(service_resolution.get("service_name_full") or "")
    resolved_key = str(service_resolution.get("service_key") or "")

    resolved_matches = resolved_matches_expected(
        row=row,
        resolved_service_key=resolved_key,
        resolved_service_name_short=resolved_name_short,
        resolved_service_name_full=resolved_name_full,
    )

    return ResolverInfo(
        status=status,
        resolved_service_key=resolved_key,
        resolved_service_name_short=resolved_name_short,
        resolved_service_name_full=resolved_name_full,
        resolved_score=to_float_or_none(service_resolution.get("score")),
        resolved_confidence=str(service_resolution.get("confidence") or ""),
        candidates=candidates,
        top_candidate_name=top.service_name_short if top else "",
        top_candidate_key=top.service_key if top else "",
        top_candidate_score=top.score if top else None,
        top_candidate_confidence=top.confidence if top else "",
        second_candidate_score=second.score if second else None,
        top_score_gap=top_gap,
        expected_in_candidates=bool(expected),
        expected_candidate_rank=expected.rank if expected else None,
        expected_candidate_score=expected.score if expected else None,
        expected_candidate_name=expected.service_name_short if expected else "",
        resolved_matches_expected=resolved_matches,
    )


def evaluate_result(row: dict[str, Any], resolver: ResolverInfo) -> DiagnosticResult:
    flags: list[str] = []
    notes: list[str] = []

    answer = str(row.get("answer_text") or "")
    answer_norm = normalize_text(answer)
    question_norm = normalize_text(row.get("question_text") or "")
    status_code = row.get("status_code")
    ok = bool(row.get("ok"))
    answer_length = len(answer)

    if status_code != 200:
        flags.append("http_error")

    if not ok:
        flags.append("api_not_ok")

    if not answer.strip():
        flags.append("empty_answer")
        return DiagnosticResult(unique(flags), unique(notes), primary_issue_class(flags))

    if len(answer.strip()) < 80:
        flags.append("answer_very_short")

    if answer_length > VERY_LONG_ANSWER_LIMIT:
        flags.append("answer_very_long")

    if answer_length > EXTREMELY_LONG_ANSWER_LIMIT:
        flags.append("answer_extremely_long")

    if is_safe_or_uncertain(answer_norm):
        flags.append("answer_safe_no_answer")

    if row.get("warnings"):
        flags.append("has_warnings")

    dump_score = technical_dump_score(answer)
    if dump_score >= 3:
        flags.append("answer_technical_table_dump")

    if eligibility_header_noise_score(answer) > 0:
        flags.append("answer_identifier_header_noise")

    # Resolver diagnostics.
    if resolver.status in {"", "unknown"}:
        flags.append("resolver_status_missing")
    elif resolver.status == "not_found":
        flags.append("resolver_not_found")
        if resolver.candidates:
            flags.append("resolver_not_found_with_candidates")
        else:
            flags.append("resolver_no_candidates")
    elif resolver.status == "ambiguous":
        if resolver.expected_candidate_rank == 1:
            flags.append("resolver_ambiguous_expected_top1")
        elif resolver.expected_in_candidates:
            flags.append("resolver_ambiguous_expected_in_candidates")
        else:
            flags.append("resolver_ambiguous_expected_not_in_candidates")
    elif resolver.status == "resolved":
        if resolver.resolved_matches_expected:
            notes.append("resolver_resolved_expected")
        else:
            flags.append("resolver_resolved_other_service")
    elif resolver.status == "service_discovery":
        notes.append("resolver_service_discovery")
    else:
        notes.append(f"resolver_unknown_status:{resolver.status}")

    if resolver.expected_in_candidates:
        notes.append(f"expected_candidate_rank:{resolver.expected_candidate_rank}")
    elif resolver.candidates:
        notes.append("expected_service_not_in_candidates")

    if resolver.top_score_gap is not None:
        notes.append(f"top_score_gap:{resolver.top_score_gap}")

    # Answer content diagnostics.
    found_target_terms = target_terms_found(row)
    if found_target_terms:
        notes.append("answer_mentions_target")
    else:
        if is_resolved_eligibility_category_answer(row, resolver, question_norm, answer_norm):
            # Для вопросов «могу ли получить / подать заявление» нормальный
            # ответ может перечислять категории заявителей, не повторяя
            # формулировку expected_service_hint из банка вопросов.
            notes.append("answer_has_applicant_category_evidence")
        else:
            # fix_08: если resolver уже не выбрал ожидаемую услугу или ушёл
            # в ambiguous/not_found, отсутствие expected_service_hint в ответе
            # является следствием resolver-дефекта и только шумит в сводке.
            # Оставляем этот флаг для случая, когда услуга выбрана правильно,
            # но сам текст ответа не отражает ожидаемую тему.
            if resolver.status == "resolved" and resolver.resolved_matches_expected:
                flags.append("answer_target_not_mentioned")
            elif resolver.status in {"resolved", "ambiguous", "not_found"}:
                notes.append("answer_target_check_skipped_due_resolver")

    if "документ" in question_norm and "документ" not in answer_norm:
        flags.append("answer_documents_question_without_document_word")

    if any(word in question_norm for word in ("отказ", "отказать", "приостанов")):
        if not any(word in answer_norm for word in ("отказ", "приостанов")):
            flags.append("answer_rejection_question_without_rejection_word")

    if any(word in question_norm for word in ("срок", "когда", "сколько ждать", "уведомят")):
        if not any(word in answer_norm for word in ("срок", "рабоч", "календар", "дн", "уведом")):
            flags.append("answer_deadline_question_without_time_word")

    weak_question_note = classify_generic_question(row)
    if weak_question_note:
        notes.append(weak_question_note)

    return DiagnosticResult(unique(flags), unique(notes), primary_issue_class(flags))


def is_resolved_eligibility_category_answer(
    row: dict[str, Any],
    resolver: ResolverInfo,
    question_norm: str,
    answer_norm: str,
) -> bool:
    if resolver.status != "resolved" or not resolver.resolved_matches_expected:
        return False

    if "категория заявителей" not in answer_norm:
        return False

    if is_safe_or_uncertain(answer_norm):
        return False

    eligibility_markers = (
        "могу ли",
        "можно ли",
        "положено",
        "имею право",
        "право на",
        "получить помощь",
        "подать заявление",
    )
    if any(marker in question_norm for marker in eligibility_markers):
        return True

    profile = str(row.get("question_profile") or "").strip().lower()
    return bool(profile) and not any(
        marker in profile
        for marker in ("documents", "rejection", "deadline", "procedure", "form")
    )


def primary_issue_class(flags: list[str]) -> str:
    if not flags:
        return "ok"
    ordered_prefixes = [
        ("http_error", "technical"),
        ("api_not_ok", "technical"),
        ("empty_answer", "technical"),
        ("resolver_resolved_other_service", "resolver"),
        ("resolver_ambiguous_expected_top1", "resolver"),
        ("resolver_ambiguous_expected_in_candidates", "resolver"),
        ("resolver_ambiguous_expected_not_in_candidates", "resolver"),
        ("resolver_not_found", "resolver"),
        ("answer_safe_no_answer", "answer"),
        ("answer_technical_table_dump", "answer"),
        ("answer_identifier_header_noise", "answer"),
        ("answer_extremely_long", "answer"),
        ("answer_very_long", "answer"),
        ("answer_target_not_mentioned", "answer"),
    ]
    flag_set = set(flags)
    for flag, group in ordered_prefixes:
        if flag in flag_set:
            return group
    return "other"


def candidate_matches_expected(row: dict[str, Any], candidate: CandidateInfo) -> bool:
    expected_key = str(row.get("service_key") or "").strip()
    if expected_key and expected_key.startswith("svc_") and candidate.service_key == expected_key:
        return True

    candidate_name_norm = normalize_text(candidate.service_name_short)
    expected_short_norm = normalize_text(row.get("service_name_short") or "")
    expected_full_norm = normalize_text(row.get("service_name_full") or "")
    expected_hint_norm = normalize_text(row.get("expected_service_hint") or "")

    if exact_or_contained_name_match(expected_short_norm, candidate_name_norm):
        return True

    if expected_full_norm and candidate_name_norm:
        if candidate_name_norm in expected_full_norm or expected_full_norm in candidate_name_norm:
            return True

    candidate_alias_blob = normalize_text(" ".join(candidate.matched_aliases + candidate.matched_terms))
    if expected_hint_norm and len(expected_hint_norm) >= 8 and expected_hint_norm in candidate_alias_blob:
        return True

    # Last-resort token overlap only for non-generic service names.
    if meaningful_token_overlap(expected_short_norm, candidate_name_norm) >= 0.8:
        return True

    return False


def resolved_matches_expected(
    *,
    row: dict[str, Any],
    resolved_service_key: str,
    resolved_service_name_short: str,
    resolved_service_name_full: str,
) -> bool:
    expected_key = str(row.get("service_key") or "").strip()
    if expected_key and expected_key.startswith("svc_") and resolved_service_key == expected_key:
        return True

    resolved_short_norm = normalize_text(resolved_service_name_short)
    resolved_full_norm = normalize_text(resolved_service_name_full)
    expected_short_norm = normalize_text(row.get("service_name_short") or "")
    expected_full_norm = normalize_text(row.get("service_name_full") or "")

    if exact_or_contained_name_match(expected_short_norm, resolved_short_norm):
        return True
    if exact_or_contained_name_match(expected_short_norm, resolved_full_norm):
        return True
    if expected_full_norm and resolved_full_norm:
        if expected_full_norm == resolved_full_norm:
            return True
    if meaningful_token_overlap(expected_short_norm, resolved_short_norm) >= 0.8:
        return True
    return False


def exact_or_contained_name_match(expected_norm: str, actual_norm: str) -> bool:
    if not expected_norm or not actual_norm:
        return False
    if expected_norm == actual_norm:
        return True
    if len(expected_norm) >= 12 and expected_norm in actual_norm:
        return True
    if len(actual_norm) >= 12 and actual_norm in expected_norm:
        return True
    return False


def meaningful_token_overlap(left_norm: str, right_norm: str) -> float:
    left = meaningful_tokens(left_norm)
    right = meaningful_tokens(right_norm)
    if not left or not right:
        return 0.0
    intersection = left & right
    denominator = min(len(left), len(right))
    if denominator == 0:
        return 0.0
    return len(intersection) / denominator


def meaningful_tokens(value: str) -> set[str]:
    stopwords = {
        "предоставление", "осуществление", "выплата", "меры", "мера", "социальной", "социальная",
        "поддержки", "компенсация", "пособие", "государственной", "услуги", "услуга",
        "граждан", "края", "красноярского", "ежемесячная", "ежегодная", "единовременная",
        "назначение", "получение", "праве", "право", "мсп", "отдельным", "категориям",
    }
    return {token for token in normalize_text(value).split() if len(token) >= 4 and token not in stopwords}


def target_terms_found(row: dict[str, Any]) -> list[str]:
    answer_norm = normalize_text(row.get("answer_text") or "")
    found: list[str] = []
    for term in build_target_terms(row):
        term_norm = normalize_text(term)
        if term_norm and term_norm in answer_norm:
            found.append(term)
    return unique(found)


def build_target_terms(row: dict[str, Any]) -> list[str]:
    raw_terms = [
        str(row.get("expected_service_hint") or ""),
        str(row.get("service_name_short") or ""),
    ]
    aliases = str(row.get("aliases_hint") or "")
    raw_terms.extend([term.strip() for term in aliases.split(";") if term.strip()])

    result: list[str] = []
    for term in raw_terms:
        normalized = normalize_text(term)
        if len(normalized) < 5:
            continue
        if len(normalized) > 90:
            parts = re.split(r"[,;:()/-]", term)
            result.extend(part.strip() for part in parts if len(part.strip()) >= 8)
        else:
            result.append(term.strip())

    # Add meaningful fragments of short service name as soft target terms.
    service_short = str(row.get("service_name_short") or "")
    for part in re.split(r"[,;:()/-]", service_short):
        part = part.strip()
        if len(normalize_text(part)) >= 8:
            result.append(part)

    return unique(result)[:14]


def is_safe_or_uncertain(answer_norm: str) -> bool:
    safe_markers = [
        "не удалось",
        "не могу надежно",
        "не могу надёжно",
        "не найдено достаточно",
        "недостаточно данных",
        "не найдено",
        "уточните",
        "нужно уточнить",
        "сервис временно недоступен",
        "я не нашел",
        "я не нашёл",
        "не удалось надежно",
        "не удалось надёжно",
    ]
    return any(marker in answer_norm for marker in safe_markers)


def eligibility_header_noise_score(answer: str) -> int:
    answer_norm = normalize_text(answer)
    markers = [
        "категория заявителей наименование признака заявителя",
        "категория заявителей идентификаторы категорий",
        "категория заявителей перечень результатов предоставления",
        "категория заявителей результат предоставления государственной услуги",
        "категория заявителей принятие решения",
    ]
    return sum(1 for marker in markers if marker in answer_norm)


def technical_dump_score(answer: str) -> int:
    answer_norm = normalize_text(answer)
    markers = [
        "таблица",
        "колонки таблицы",
        "n п п",
        "п п",
        "идентификаторы категорий",
        "metadata json",
        "row json",
        "document table row",
        "table row",
        "исчерпывающий перечень документов",
        "исчерпывающий перечень оснований",
    ]
    score = 0
    for marker in markers:
        if marker in answer_norm:
            score += 1
    # Repeated table-like fragments are especially bad for a user-facing answer.
    if answer_norm.count("таблица") >= 3:
        score += 2
    if answer_norm.count("n п п") >= 2 or answer_norm.count("п п") >= 4:
        score += 1
    return score


def classify_generic_question(row: dict[str, Any]) -> str:
    question_norm = normalize_text(row.get("question_text") or "")
    generic_patterns = [
        "можно ли получить помощь от соцзащиты",
        "какую помощь можно оформить",
        "что положено",
        "положена ли помощь",
    ]
    if any(pattern in question_norm for pattern in generic_patterns):
        if not target_terms_found(row):
            return "question_may_be_too_generic_for_expected_service_eval"
    return ""


def has_technical_failure(result: dict[str, Any]) -> bool:
    flags = set(result.get("issue_flags") or [])
    return bool(flags & {"http_error", "api_not_ok", "empty_answer"})


def normalize_status(value: Any) -> str:
    text = str(value or "").strip().casefold()
    if not text:
        return ""
    aliases = {
        "resolved": "resolved",
        "ambiguous": "ambiguous",
        "not_found": "not_found",
        "not found": "not_found",
        "service_discovery": "service_discovery",
        "discovery": "service_discovery",
    }
    return aliases.get(text, text)


def normalize_text(value: Any) -> str:
    text = str(value or "").casefold().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def list_of_strings(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None:
        return []
    return [str(value)] if str(value).strip() else []


def to_float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def count_lines(value: str) -> int:
    text = str(value or "")
    if not text:
        return 0
    return len(text.splitlines())


def unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value)
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def print_progress(index: int, total: int, question: QuestionCase, result: dict[str, Any]) -> None:
    flags = ",".join(result.get("issue_flags") or [])
    diagnostics = result.get("diagnostics") or {}
    status = result.get("status_code")
    elapsed = result.get("elapsed_seconds")
    answer_mode = result.get("answer_mode") or "-"
    resolver_status = diagnostics.get("resolver_status") or "-"
    top_name = diagnostics.get("top_candidate_name") or "-"
    flag_text = f" flags={flags}" if flags else ""
    print(
        f"[{index}/{total}] {question.question_id} "
        f"status={status} ok={result.get('ok')} mode={answer_mode} "
        f"resolver={resolver_status} top={preview(top_name, 45)} "
        f"time={elapsed}s{flag_text}"
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    enrich_result_diagnostics(row)
                    rows.append(row)
            except json.JSONDecodeError:
                print(f"Не удалось прочитать JSONL строку {line_number}: {path}", file=sys.stderr)
    return rows


def build_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    by_service: dict[str, Counter[str]] = defaultdict(Counter)
    by_mode: Counter[str] = Counter()
    by_resolver_status: Counter[str] = Counter()
    by_primary_issue_class: Counter[str] = Counter()
    issue_counter: Counter[str] = Counter()
    elapsed_values: list[float] = []
    answer_lengths: list[int] = []
    issues: list[dict[str, Any]] = []

    for row in rows:
        diagnostics = row.get("diagnostics") or {}
        category = str(row.get("category") or "")
        service_name = str(row.get("service_name_short") or "")
        mode = str(row.get("answer_mode") or "-")
        flags = list(row.get("issue_flags") or [])
        resolver_status = str(diagnostics.get("resolver_status") or "-")
        primary_issue = str(diagnostics.get("primary_issue_class") or "ok")
        answer_length = int(diagnostics.get("answer_length") or len(str(row.get("answer_text") or "")))

        by_category[category]["total"] += 1
        by_service[service_name]["total"] += 1
        by_mode[mode] += 1
        by_resolver_status[resolver_status] += 1
        by_primary_issue_class[primary_issue] += 1
        answer_lengths.append(answer_length)

        if row.get("ok"):
            by_category[category]["ok"] += 1
            by_service[service_name]["ok"] += 1

        if flags:
            by_category[category]["with_flags"] += 1
            by_service[service_name]["with_flags"] += 1
            for flag in flags:
                issue_counter[flag] += 1
                by_category[category][flag] += 1
                by_service[service_name][flag] += 1
            issues.append(row)

        elapsed = row.get("elapsed_seconds")
        if isinstance(elapsed, (int, float)):
            elapsed_values.append(float(elapsed))

    elapsed_values.sort()
    answer_lengths.sort()
    avg_elapsed = round(sum(elapsed_values) / len(elapsed_values), 4) if elapsed_values else None
    p95_elapsed = percentile(elapsed_values, 95)
    avg_answer_length = round(sum(answer_lengths) / len(answer_lengths), 1) if answer_lengths else None
    p95_answer_length = percentile_int(answer_lengths, 95)

    return {
        "total": len(rows),
        "ok": sum(1 for row in rows if row.get("ok")),
        "with_flags": sum(1 for row in rows if row.get("issue_flags")),
        "answer_modes": dict(by_mode),
        "resolver_statuses": dict(by_resolver_status),
        "primary_issue_classes": dict(by_primary_issue_class),
        "issues": issues,
        "issue_counts": dict(issue_counter),
        "by_category": {key: dict(counter) for key, counter in sorted(by_category.items())},
        "by_service": {key: dict(counter) for key, counter in sorted(by_service.items())},
        "avg_elapsed_seconds": avg_elapsed,
        "p95_elapsed_seconds": p95_elapsed,
        "avg_answer_length": avg_answer_length,
        "p95_answer_length": p95_answer_length,
        "max_answer_length": max(answer_lengths) if answer_lengths else None,
    }


def percentile(values: list[float], percent: int) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 4)
    index = (len(values) - 1) * percent / 100
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    result = values[lower] * (1 - weight) + values[upper] * weight
    return round(result, 4)


def percentile_int(values: list[int], percent: int) -> int | None:
    value = percentile([float(v) for v in values], percent)
    return int(round(value)) if value is not None else None


def write_analysis(out_dir: Path, analysis: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    write_json(out_dir / "summary.json", analysis_without_rows(analysis))
    write_summary_md(out_dir / "summary.md", analysis)
    write_counter_tsv(out_dir / "by_category.tsv", analysis["by_category"], key_name="category")
    write_counter_tsv(out_dir / "by_service.tsv", analysis["by_service"], key_name="service_name")
    write_issues_tsv(out_dir / "issues.tsv", analysis["issues"])
    write_answers_tsv(out_dir / "answers.tsv", rows)
    write_resolver_summary_tsv(out_dir / "resolver_summary.tsv", rows)
    write_selected_cases(out_dir / "top_bad_resolver_cases.tsv", rows, resolver_problem_filter)
    write_selected_cases(out_dir / "safe_no_answer_cases.tsv", rows, lambda row: "answer_safe_no_answer" in row_flags(row))
    write_selected_cases(out_dir / "long_answers.tsv", rows, lambda row: bool(set(row_flags(row)) & {"answer_very_long", "answer_extremely_long"}))
    write_selected_cases(out_dir / "technical_table_dump.tsv", rows, lambda row: "answer_technical_table_dump" in row_flags(row))
    write_selected_cases(out_dir / "resolved_other_service.tsv", rows, lambda row: "resolver_resolved_other_service" in row_flags(row))
    write_missing_alias_suggestions_tsv(out_dir / "missing_alias_suggestions.tsv", rows)


def analysis_without_rows(analysis: dict[str, Any]) -> dict[str, Any]:
    result = dict(analysis)
    result.pop("issues", None)
    return result


def write_summary_md(path: Path, analysis: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Сводка массового прогона вопросов RAG2")
    lines.append("")
    lines.append(f"- Всего вопросов: {analysis['total']}")
    lines.append(f"- Успешных API-ответов: {analysis['ok']}")
    lines.append(f"- Ответов с флагами проверки: {analysis['with_flags']}")
    lines.append(f"- Среднее время ответа, сек.: {analysis['avg_elapsed_seconds']}")
    lines.append(f"- 95-й процентиль времени ответа, сек.: {analysis['p95_elapsed_seconds']}")
    lines.append(f"- Средняя длина ответа, знаков: {analysis['avg_answer_length']}")
    lines.append(f"- 95-й процентиль длины ответа, знаков: {analysis['p95_answer_length']}")
    lines.append(f"- Максимальная длина ответа, знаков: {analysis['max_answer_length']}")
    lines.append("")

    lines.append("## Режимы ответа")
    for mode, count in sorted(analysis["answer_modes"].items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- {mode}: {count}")
    lines.append("")

    lines.append("## Статусы определения услуги")
    for status, count in sorted(analysis["resolver_statuses"].items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- {status}: {count}")
    lines.append("")

    lines.append("## Основной класс проблемы")
    for issue_class, count in sorted(analysis["primary_issue_classes"].items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- {issue_class}: {count}")
    lines.append("")

    lines.append("## Флаги проблем")
    if analysis["issue_counts"]:
        for flag, count in sorted(analysis["issue_counts"].items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"- {flag}: {count}")
    else:
        lines.append("- Флаги проблем не обнаружены.")
    lines.append("")

    lines.append("## Категории с наибольшим числом флагов")
    ranked_categories = sorted(
        analysis["by_category"].items(),
        key=lambda item: (-item[1].get("with_flags", 0), item[0]),
    )
    for category, stats in ranked_categories[:20]:
        lines.append(
            f"- {category}: всего {stats.get('total', 0)}, "
            f"с флагами {stats.get('with_flags', 0)}, ok {stats.get('ok', 0)}, "
            f"resolver_ambiguous_top1 {stats.get('resolver_ambiguous_expected_top1', 0)}, "
            f"safe {stats.get('answer_safe_no_answer', 0)}, "
            f"long {stats.get('answer_very_long', 0)}"
        )
    lines.append("")

    lines.append("## Услуги с наибольшим числом флагов")
    ranked_services = sorted(
        analysis["by_service"].items(),
        key=lambda item: (-item[1].get("with_flags", 0), item[0]),
    )
    for service, stats in ranked_services[:30]:
        lines.append(
            f"- {service}: всего {stats.get('total', 0)}, "
            f"с флагами {stats.get('with_flags', 0)}, ok {stats.get('ok', 0)}, "
            f"resolver {stats.get('resolver_ambiguous_expected_top1', 0) + stats.get('resolver_resolved_other_service', 0) + stats.get('resolver_not_found', 0)}, "
            f"safe {stats.get('answer_safe_no_answer', 0)}, "
            f"long {stats.get('answer_very_long', 0)}"
        )
    lines.append("")

    lines.append("## Дополнительные отчёты")
    lines.append("- `answers.tsv` — все вопросы и ответы с расширенной диагностикой.")
    lines.append("- `issues.tsv` — только строки с флагами.")
    lines.append("- `resolver_summary.tsv` — сводка по определению услуги.")
    lines.append("- `top_bad_resolver_cases.tsv` — самые полезные случаи для правки service_resolver.")
    lines.append("- `safe_no_answer_cases.tsv` — safe/no-answer для ручного разбора.")
    lines.append("- `long_answers.tsv` — слишком длинные ответы.")
    lines.append("- `technical_table_dump.tsv` — технические табличные дампы.")
    lines.append("- `resolved_other_service.tsv` — случаи выбора другой услуги.")
    lines.append("- `missing_alias_suggestions.tsv` — подсказки, какие слова можно добавить в алиасы услуги или общий словарь живого языка.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_counter_tsv(path: Path, data: dict[str, dict[str, int]], *, key_name: str) -> None:
    optional_flags = [
        "resolver_not_found",
        "resolver_ambiguous_expected_top1",
        "resolver_ambiguous_expected_in_candidates",
        "resolver_ambiguous_expected_not_in_candidates",
        "resolver_resolved_other_service",
        "answer_safe_no_answer",
        "answer_very_long",
        "answer_extremely_long",
        "answer_technical_table_dump",
        "answer_identifier_header_noise",
        "answer_target_not_mentioned",
    ]
    headers = [key_name, "total", "ok", "with_flags"] + optional_flags
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for key, stats in sorted(data.items()):
            row = {
                key_name: key,
                "total": stats.get("total", 0),
                "ok": stats.get("ok", 0),
                "with_flags": stats.get("with_flags", 0),
            }
            for flag in optional_flags:
                row[flag] = stats.get(flag, 0)
            writer.writerow(row)


def base_case_headers() -> list[str]:
    return [
        "question_id",
        "service_index",
        "expected_service_key",
        "expected_service_name_short",
        "category",
        "question_profile",
        "question_text",
        "expected_service_hint",
        "resolver_status",
        "resolved_service_key",
        "resolved_service_name_short",
        "resolved_score",
        "resolved_confidence",
        "candidates_count",
        "top_candidate_key",
        "top_candidate_name",
        "top_candidate_score",
        "top_candidate_confidence",
        "second_candidate_score",
        "top_score_gap",
        "expected_in_candidates",
        "expected_candidate_rank",
        "expected_candidate_score",
        "expected_candidate_name",
        "resolved_matches_expected",
        "status_code",
        "ok",
        "answer_mode",
        "elapsed_seconds",
        "answer_length",
        "answer_line_count",
        "technical_dump_score",
        "primary_issue_class",
        "issue_flags",
        "quality_notes",
        "answer_preview",
        "error",
    ]


def row_for_case(row: dict[str, Any], *, include_full_answer: bool = False) -> dict[str, Any]:
    diagnostics = row.get("diagnostics") or {}
    result = {
        "question_id": row.get("question_id"),
        "service_index": row.get("service_index"),
        "expected_service_key": row.get("service_key"),
        "expected_service_name_short": row.get("service_name_short"),
        "category": row.get("category"),
        "question_profile": row.get("question_profile"),
        "question_text": row.get("question_text"),
        "expected_service_hint": row.get("expected_service_hint"),
        "resolver_status": diagnostics.get("resolver_status"),
        "resolved_service_key": diagnostics.get("resolved_service_key"),
        "resolved_service_name_short": diagnostics.get("resolved_service_name_short"),
        "resolved_score": diagnostics.get("resolved_score"),
        "resolved_confidence": diagnostics.get("resolved_confidence"),
        "candidates_count": diagnostics.get("candidates_count"),
        "top_candidate_key": diagnostics.get("top_candidate_key"),
        "top_candidate_name": diagnostics.get("top_candidate_name"),
        "top_candidate_score": diagnostics.get("top_candidate_score"),
        "top_candidate_confidence": diagnostics.get("top_candidate_confidence"),
        "second_candidate_score": diagnostics.get("second_candidate_score"),
        "top_score_gap": diagnostics.get("top_score_gap"),
        "expected_in_candidates": diagnostics.get("expected_in_candidates"),
        "expected_candidate_rank": diagnostics.get("expected_candidate_rank"),
        "expected_candidate_score": diagnostics.get("expected_candidate_score"),
        "expected_candidate_name": diagnostics.get("expected_candidate_name"),
        "resolved_matches_expected": diagnostics.get("resolved_matches_expected"),
        "status_code": row.get("status_code"),
        "ok": row.get("ok"),
        "answer_mode": row.get("answer_mode"),
        "elapsed_seconds": row.get("elapsed_seconds"),
        "answer_length": diagnostics.get("answer_length"),
        "answer_line_count": diagnostics.get("answer_line_count"),
        "technical_dump_score": diagnostics.get("technical_dump_score"),
        "primary_issue_class": diagnostics.get("primary_issue_class"),
        "issue_flags": "; ".join(row.get("issue_flags") or []),
        "quality_notes": "; ".join(row.get("quality_notes") or []),
        "answer_preview": preview(row.get("answer_text") or "", MAX_PREVIEW_LENGTH),
        "error": preview(row.get("error") or "", MAX_PREVIEW_LENGTH),
    }
    if include_full_answer:
        result["answer_text"] = row.get("answer_text") or ""
    return result


def write_issues_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = base_case_headers()
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row_for_case(row))


def write_answers_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = base_case_headers() + ["answer_text"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row_for_case(row, include_full_answer=True))


def write_resolver_summary_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = [
        "resolver_status",
        "total",
        "resolved_expected",
        "expected_top1",
        "expected_in_candidates",
        "expected_not_in_candidates",
        "avg_top_score",
        "avg_expected_score",
        "safe_no_answer",
        "very_long",
    ]
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        status = str((row.get("diagnostics") or {}).get("resolver_status") or "-")
        buckets[status].append(row)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for status, status_rows in sorted(buckets.items()):
            top_scores = [to_float_or_none((r.get("diagnostics") or {}).get("top_candidate_score")) for r in status_rows]
            expected_scores = [to_float_or_none((r.get("diagnostics") or {}).get("expected_candidate_score")) for r in status_rows]
            top_scores = [v for v in top_scores if v is not None]
            expected_scores = [v for v in expected_scores if v is not None]
            writer.writerow({
                "resolver_status": status,
                "total": len(status_rows),
                "resolved_expected": sum(1 for r in status_rows if (r.get("diagnostics") or {}).get("resolved_matches_expected")),
                "expected_top1": sum(1 for r in status_rows if (r.get("diagnostics") or {}).get("expected_candidate_rank") == 1),
                "expected_in_candidates": sum(1 for r in status_rows if (r.get("diagnostics") or {}).get("expected_in_candidates")),
                "expected_not_in_candidates": sum(1 for r in status_rows if not (r.get("diagnostics") or {}).get("expected_in_candidates")),
                "avg_top_score": round(sum(top_scores) / len(top_scores), 4) if top_scores else "",
                "avg_expected_score": round(sum(expected_scores) / len(expected_scores), 4) if expected_scores else "",
                "safe_no_answer": sum(1 for r in status_rows if "answer_safe_no_answer" in row_flags(r)),
                "very_long": sum(1 for r in status_rows if "answer_very_long" in row_flags(r)),
            })


def write_selected_cases(path: Path, rows: list[dict[str, Any]], predicate) -> None:
    selected = [row for row in rows if predicate(row)]
    selected = sorted(selected, key=case_sort_key)
    write_issues_tsv(path, selected)


def write_missing_alias_suggestions_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = [
        "question_id",
        "service_index",
        "expected_service_name_short",
        "category",
        "question_text",
        "issue_flags",
        "resolver_status",
        "top_candidate_name",
        "top_candidate_score",
        "expected_candidate_rank",
        "suggested_service_aliases",
        "suggested_runtime_vocabulary_terms",
        "where_to_edit",
    ]
    problem_flags = {
        "resolver_not_found",
        "resolver_no_candidates",
        "resolver_ambiguous_expected_top1",
        "resolver_ambiguous_expected_in_candidates",
        "resolver_ambiguous_expected_not_in_candidates",
        "resolver_resolved_other_service",
        "answer_safe_no_answer",
        "answer_target_not_mentioned",
    }
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            flags = set(row_flags(row))
            if not flags.intersection(problem_flags):
                continue
            suggestions = suggest_alias_terms(row)
            runtime_terms = [term for term in suggestions if normalize_text(term) in _RUNTIME_VOCABULARY_HINTS]
            if not suggestions and not runtime_terms:
                continue
            diagnostics = row.get("diagnostics") or {}
            writer.writerow({
                "question_id": row.get("question_id"),
                "service_index": row.get("service_index"),
                "expected_service_name_short": row.get("service_name_short"),
                "category": row.get("category"),
                "question_text": row.get("question_text"),
                "issue_flags": "; ".join(row_flags(row)),
                "resolver_status": diagnostics.get("resolver_status"),
                "top_candidate_name": diagnostics.get("top_candidate_name"),
                "top_candidate_score": diagnostics.get("top_candidate_score"),
                "expected_candidate_rank": diagnostics.get("expected_candidate_rank"),
                "suggested_service_aliases": "; ".join(suggestions),
                "suggested_runtime_vocabulary_terms": "; ".join(runtime_terms),
                "where_to_edit": choose_alias_edit_target(runtime_terms),
            })


_RUNTIME_VOCABULARY_HINTS = {
    "коммуналка", "жкх", "жку", "квартплата", "садик", "детсад", "маткапитал",
    "автошкола", "похороны", "памятник", "зуб", "зубы", "зубные", "протезы",
    "печка", "дрова", "чс", "тжс", "тср", "чаэс", "чернобыль", "донор", "доноры",
    "соцработник", "гемодиализ", "лагерь", "затопило", "сгорел", "холодильник",
}


def suggest_alias_terms(row: dict[str, Any]) -> list[str]:
    question_text = str(row.get("question_text") or "")
    existing_blob = normalize_text(" ".join([
        str(row.get("aliases_hint") or ""),
        str(row.get("service_name_short") or ""),
        str(row.get("service_name_full") or ""),
        str(row.get("expected_service_hint") or ""),
    ]))

    words = re.findall(r"[0-9a-zа-я]+", question_text.casefold().replace("ё", "е"))
    normalized_words = [word for word in words if len(word) >= 4 and word not in _EVAL_ALIAS_GENERIC_WORDS]
    suggestions: list[str] = []

    for word in normalized_words:
        if normalize_text(word) not in existing_blob:
            suggestions.append(word)

    for size in (3, 2):
        for index in range(0, max(0, len(words) - size + 1)):
            phrase_words = words[index:index + size]
            if any(word in _EVAL_ALIAS_GENERIC_WORDS for word in phrase_words):
                continue
            phrase = " ".join(phrase_words)
            phrase_norm = normalize_text(phrase)
            if len(phrase_norm) < 8:
                continue
            if phrase_norm not in existing_blob:
                suggestions.append(phrase)

    return unique(suggestions)[:8]


_EVAL_ALIAS_GENERIC_WORDS = {
    "можно", "получить", "оформить", "положено", "положена", "положены", "помощь",
    "помогите", "какие", "какой", "какая", "нужно", "надо", "соцзащита", "заявление",
    "документы", "документ", "выплата", "выплаты", "компенсация", "пособие", "услуга",
    "есть", "куда", "обращаться", "если", "меня", "хочу", "могу", "могут", "почему",
    "нужна", "нужен", "нужны", "самому", "самой", "тяжело", "соцзащиты", "соцзащиту",
}


def choose_alias_edit_target(runtime_terms: list[str]) -> str:
    if runtime_terms:
        return "проверить оба места: общие слова — app/config/runtime_vocabulary.json; алиасы услуги — Актуальный_приказ5.xlsx"
    return "Актуальный_приказ5.xlsx / колонка 'Ключевые слова / алиасы'"


def resolver_problem_filter(row: dict[str, Any]) -> bool:
    flags = set(row_flags(row))
    return bool(flags & {
        "resolver_resolved_other_service",
        "resolver_ambiguous_expected_top1",
        "resolver_ambiguous_expected_in_candidates",
        "resolver_ambiguous_expected_not_in_candidates",
        "resolver_not_found",
    })


def case_sort_key(row: dict[str, Any]) -> tuple[int, float, str]:
    diagnostics = row.get("diagnostics") or {}
    flags = set(row_flags(row))
    priority = 50
    if "resolver_resolved_other_service" in flags:
        priority = 1
    elif "resolver_ambiguous_expected_top1" in flags:
        priority = 2
    elif "resolver_ambiguous_expected_in_candidates" in flags:
        priority = 3
    elif "resolver_not_found" in flags:
        priority = 4
    elif "answer_safe_no_answer" in flags:
        priority = 5
    elif "answer_technical_table_dump" in flags:
        priority = 6
    elif "answer_very_long" in flags:
        priority = 7
    score = diagnostics.get("top_candidate_score")
    score_value = -float(score) if isinstance(score, (int, float)) else 0.0
    return (priority, score_value, str(row.get("question_id") or ""))


def row_flags(row: dict[str, Any]) -> list[str]:
    return list(row.get("issue_flags") or [])


def preview(value: Any, max_length: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_length:
        return text
    return text[: max_length - 1].rstrip() + "…"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def print_finished(out_dir: Path) -> None:
    print("\nГотово.")
    print(f"Сводка: {out_dir / 'summary.md'}")
    print(f"Проблемы: {out_dir / 'issues.tsv'}")
    print(f"Диагностика resolver: {out_dir / 'resolver_summary.tsv'}")


if __name__ == "__main__":
    raise SystemExit(main())
