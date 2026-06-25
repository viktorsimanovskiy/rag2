from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


EXPECTED_COUNTS = {
    "service_registry_total": 110,
    "active_services": 110,
    "active_documents": 110,
    "active_documents_without_service_key": 0,
    "active_documents_with_broken_service_link": 0,
    "active_services_without_active_document": 0,
    "services_without_aliases": 0,
    "trim_for_rag_documents": 110,
    "preprocessing_applied_documents": 110,
    "trim_safety_passed_documents": 110,
    "tail_contains_core_table_documents": 0,
    "report_core_tables_1_1_1_documents": 110,
    "residual_form_or_generic_tables": 0,
    "residual_consultant_noise_tables": 0,
    "refusal_rows_without_scope": 0,
    "unexpected_requirement_group_rows": 0,
    "old_measure_columns": 0,
}

EXPECTED_TABLE_TYPES = {
    "documents": 110,
    "identifiers": 110,
    "refusal_reasons": 110,
}

CORPUS_SUMMARY_SQL = """
select
    (select count(*) from public.service_registry) as service_registry_total,
    (select count(*) from public.service_registry where is_active = true) as active_services,
    (select count(*) from public.document_registry where status = 'active') as active_documents,
    (
        select count(*)
        from public.document_registry
        where status = 'active'
          and service_key is null
    ) as active_documents_without_service_key,
    (
        select count(*)
        from public.document_registry dr
        left join public.service_registry sr on sr.service_key = dr.service_key
        where dr.status = 'active'
          and dr.service_key is not null
          and sr.service_key is null
    ) as active_documents_with_broken_service_link,
    (
        select count(*)
        from public.service_registry sr
        left join public.document_registry dr
          on dr.service_key = sr.service_key
         and dr.status = 'active'
        where sr.is_active = true
          and dr.document_id is null
    ) as active_services_without_active_document,
    (
        select count(*)
        from public.service_registry
        where aliases_json is null
           or jsonb_array_length(aliases_json) = 0
    ) as services_without_aliases,
    (
        select count(*)
        from public.document_registry dr
        where dr.status = 'active'
          and dr.publication_payload_json #>> '{parser_payload_json,docx_preprocessing,mode}' = 'trim_for_rag'
    ) as trim_for_rag_documents,
    (
        select count(*)
        from public.document_registry dr
        where dr.status = 'active'
          and coalesce((dr.publication_payload_json #>> '{parser_payload_json,docx_preprocessing,applied_to_published_content}')::boolean, false) = true
    ) as preprocessing_applied_documents,
    (
        select count(*)
        from public.document_registry dr
        where dr.status = 'active'
          and coalesce((dr.publication_payload_json #>> '{parser_payload_json,docx_preprocessing,trim_safety_passed}')::boolean, false) = true
    ) as trim_safety_passed_documents,
    (
        select count(*)
        from public.document_registry dr
        where dr.status = 'active'
          and coalesce((dr.publication_payload_json #>> '{parser_payload_json,docx_preprocessing,report,tail_contains_core_table}')::boolean, true) = true
    ) as tail_contains_core_table_documents,
    (
        select count(*)
        from public.document_registry dr
        where dr.status = 'active'
          and coalesce((dr.publication_payload_json #>> '{parser_payload_json,docx_preprocessing,report,has_exactly_one_each_core_table}')::boolean, false) = true
    ) as report_core_tables_1_1_1_documents,
    (
        select count(*)
        from public.document_tables dt
        join public.document_registry dr on dr.document_id = dt.document_id
        where dr.status = 'active'
          and dt.table_type in ('form_fields', 'generic')
    ) as residual_form_or_generic_tables,
    (
        select count(*)
        from public.document_tables dt
        join public.document_registry dr on dr.document_id = dt.document_id
        where dr.status = 'active'
          and dt.table_type = 'consultant_noise'
    ) as residual_consultant_noise_tables,
    (
        select count(*)
        from public.document_registry dr
        join public.document_tables dt on dt.document_id = dr.document_id
        join public.document_table_rows dtr on dtr.table_id = dt.table_id
        where dr.status = 'active'
          and dt.table_type = 'refusal_reasons'
          and coalesce(
                nullif(dtr.normalized_row_json ->> 'row_scope', ''),
                nullif(dtr.metadata_json ->> 'row_scope', '')
              ) is null
    ) as refusal_rows_without_scope,
    (
        select count(*)
        from public.document_registry dr
        join public.document_tables dt on dt.document_id = dr.document_id
        join public.document_table_rows dtr on dtr.table_id = dt.table_id
        where dr.status = 'active'
          and dt.table_type = 'documents'
          and coalesce(
                nullif(dtr.normalized_row_json ->> 'requirement_group', ''),
                nullif(dtr.metadata_json ->> 'requirement_group', ''),
                '<empty>'
              ) not in ('required', 'optional')
    ) as unexpected_requirement_group_rows,
    (
        select count(*)
        from information_schema.columns
        where table_schema = 'public'
          and (
               column_name in ('primary_measure_code', 'measure_code')
            or table_name = 'measure_aliases'
          )
    ) as old_measure_columns;
"""

TABLE_TYPE_COUNTS_SQL = """
select
    coalesce(dt.table_type, '<null>') as table_type,
    count(*) as tables_count
from public.document_tables dt
join public.document_registry dr on dr.document_id = dt.document_id
where dr.status = 'active'
group by coalesce(dt.table_type, '<null>')
order by table_type;
"""

PROBLEM_SAMPLES_SQL = {
    "services_without_document": """
        select sr.service_key, sr.service_name_short, sr.cleaned_filename, sr.raw_filename
        from public.service_registry sr
        left join public.document_registry dr
          on dr.service_key = sr.service_key
         and dr.status = 'active'
        where sr.is_active = true
          and dr.document_id is null
        order by sr.service_key
        limit :limit;
    """,
    "documents_without_service_key": """
        select document_id, original_filename, service_name_short
        from public.document_registry
        where status = 'active'
          and service_key is null
        order by original_filename
        limit :limit;
    """,
    "residual_unwanted_tables": """
        select
            dr.service_key,
            dr.service_name_short,
            dr.original_filename,
            dt.table_type,
            dt.table_number,
            left(coalesce(dt.table_title, ''), 220) as table_title,
            dt.rows_count
        from public.document_tables dt
        join public.document_registry dr on dr.document_id = dt.document_id
        where dr.status = 'active'
          and dt.table_type in ('form_fields', 'generic', 'consultant_noise')
        order by dr.original_filename, dt.table_type, dt.table_number nulls last
        limit :limit;
    """,
    "refusal_rows_without_scope": """
        select
            dr.service_key,
            dr.service_name_short,
            dr.original_filename,
            count(*) as rows_without_scope
        from public.document_registry dr
        join public.document_tables dt on dt.document_id = dr.document_id
        join public.document_table_rows dtr on dtr.table_id = dt.table_id
        where dr.status = 'active'
          and dt.table_type = 'refusal_reasons'
          and coalesce(
                nullif(dtr.normalized_row_json ->> 'row_scope', ''),
                nullif(dtr.metadata_json ->> 'row_scope', '')
              ) is null
        group by dr.service_key, dr.service_name_short, dr.original_filename
        order by dr.original_filename
        limit :limit;
    """,
    "unexpected_requirement_groups": """
        select
            dr.service_key,
            dr.service_name_short,
            dr.original_filename,
            coalesce(
                nullif(dtr.normalized_row_json ->> 'requirement_group', ''),
                nullif(dtr.metadata_json ->> 'requirement_group', ''),
                '<empty>'
            ) as requirement_group,
            count(*) as rows_count
        from public.document_registry dr
        join public.document_tables dt on dt.document_id = dr.document_id
        join public.document_table_rows dtr on dtr.table_id = dt.table_id
        where dr.status = 'active'
          and dt.table_type = 'documents'
          and coalesce(
                nullif(dtr.normalized_row_json ->> 'requirement_group', ''),
                nullif(dtr.metadata_json ->> 'requirement_group', ''),
                '<empty>'
              ) not in ('required', 'optional')
        group by dr.service_key, dr.service_name_short, dr.original_filename, requirement_group
        order by dr.original_filename, requirement_group
        limit :limit;
    """,
    "old_measure_columns": """
        select table_name, column_name
        from information_schema.columns
        where table_schema = 'public'
          and (
               column_name in ('primary_measure_code', 'measure_code')
            or table_name = 'measure_aliases'
          )
        order by table_name, column_name
        limit :limit;
    """,
}


@dataclass(slots=True)
class CheckResult:
    name: str
    ok: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value") and isinstance(getattr(value, "value"), str):
        return value.value
    return str(value)


def _normalize_text(value: str) -> str:
    return " ".join((value or "").strip().lower().replace("ё", "е").split())


def _format_seconds(value: float) -> str:
    return f"{value:.2f} сек."


def _parse_intent(value: str) -> Any:
    from app.db.models.enums import QuestionIntentEnum

    normalized = (value or "").strip().lower()
    aliases = {
        "documents": QuestionIntentEnum.DOCUMENTS_QUESTION,
        "docs": QuestionIntentEnum.DOCUMENTS_QUESTION,
        "refusal": QuestionIntentEnum.REJECTION_QUESTION,
        "rejection": QuestionIntentEnum.REJECTION_QUESTION,
        "deadline": QuestionIntentEnum.DEADLINE_QUESTION,
        "eligibility": QuestionIntentEnum.ELIGIBILITY_QUESTION,
        "procedure": QuestionIntentEnum.PROCEDURE_QUESTION,
        "payment": QuestionIntentEnum.PAYMENT_TIMING_QUESTION,
        "amount": QuestionIntentEnum.AMOUNT_QUESTION,
        "appeal": QuestionIntentEnum.APPEAL_QUESTION,
        "other": QuestionIntentEnum.OTHER,
    }
    if normalized in aliases:
        return aliases[normalized]
    try:
        return QuestionIntentEnum(normalized)
    except ValueError as exc:
        supported = ", ".join(item.value for item in QuestionIntentEnum)
        raise ValueError(f"Неизвестный тип вопроса: {value}. Допустимые значения: {supported}") from exc


def _load_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл контрольных вопросов: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Файл контрольных вопросов должен содержать JSON-массив.")
    result: list[dict[str, Any]] = []
    for index, raw_case in enumerate(data, start=1):
        if not isinstance(raw_case, dict):
            raise ValueError(f"Контрольный вопрос №{index} должен быть объектом.")
        if not raw_case.get("id"):
            raise ValueError(f"У контрольного вопроса №{index} не заполнен id.")
        if not raw_case.get("question"):
            raise ValueError(f"У контрольного вопроса {raw_case.get('id')} не заполнен question.")
        if not raw_case.get("intent"):
            raise ValueError(f"У контрольного вопроса {raw_case.get('id')} не заполнен intent.")
        result.append(raw_case)
    return result


def _run_subprocess(command: list[str], *, cwd: Path) -> tuple[int, str, str]:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


def check_compileall() -> CheckResult:
    started = time.perf_counter()
    code, stdout, stderr = _run_subprocess(
        [sys.executable, "-m", "compileall", "-q", "app", "scripts"],
        cwd=ROOT,
    )
    elapsed = time.perf_counter() - started
    return CheckResult(
        name="Проверка синтаксиса",
        ok=code == 0,
        message="Синтаксис app и scripts в порядке." if code == 0 else "Есть ошибки синтаксиса.",
        details={"return_code": code, "stdout": stdout, "stderr": stderr},
        elapsed_seconds=elapsed,
    )


async def _fetch_one(manager: Any, sql: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    from sqlalchemy import text

    async with manager.session_scope() as session:
        result = await session.execute(text(sql), params or {})
        return dict(result.mappings().one())


async def _fetch_all(manager: Any, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    from sqlalchemy import text

    async with manager.session_scope() as session:
        result = await session.execute(text(sql), params or {})
        return [dict(row) for row in result.mappings().all()]


async def check_database_connection(manager: Any) -> CheckResult:
    started = time.perf_counter()
    try:
        await manager.check_connection()
        ok = True
        message = "Подключение к БД работает."
        details: dict[str, Any] = {}
    except Exception as exc:
        ok = False
        message = "Не удалось подключиться к БД."
        details = {"error": repr(exc)}
    return CheckResult(
        name="Подключение к БД",
        ok=ok,
        message=message,
        details=details,
        elapsed_seconds=time.perf_counter() - started,
    )


async def check_corpus(manager: Any, *, limit: int) -> CheckResult:
    started = time.perf_counter()
    problems: list[str] = []
    details: dict[str, Any] = {}

    try:
        summary = await _fetch_one(manager, CORPUS_SUMMARY_SQL)
        table_rows = await _fetch_all(manager, TABLE_TYPE_COUNTS_SQL)
        table_type_counts = {str(row["table_type"]): int(row["tables_count"]) for row in table_rows}

        details["summary"] = summary
        details["table_type_counts"] = table_type_counts

        for key, expected in EXPECTED_COUNTS.items():
            actual = summary.get(key)
            if actual != expected:
                problems.append(f"{key}: ожидалось {expected}, получено {actual}")

        if table_type_counts != EXPECTED_TABLE_TYPES:
            problems.append(
                "Состав table_type не совпадает с ожидаемым: "
                f"ожидалось {EXPECTED_TABLE_TYPES}, получено {table_type_counts}"
            )

        sample_sections: dict[str, list[dict[str, Any]]] = {}
        for section, sql in PROBLEM_SAMPLES_SQL.items():
            rows = await _fetch_all(manager, sql, {"limit": limit})
            if rows:
                sample_sections[section] = rows
        details["problem_samples"] = sample_sections

        ok = not problems
        message = "Корпус и привязка услуг в порядке." if ok else "Найдены проблемы в корпусе или привязке услуг."
        details["problems"] = problems
    except Exception as exc:
        ok = False
        message = "Проверка корпуса завершилась ошибкой."
        details = {"error": repr(exc)}

    return CheckResult(
        name="Корпус и привязка услуг",
        ok=ok,
        message=message,
        details=details,
        elapsed_seconds=time.perf_counter() - started,
    )


def _get_service_resolution(result: Any) -> dict[str, Any]:
    runtime_payload = getattr(result, "runtime_payload_json", None) or {}
    value = runtime_payload.get("service_resolution")
    return value if isinstance(value, dict) else {}


def _get_answer_mode(result: Any) -> str | None:
    generation_result = getattr(result, "generation_result", None)
    answer_mode = getattr(generation_result, "answer_mode", None)
    if answer_mode is None:
        return None
    return getattr(answer_mode, "value", str(answer_mode))


def _get_answer_text(result: Any) -> str:
    generation_result = getattr(result, "generation_result", None)
    return str(getattr(generation_result, "answer_text", "") or "")


def _get_selected_candidates_count(result: Any) -> int:
    evidence_package = getattr(result, "evidence_package", None)
    candidates = getattr(evidence_package, "selected_candidates", None) or []
    return len(candidates)


def _get_strategy_code(result: Any) -> str | None:
    evidence_package = getattr(result, "evidence_package", None)
    return getattr(evidence_package, "strategy_code", None)


def _check_case_expectations(case: dict[str, Any], case_result: dict[str, Any]) -> list[str]:
    problems: list[str] = []

    answer_mode = case_result.get("answer_mode")
    service_status = case_result.get("service_resolution_status")
    selected_candidates_count = int(case_result.get("selected_candidates_count") or 0)
    answer_text_normalized = _normalize_text(str(case_result.get("answer_text") or ""))

    expected_answer_modes = case.get("expected_answer_modes") or []
    if expected_answer_modes and answer_mode not in expected_answer_modes:
        problems.append(
            f"режим ответа: ожидалось одно из {expected_answer_modes}, получено {answer_mode}"
        )

    forbidden_answer_modes = case.get("forbidden_answer_modes") or []
    if forbidden_answer_modes and answer_mode in forbidden_answer_modes:
        problems.append(f"режим ответа {answer_mode} запрещён для этого вопроса")

    expected_statuses = case.get("expected_service_resolution_statuses") or []
    if expected_statuses and service_status not in expected_statuses:
        problems.append(
            f"статус услуги: ожидалось одно из {expected_statuses}, получено {service_status}"
        )

    min_selected = int(case.get("min_selected_candidates") or 0)
    if selected_candidates_count < min_selected:
        problems.append(
            f"источников выбрано меньше ожидаемого: минимум {min_selected}, получено {selected_candidates_count}"
        )

    required_any = [str(item).lower().replace("ё", "е") for item in (case.get("required_answer_terms_any") or [])]
    if required_any and not any(term in answer_text_normalized for term in required_any):
        problems.append(f"в ответе не найдено ни одного ожидаемого фрагмента: {required_any}")

    forbidden_terms = [str(item).lower().replace("ё", "е") for item in (case.get("forbidden_answer_terms_any") or [])]
    found_forbidden = [term for term in forbidden_terms if term in answer_text_normalized]
    if found_forbidden:
        problems.append(f"в ответе найдены запрещённые фрагменты: {found_forbidden}")

    if bool(case.get("expected_requires_service_discovery")):
        strategy_code = case_result.get("strategy_code")
        if service_status != "service_discovery":
            problems.append(
                "режим подбора мер не сработал: "
                f"ожидался статус service_discovery, получено {service_status}"
            )
        if strategy_code != "service_discovery":
            problems.append(
                "использована неверная стратегия: "
                f"ожидалась service_discovery, получено {strategy_code}"
            )
        if answer_mode == "safe_no_answer":
            problems.append("для вопроса под подбор мер получен safe_no_answer")

    return problems


async def check_intent_classifier(cases_path: Path) -> CheckResult:
    started = time.perf_counter()
    details: dict[str, Any] = {"cases_path": str(cases_path)}
    case_results: list[dict[str, Any]] = []
    problems: list[str] = []

    try:
        cases = _load_cases(cases_path)
        from app.services.answers.intent_classifier import RuleBasedIntentClassifier

        classifier = RuleBasedIntentClassifier()

        for case in cases:
            question = str(case["question"])
            classification = await classifier.classify(question)
            intent_value = classification.get("intent_type")
            actual_intent = getattr(intent_value, "value", str(intent_value))
            routing_payload = classification.get("routing_payload_json") or {}
            constraints = classification.get("query_constraints_json") or {}

            expected_auto_intents = case.get("expected_auto_intents") or [case.get("intent")]
            case_problems: list[str] = []

            if expected_auto_intents and actual_intent not in expected_auto_intents:
                case_problems.append(
                    f"тип вопроса: ожидалось одно из {expected_auto_intents}, получено {actual_intent}"
                )

            if case.get("expected_requires_service_discovery") is not None:
                expected_flag = bool(case.get("expected_requires_service_discovery"))
                actual_flag = bool(constraints.get("requires_service_discovery"))
                if actual_flag != expected_flag:
                    case_problems.append(
                        "признак подбора мер: "
                        f"ожидалось {expected_flag}, получено {actual_flag}"
                    )

            case_result = {
                "id": case["id"],
                "question": question,
                "expected_auto_intents": expected_auto_intents,
                "actual_intent": actual_intent,
                "confidence": routing_payload.get("confidence"),
                "matched_rules": routing_payload.get("matched_rules"),
                "query_constraints_json": constraints,
                "ok": not case_problems,
                "problems": case_problems,
            }
            case_results.append(case_result)
            if case_problems:
                problems.append(f"{case['id']}: " + "; ".join(case_problems))

        ok = not problems
        message = "Классификатор намерений работает ожидаемо." if ok else "Есть ошибки классификации намерений."
        details["cases"] = case_results
        details["problems"] = problems
    except Exception as exc:
        ok = False
        message = "Проверка классификатора намерений завершилась ошибкой."
        details["error"] = repr(exc)
        details["cases"] = case_results
        details["problems"] = problems

    return CheckResult(
        name="Классификатор намерений",
        ok=ok,
        message=message,
        details=details,
        elapsed_seconds=time.perf_counter() - started,
    )


async def _resolve_runtime_intent(
    *,
    mode: str,
    case: dict[str, Any],
    question: str,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    if mode == "from-cases":
        intent = _parse_intent(str(case["intent"]))
        routing_payload = dict(case.get("routing_payload_json") or {})
        query_constraints = dict(case.get("query_constraints_json") or {})
        if bool(case.get("expected_requires_service_discovery")):
            query_constraints.setdefault("requires_service_discovery", True)
            query_constraints.setdefault("avoid_single_service_resolution", True)
            query_constraints.setdefault("routing_mode", "service_discovery")
        return intent, routing_payload, query_constraints

    from app.services.answers.intent_classifier import RuleBasedIntentClassifier

    classifier = RuleBasedIntentClassifier()
    classification = await classifier.classify(question)
    intent_value = classification.get("intent_type")
    intent = intent_value if hasattr(intent_value, "value") else _parse_intent(str(intent_value))
    return (
        intent,
        classification.get("routing_payload_json") or {},
        classification.get("query_constraints_json") or {},
    )


async def check_runtime_cases(cases_path: Path, *, mode: str = "from-cases") -> CheckResult:
    started = time.perf_counter()
    details: dict[str, Any] = {"cases_path": str(cases_path), "intent_source": mode}
    case_results: list[dict[str, Any]] = []
    problems: list[str] = []

    try:
        cases = _load_cases(cases_path)
        from app.config.settings import load_settings
        from app.runtime.app_runtime import AppRuntime, AppRuntimeConfig
        from app.services.answers.runtime_answer_service import RuntimeAnswerInput

        settings = load_settings()
        runtime = AppRuntime(config=AppRuntimeConfig(database=settings.database))
        await runtime.startup()

        try:
            for case in cases:
                question = str(case["question"])
                intent, routing_payload, query_constraints = await _resolve_runtime_intent(
                    mode=mode,
                    case=case,
                    question=question,
                )
                normalized_question = _normalize_text(question)
                one_started = time.perf_counter()

                async with runtime.session_scope() as session:
                    service_factory = runtime.build_service_factory(session)
                    service = service_factory.get_runtime_answer_service()
                    result = await service.build_answer(
                        RuntimeAnswerInput(
                            session_id=uuid4(),
                            question_event_id=uuid4(),
                            channel_code="QUALITY_GATE",
                            question_text_raw=question,
                            question_text_normalized=normalized_question,
                            language_code="ru",
                            intent_type=intent,
                            routing_payload_json=routing_payload,
                            query_constraints_json=query_constraints,
                        )
                    )

                service_resolution = _get_service_resolution(result)
                answer_text = _get_answer_text(result)
                case_result = {
                    "id": case["id"],
                    "question": question,
                    "intent_source": mode,
                    "intent": intent.value,
                    "classifier_confidence": routing_payload.get("confidence"),
                    "classifier_matched_rules": routing_payload.get("matched_rules"),
                    "query_constraints_json": query_constraints,
                    "answer_mode": _get_answer_mode(result),
                    "service_resolution_status": service_resolution.get("resolution_status"),
                    "service_key": service_resolution.get("service_key"),
                    "service_name_short": service_resolution.get("service_name_short"),
                    "selected_candidates_count": _get_selected_candidates_count(result),
                    "strategy_code": _get_strategy_code(result),
                    "elapsed_seconds": round(time.perf_counter() - one_started, 3),
                    "answer_preview": answer_text[:700],
                }
                case_problems = _check_case_expectations(case, {**case_result, "answer_text": answer_text})
                case_result["ok"] = not case_problems
                case_result["problems"] = case_problems
                case_results.append(case_result)

                if case_problems:
                    problems.append(f"{case['id']}: " + "; ".join(case_problems))
        finally:
            await runtime.shutdown()

        ok = not problems
        if mode == "auto":
            name = "Контрольные вопросы с автоматическим типом"
            message = "Контрольные вопросы прошли без ручного указания типа." if ok else "Есть регрессия при автоматическом определении типа."
        else:
            name = "Контрольные вопросы с заданным типом"
            message = "Контрольные вопросы прошли без регрессии." if ok else "Есть регрессия в контрольных вопросах."

        details["cases"] = case_results
        details["problems"] = problems
    except Exception as exc:
        ok = False
        name = "Контрольные вопросы с автоматическим типом" if mode == "auto" else "Контрольные вопросы с заданным типом"
        message = "Проверка контрольных вопросов завершилась ошибкой."
        details["error"] = repr(exc)
        details["cases"] = case_results
        details["problems"] = problems

    return CheckResult(
        name=name,
        ok=ok,
        message=message,
        details=details,
        elapsed_seconds=time.perf_counter() - started,
    )


def _render_text_report(results: list[CheckResult], *, total_elapsed: float) -> None:
    print("=" * 100)
    print("ЕДИНАЯ ПРОВЕРКА КАЧЕСТВА RAG2")
    print("=" * 100)

    for result in results:
        marker = "OK" if result.ok else "ОШИБКА"
        print(f"\n[{marker}] {result.name} — {result.message} ({_format_seconds(result.elapsed_seconds)})")

        if result.name == "Корпус и привязка услуг":
            summary = result.details.get("summary") or {}
            table_counts = result.details.get("table_type_counts") or {}
            print("  Основные числа:")
            for key in EXPECTED_COUNTS:
                if key in summary:
                    print(f"    {key}: {summary[key]}")
            print(f"  table_type: {table_counts}")
            problems = result.details.get("problems") or []
            for problem in problems[:20]:
                print(f"  - {problem}")

        if result.name == "Классификатор намерений":
            for case in result.details.get("cases") or []:
                case_marker = "OK" if case.get("ok") else "ОШИБКА"
                print(
                    f"  [{case_marker}] {case.get('id')} | "
                    f"тип={case.get('actual_intent')} | "
                    f"уверенность={case.get('confidence')} | "
                    f"правила={case.get('matched_rules')}"
                )
                for problem in case.get("problems") or []:
                    print(f"      - {problem}")

        if result.name.startswith("Контрольные вопросы"):
            for case in result.details.get("cases") or []:
                case_marker = "OK" if case.get("ok") else "ОШИБКА"
                print(
                    f"  [{case_marker}] {case.get('id')} | "
                    f"режим={case.get('answer_mode')} | "
                    f"услуга={case.get('service_resolution_status')}:{case.get('service_key')} | "
                    f"источников={case.get('selected_candidates_count')} | "
                    f"{case.get('elapsed_seconds')} сек."
                )
                for problem in case.get("problems") or []:
                    print(f"      - {problem}")

        if not result.ok and result.details.get("error"):
            print(f"  Ошибка: {result.details['error']}")

    ok = all(result.ok for result in results)
    print("\n" + "=" * 100)
    print(f"ИТОГ: {'ПРОЙДЕНО' if ok else 'ЕСТЬ ПРОБЛЕМЫ'}")
    print(f"Общее время: {_format_seconds(total_elapsed)}")
    print("=" * 100)


def _write_json_report(path: Path, results: list[CheckResult], *, total_elapsed: float) -> None:
    report = {
        "ok": all(result.ok for result in results),
        "total_elapsed_seconds": round(total_elapsed, 3),
        "checks": [
            {
                "name": result.name,
                "ok": result.ok,
                "message": result.message,
                "elapsed_seconds": round(result.elapsed_seconds, 3),
                "details": _jsonable(result.details),
            }
            for result in results
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


async def run(args: argparse.Namespace) -> int:
    total_started = time.perf_counter()
    results: list[CheckResult] = []

    if not args.skip_compile:
        results.append(check_compileall())

    if not args.skip_db:
        from app.config.settings import load_settings
        from app.db.session import DatabaseSessionManager

        settings = load_settings()
        manager = DatabaseSessionManager(settings.database)
        manager.initialize()

        try:
            db_result = await check_database_connection(manager)
            results.append(db_result)
            if db_result.ok:
                results.append(await check_corpus(manager, limit=args.limit))
        finally:
            await manager.dispose()

    cases_path = Path(args.cases).expanduser().resolve()

    if not args.skip_intent_classifier:
        results.append(await check_intent_classifier(cases_path))

    if not args.skip_runtime:
        if args.runtime_intents in {"from-cases", "both"}:
            results.append(await check_runtime_cases(cases_path, mode="from-cases"))
        if args.runtime_intents in {"auto", "both"}:
            results.append(await check_runtime_cases(cases_path, mode="auto"))

    total_elapsed = time.perf_counter() - total_started
    if args.json_report:
        _write_json_report(Path(args.json_report).expanduser().resolve(), results, total_elapsed=total_elapsed)

    if not args.quiet:
        _render_text_report(results, total_elapsed=total_elapsed)
        if args.json_report:
            print(f"\nJSON-отчёт сохранён: {Path(args.json_report).expanduser().resolve()}")

    return 0 if all(result.ok for result in results) else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Единая проверка качества RAG2 после загрузки корпуса и перед живым использованием."
    )
    parser.add_argument(
        "--cases",
        default=str(ROOT / "scripts" / "temp" / "runtime_regression_cases.json"),
        help="Путь к JSON-файлу с контрольными вопросами.",
    )
    parser.add_argument("--json-report", help="Куда сохранить подробный JSON-отчёт.")
    parser.add_argument("--limit", type=int, default=30, help="Сколько проблемных строк показывать в выборках.")
    parser.add_argument("--skip-compile", action="store_true", help="Не проверять синтаксис.")
    parser.add_argument("--skip-db", action="store_true", help="Не проверять БД и корпус.")
    parser.add_argument("--skip-intent-classifier", action="store_true", help="Не проверять классификатор намерений.")
    parser.add_argument("--skip-runtime", action="store_true", help="Не прогонять контрольные вопросы.")
    parser.add_argument(
        "--runtime-intents",
        choices=["from-cases", "auto", "both"],
        default="both",
        help=(
            "Как задавать тип вопроса при проверке ответов: "
            "from-cases — из файла вопросов; auto — через классификатор; both — оба варианта."
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="Не печатать человекочитаемый отчёт.")
    args = parser.parse_args()

    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
