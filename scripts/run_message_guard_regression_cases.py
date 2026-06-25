#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Регрессионный прогон входного MessageGuard через HTTP API RAG2.

Назначение:
- проверить, что служебные, пустые по смыслу и подозрительные сообщения
  не уходят в retrieval по НПА;
- проверить, что обычный предметный вопрос по социальной поддержке не
  блокируется guard'ом.

Запуск:
    python scripts/run_message_guard_regression_cases.py \
      --url http://127.0.0.1:8000/api/v1/answer \
      --out-dir /home/logs/second_step_40/message_guard_$(date +%Y%m%d_%H%M%S) \
      --debug

Скрипт использует только стандартную библиотеку Python.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

VERSION = "second_step_40_message_guard_regression_v1"


@dataclass(frozen=True)
class Case:
    case_id: str
    question_text: str
    expected_blocked: bool
    expected_reason_code: str = ""
    diagnostic_note: str = ""


CASES: List[Case] = [
    Case(
        case_id="technical_ping",
        question_text="проверка связи",
        expected_blocked=True,
        expected_reason_code="technical_ping",
        diagnostic_note="Техническая проверка связи не должна уходить в retrieval.",
    ),
    Case(
        case_id="start_command",
        question_text="/start",
        expected_blocked=True,
        expected_reason_code="start_command",
        diagnostic_note="Команда start должна давать служебный ответ.",
    ),
    Case(
        case_id="help_request",
        question_text="/help",
        expected_blocked=True,
        expected_reason_code="help_request",
        diagnostic_note="Команда help должна давать служебный ответ.",
    ),
    Case(
        case_id="greeting",
        question_text="привет",
        expected_blocked=True,
        expected_reason_code="greeting_without_domain_question",
        diagnostic_note="Одно приветствие не должно запускать поиск по НПА.",
    ),
    Case(
        case_id="thanks",
        question_text="спасибо",
        expected_blocked=True,
        expected_reason_code="thanks_or_ack_without_domain_question",
        diagnostic_note="Благодарность не должна запускать поиск по НПА.",
    ),
    Case(
        case_id="suspicious_sql",
        question_text="SELECT * FROM users WHERE id = 1",
        expected_blocked=True,
        expected_reason_code="suspicious_code_or_injection_pattern",
        diagnostic_note="SQL-подобная строка не должна попадать в retrieval.",
    ),
    Case(
        case_id="url_only",
        question_text="https://example.com/test",
        expected_blocked=True,
        expected_reason_code="url_only_message",
        diagnostic_note="Одна ссылка без вопроса не должна попадать в retrieval.",
    ),
    Case(
        case_id="garbage_symbols",
        question_text="!!! ??? ###",
        expected_blocked=True,
        expected_reason_code="mostly_symbols_or_unreadable",
        diagnostic_note="Нечитаемый набор символов должен останавливаться до поиска.",
    ),
    Case(
        case_id="domain_question_allowed",
        question_text="какие документы нужны для субсидии на оплату ЖКУ?",
        expected_blocked=False,
        diagnostic_note="Предметный вопрос по мере социальной поддержки должен проходить в RAG.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MessageGuard regression cases against RAG2 API.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/api/v1/answer", help="RAG2 answer API URL.")
    parser.add_argument("--out-dir", required=True, help="Directory for JSON responses and summary files.")
    parser.add_argument("--debug", action="store_true", help="Pass debug=true to API. Recommended for this script.")
    parser.add_argument("--channel", default="test_console", help="Channel value for API request.")
    parser.add_argument("--external-user-id", default="message_guard_regression_user", help="External user id.")
    parser.add_argument("--external-chat-id", default="message_guard_regression_chat", help="External chat id.")
    parser.add_argument("--timeout", type=float, default=45.0, help="HTTP timeout in seconds.")
    return parser.parse_args()


def post_json(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=raw,
        method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            elapsed = time.perf_counter() - started
            parsed = json.loads(body) if body else None
            return {
                "ok": 200 <= int(response.status) < 300,
                "status_code": int(response.status),
                "elapsed_seconds": round(elapsed, 6),
                "response": parsed,
                "error": None,
            }
    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - started
        body = exc.read().decode("utf-8", errors="replace")
        parsed_error: Any
        try:
            parsed_error = json.loads(body)
        except Exception:
            parsed_error = body
        return {
            "ok": False,
            "status_code": int(exc.code),
            "elapsed_seconds": round(elapsed, 6),
            "response": parsed_error,
            "error": body,
        }
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        elapsed = time.perf_counter() - started
        return {
            "ok": False,
            "status_code": None,
            "elapsed_seconds": round(elapsed, 6),
            "response": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def get_nested(data: Any, path: Iterable[str], default: Any = None) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def extract_guard_payload(response: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(response, dict):
        return {}

    direct = get_nested(response, ["debug", "debug_payload_json", "message_guard"], {})
    if isinstance(direct, dict) and direct:
        return direct

    answer_payload = get_nested(response, ["debug", "answer_payload_json"], {})
    direct = answer_payload.get("message_guard") if isinstance(answer_payload, dict) else {}
    if isinstance(direct, dict) and direct:
        return direct

    routing_payload = get_nested(response, ["debug", "answer_payload_json", "runtime_answer_service", "debug_payload_json", "routing_payload_json"], {})
    direct = routing_payload.get("message_guard") if isinstance(routing_payload, dict) else {}
    if isinstance(direct, dict) and direct:
        return direct

    return {}


def detect_findings(case: Case, result: Dict[str, Any]) -> List[str]:
    findings: List[str] = []
    response = result.get("response") if isinstance(result.get("response"), dict) else {}

    if not result.get("ok"):
        findings.append("api_error")
        return findings

    guard = extract_guard_payload(response)
    answer_payload = get_nested(response, ["debug", "answer_payload_json"], {})
    strategy_code = answer_payload.get("strategy_code") if isinstance(answer_payload, dict) else ""
    guard_blocked = bool(
        strategy_code == "message_guard_no_retrieval"
        or (guard and guard.get("should_run_rag") is False)
    )
    reason_code = str(guard.get("reason_code") or "") if isinstance(guard, dict) else ""

    if case.expected_blocked and not guard_blocked:
        findings.append("message_was_not_blocked")
    if not case.expected_blocked and guard_blocked:
        findings.append("domain_question_was_blocked")
    if case.expected_reason_code and reason_code != case.expected_reason_code:
        findings.append(f"unexpected_reason_code:{reason_code or 'missing'}")

    if case.expected_blocked:
        citations = response.get("citations") if isinstance(response, dict) else []
        service_resolution = response.get("service_resolution") if isinstance(response, dict) else {}
        if citations:
            findings.append("blocked_message_has_citations")
        if isinstance(service_resolution, dict) and service_resolution.get("candidates"):
            findings.append("blocked_message_has_service_candidates")

    return findings


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for case in CASES:
        payload = {
            "question_text": case.question_text,
            "channel": args.channel,
            "external_user_id": args.external_user_id,
            "external_chat_id": args.external_chat_id,
            "debug": bool(args.debug),
        }
        result = post_json(args.url, payload, timeout=args.timeout)
        full = {
            "version": VERSION,
            "case_id": case.case_id,
            "expected_blocked": case.expected_blocked,
            "expected_reason_code": case.expected_reason_code,
            "diagnostic_note": case.diagnostic_note,
            "request": payload,
            **result,
        }
        (out_dir / f"{case.case_id}.json").write_text(
            json.dumps(full, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        response = result.get("response") if isinstance(result.get("response"), dict) else {}
        guard = extract_guard_payload(response)
        answer_payload = get_nested(response, ["debug", "answer_payload_json"], {})
        strategy_code = answer_payload.get("strategy_code") if isinstance(answer_payload, dict) else ""
        guard_blocked = bool(
            strategy_code == "message_guard_no_retrieval"
            or (guard and guard.get("should_run_rag") is False)
        )
        findings = detect_findings(case, result)
        rows.append(
            {
                "case_id": case.case_id,
                "ok": result.get("ok"),
                "status_code": result.get("status_code"),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "expected_blocked": case.expected_blocked,
                "guard_blocked": guard_blocked,
                "reason_code": str(guard.get("reason_code") or "") if isinstance(guard, dict) else "",
                "message_kind": str(guard.get("message_kind") or "") if isinstance(guard, dict) else "",
                "answer_mode": response.get("answer_mode") if isinstance(response, dict) else "",
                "strategy_code": strategy_code,
                "findings": ";".join(findings),
                "answer_start": str(response.get("answer_text") or "").replace("\n", " ")[:300] if isinstance(response, dict) else "",
            }
        )

    with (out_dir / "summary.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# MessageGuard regression findings", "", f"version: `{VERSION}`", ""]
    for row in rows:
        lines.append(f"## {row['case_id']}")
        lines.append(f"- ok: {row['ok']}")
        lines.append(f"- expected_blocked: {row['expected_blocked']}")
        lines.append(f"- guard_blocked: {row['guard_blocked']}")
        lines.append(f"- reason_code: {row['reason_code']}")
        lines.append(f"- answer_mode: {row['answer_mode']}")
        lines.append(f"- findings: {row['findings'] or 'нет автоматических флагов'}")
        lines.append("")
    (out_dir / "findings.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved {len(rows)} cases to {out_dir}")
    failed = [r for r in rows if r["findings"]]
    print(f"Cases with findings: {len(failed)}")
    for row in rows:
        print(
            f"{row['case_id']}: blocked={row['guard_blocked']} "
            f"reason={row['reason_code'] or '-'} findings={row['findings'] or '-'}"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
