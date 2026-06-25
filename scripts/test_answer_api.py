# ============================================================
# File: scripts/test_answer_api.py
# Purpose:
#   Smoke test for the running RAG2 HTTP API.
#
# Usage:
#   python scripts/test_answer_api.py \
#     --url http://127.0.0.1:8000/api/v1/answer \
#     --json-report /home/logs/step33_http_api/api_smoke.json
# ============================================================

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_QUESTION = "я ветеран труда края какие документы нужны для получения едв"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Проверка работающего HTTP API RAG2.")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/api/v1/answer",
        help="Адрес endpoint /api/v1/answer.",
    )
    parser.add_argument(
        "--question",
        default=DEFAULT_QUESTION,
        help="Вопрос для проверки.",
    )
    parser.add_argument(
        "--channel",
        default="test_console",
        help="Канал запроса.",
    )
    parser.add_argument(
        "--external-user-id",
        default="api_smoke_user",
        help="Внешний id пользователя.",
    )
    parser.add_argument(
        "--external-chat-id",
        default="api_smoke_chat",
        help="Внешний id чата.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="Таймаут HTTP-запроса.",
    )
    parser.add_argument(
        "--json-report",
        default="",
        help="Куда сохранить JSON-отчёт.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()

    request_payload = {
        "question_text": args.question,
        "channel": args.channel,
        "external_user_id": args.external_user_id,
        "external_chat_id": args.external_chat_id,
        "debug": True,
    }

    report: dict[str, Any] = {
        "ok": False,
        "url": args.url,
        "request": request_payload,
        "elapsed_seconds": None,
        "status_code": None,
        "response": None,
        "error": None,
    }

    try:
        raw_body = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            args.url,
            data=raw_body,
            method="POST",
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
            },
        )
        with urlopen(request, timeout=args.timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
            response_json = json.loads(response_body)
            report["status_code"] = int(response.status)
            report["response"] = response_json
            report["ok"] = bool(response_json.get("ok")) and bool(response_json.get("answer_text"))

    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        report["status_code"] = int(exc.code)
        report["error"] = body
    except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        report["error"] = repr(exc)

    report["elapsed_seconds"] = round(time.perf_counter() - started, 6)

    if args.json_report:
        path = Path(args.json_report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
