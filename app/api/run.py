# ============================================================
# File: app/api/run.py
# Purpose:
#   Command-line runner for the RAG2 HTTP API.
# ============================================================

from __future__ import annotations

import argparse

import uvicorn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Запуск HTTP API RAG2.")
    parser.add_argument("--host", default="127.0.0.1", help="Адрес для запуска API.")
    parser.add_argument("--port", type=int, default=8000, help="Порт для запуска API.")
    parser.add_argument("--reload", action="store_true", help="Перезапускать сервер при изменении файлов.")
    parser.add_argument("--log-level", default="info", help="Уровень технического журнала uvicorn.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uvicorn.run(
        "app.api.http_app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
