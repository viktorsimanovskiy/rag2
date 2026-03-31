from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config.settings import load_settings
from app.db.session import DatabaseSessionManager
from app.services.ingestion.basic_document_semantic_enricher import (
    BasicDocumentSemanticEnricher,
)
from app.services.ingestion.docx_structure_extractor import DocxStructureExtractor
from app.services.ingestion.docx_text_normalizer import DocxTextNormalizer
from app.services.ingestion.document_ingestion_pipeline import (
    DocumentIngestionInput,
    DocumentIngestionPipeline,
)
from app.services.ingestion.document_publisher import DocumentPublisher
from app.services.ingestion.structural_qc_service import StructuralQcService


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_name(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^\w\-\.]+", "_", value, flags=re.UNICODE)
    value = re.sub(r"_+", "_", value).strip("._")
    return value or "file"


def collect_docx_files(input_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".docx"]
    else:
        files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".docx"]
    return sorted(files, key=lambda p: p.name.lower())


async def ingest_one(
    *,
    file_path: Path,
    source_type: str,
    uploaded_by: str | None,
    manager: DatabaseSessionManager,
) -> dict:
    started_at = utcnow_iso()

    try:
        async with manager.session_scope() as session:
            pipeline = DocumentIngestionPipeline(
                session,
                normalizer=DocxTextNormalizer(),
                extractor=DocxStructureExtractor(),
                enricher=BasicDocumentSemanticEnricher(),
                qc=StructuralQcService(),
                publisher=DocumentPublisher(session),
            )

            result = await pipeline.ingest_document(
                DocumentIngestionInput(
                    file_path=str(file_path),
                    original_filename=file_path.name,
                    source_type=source_type,
                    uploaded_by=uploaded_by,
                    metadata_json={
                        "run_mode": "manual_batch_docx_dir",
                        "source_format": "docx",
                        "runner": "scripts/batch_ingest_docx_dir.py",
                    },
                )
            )

            status = str(result.status)
            ok = status.lower() not in {"failed", "error"}

            return {
                "ok": ok,
                "file_path": str(file_path),
                "original_filename": file_path.name,
                "started_at": started_at,
                "finished_at": utcnow_iso(),
                "ingestion_job_id": str(result.ingestion_job_id),
                "document_id": str(result.document_id) if result.document_id else None,
                "status": status,
                "file_hash": result.file_hash,
                "content_hash": result.content_hash,
                "warnings": result.warnings,
                "payload_json": result.payload_json,
            }

    except Exception as exc:
        return {
            "ok": False,
            "file_path": str(file_path),
            "original_filename": file_path.name,
            "started_at": started_at,
            "finished_at": utcnow_iso(),
            "status": "exception",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


async def run(
    *,
    input_dir: str,
    logs_dir: str,
    source_type: str,
    uploaded_by: str | None,
    recursive: bool,
) -> int:
    input_path = Path(input_dir).expanduser().resolve()
    logs_path = Path(logs_dir).expanduser().resolve()

    if not input_path.exists() or not input_path.is_dir():
        print(
            json.dumps(
                {
                    "ok": False,
                    "status": "invalid_input_dir",
                    "input_dir": str(input_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2

    logs_path.mkdir(parents=True, exist_ok=True)

    files = collect_docx_files(input_path, recursive=recursive)
    if not files:
        print(
            json.dumps(
                {
                    "ok": False,
                    "status": "no_docx_files",
                    "input_dir": str(input_path),
                    "recursive": recursive,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 3

    settings = load_settings()
    manager = DatabaseSessionManager(settings.database)
    manager.initialize()
    await manager.check_connection()

    batch_started_at = utcnow_iso()
    summary_items: list[dict] = []

    try:
        for idx, file_path in enumerate(files, start=1):
            result = await ingest_one(
                file_path=file_path,
                source_type=source_type,
                uploaded_by=uploaded_by,
                manager=manager,
            )

            log_filename = f"{idx:03d}_{sanitize_name(file_path.stem)}.json"
            log_path = logs_path / log_filename

            with log_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)

            summary_items.append(
                {
                    "index": idx,
                    "original_filename": file_path.name,
                    "file_path": str(file_path),
                    "log_path": str(log_path),
                    "ok": result.get("ok", False),
                    "status": result.get("status"),
                    "document_id": result.get("document_id"),
                    "ingestion_job_id": result.get("ingestion_job_id"),
                }
            )

            mark = "OK" if result.get("ok") else "FAILED"
            print(f"[{idx}/{len(files)}] {mark} -> {file_path.name} -> {log_path}")

        ok_count = sum(1 for item in summary_items if item["ok"])
        failed_count = len(summary_items) - ok_count

        summary = {
            "ok": failed_count == 0,
            "status": "completed",
            "input_dir": str(input_path),
            "logs_dir": str(logs_path),
            "recursive": recursive,
            "source_type": source_type,
            "uploaded_by": uploaded_by,
            "started_at": batch_started_at,
            "finished_at": utcnow_iso(),
            "total_files": len(summary_items),
            "ok_files": ok_count,
            "failed_files": failed_count,
            "files": summary_items,
        }

        summary_path = logs_path / "batch_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

        print()
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))

        return 0 if failed_count == 0 else 1

    finally:
        await manager.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch-ingest all .docx files from a directory and save one JSON log per file."
    )
    parser.add_argument("--input-dir", default="/home/docs", help="Directory with .docx files")
    parser.add_argument("--logs-dir", default="/home/logs", help="Directory for per-file JSON logs")
    parser.add_argument("--source-type", default="manual_batch", help="source_type for ingestion input")
    parser.add_argument("--uploaded-by", default=None, help="uploaded_by value for ingestion input")
    parser.add_argument("--recursive", action="store_true", help="Scan subdirectories recursively")
    args = parser.parse_args()

    return asyncio.run(
        run(
            input_dir=args.input_dir,
            logs_dir=args.logs_dir,
            source_type=args.source_type,
            uploaded_by=args.uploaded_by,
            recursive=args.recursive,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())