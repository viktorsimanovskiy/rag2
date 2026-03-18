from __future__ import annotations

import argparse
import asyncio
import json
import sys
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


async def run(file_path: str, source_type: str, uploaded_by: str | None) -> int:
    settings = load_settings()

    manager = DatabaseSessionManager(settings.database)
    manager.initialize()
    await manager.check_connection()

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
                    file_path=file_path,
                    original_filename=Path(file_path).name,
                    source_type=source_type,
                    uploaded_by=uploaded_by,
                    metadata_json={
                        "run_mode": "manual_docx_test",
                        "source_format": "docx",
                        "runner": "scripts/test_ingest_docx.py",
                    },
                )
            )

            print(
                json.dumps(
                    {
                        "ingestion_job_id": str(result.ingestion_job_id),
                        "document_id": str(result.document_id)
                        if result.document_id
                        else None,
                        "status": result.status,
                        "file_hash": result.file_hash,
                        "content_hash": result.content_hash,
                        "warnings": result.warnings,
                        "payload_json": result.payload_json,
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            )
        return 0
    finally:
        await manager.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run DOCX ingestion through the current ingestion pipeline."
    )
    parser.add_argument("file_path", help="Path to .docx file")
    parser.add_argument("--source-type", default="manual_test")
    parser.add_argument("--uploaded-by", default=None)
    args = parser.parse_args()

    return asyncio.run(run(args.file_path, args.source_type, args.uploaded_by))


if __name__ == "__main__":
    raise SystemExit(main())