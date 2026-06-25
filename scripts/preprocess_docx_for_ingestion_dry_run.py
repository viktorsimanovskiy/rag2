from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.ingestion.docx_preprocessor import (  # noqa: E402
    ConsultantPlusDocxPreprocessor,
    DocxPreprocessingReport,
)


def _safe_extract_zip(zip_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = output_dir / member.filename
            resolved = target.resolve()
            if not str(resolved).startswith(str(output_dir.resolve())):
                raise ValueError(f"Unsafe zip member path: {member.filename}")
            if member.is_dir():
                resolved.mkdir(parents=True, exist_ok=True)
                continue
            resolved.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, resolved.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def _collect_docx_files(path: Path) -> list[Path]:
    return sorted(
        p for p in path.rglob("*.docx")
        if p.is_file() and not p.name.startswith("~$")
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _write_csv(path: Path, reports: list[DocxPreprocessingReport]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "official_start_found",
            "items_before_official_start_count",
            "core_identifiers",
            "core_documents",
            "core_refusal_reasons",
            "has_exactly_one_each_core_table",
            "trim_candidate",
            "kept_items_count",
            "kept_tables_count",
            "removed_before_official_start_count",
            "removed_consultant_noise_count",
            "trimmed_after_last_core_count",
            "trimmed_after_last_core_tables_count",
            "trimmed_after_last_core_chars_count",
            "trimmed_tail_table_type_counts_json",
            "trimmed_tail_category_counts_json",
            "warnings_json",
        ])
        for r in reports:
            writer.writerow([
                r.filename,
                r.official_start_found,
                r.items_before_official_start_count,
                r.core_table_counts.get("identifiers", 0),
                r.core_table_counts.get("documents", 0),
                r.core_table_counts.get("refusal_reasons", 0),
                r.has_exactly_one_each_core_table,
                r.trim_after_last_core_table_candidate,
                r.kept_items_count,
                r.kept_tables_count,
                r.removed_before_official_start_count,
                r.removed_consultant_noise_count,
                r.trimmed_after_last_core_count,
                r.trimmed_after_last_core_tables_count,
                r.trimmed_after_last_core_chars_count,
                json.dumps(r.trimmed_tail_table_type_counts, ensure_ascii=False),
                json.dumps(r.trimmed_tail_category_counts, ensure_ascii=False),
                json.dumps(r.warnings, ensure_ascii=False),
            ])


def _build_summary(reports: list[DocxPreprocessingReport], errors: list[dict[str, str]]) -> dict[str, object]:
    tail_table_types: Counter[str] = Counter()
    tail_categories: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    core_patterns: Counter[str] = Counter()
    for r in reports:
        tail_table_types.update(r.trimmed_tail_table_type_counts)
        tail_categories.update(r.trimmed_tail_category_counts)
        warning_counts.update(r.warnings)
        pattern = "/".join(str(r.core_table_counts.get(kind, 0)) for kind in ("identifiers", "documents", "refusal_reasons"))
        core_patterns[pattern] += 1

    return {
        "total_files": len(reports) + len(errors),
        "ok_files": len(reports),
        "error_files": len(errors),
        "official_start_found_files": sum(1 for r in reports if r.official_start_found),
        "exact_core_1_1_1_files": sum(1 for r in reports if r.has_exactly_one_each_core_table),
        "all_core_found_files": sum(1 for r in reports if r.has_all_core_tables),
        "trim_candidate_files": sum(1 for r in reports if r.trim_after_last_core_table_candidate),
        "tail_contains_core_table_files": sum(1 for r in reports if r.tail_contains_core_table),
        "removed_before_official_start_total": sum(r.removed_before_official_start_count for r in reports),
        "removed_consultant_noise_total": sum(r.removed_consultant_noise_count for r in reports),
        "trimmed_after_last_core_items_total": sum(r.trimmed_after_last_core_count for r in reports),
        "trimmed_after_last_core_tables_total": sum(r.trimmed_after_last_core_tables_count for r in reports),
        "trimmed_after_last_core_chars_total": sum(r.trimmed_after_last_core_chars_count for r in reports),
        "kept_items_total": sum(r.kept_items_count for r in reports),
        "kept_tables_total": sum(r.kept_tables_count for r in reports),
        "core_count_patterns": dict(sorted(core_patterns.items())),
        "tail_table_type_counts": dict(sorted(tail_table_types.items())),
        "tail_category_counts": dict(sorted(tail_categories.items())),
        "warning_counts": dict(sorted(warning_counts.items())),
    }


def _write_review(path: Path, summary: dict[str, object], reports: list[DocxPreprocessingReport], errors: list[dict[str, str]]) -> None:
    lines: list[str] = []
    lines.append("DRY-RUN ПРОГРАММНОЙ ПОДГОТОВКИ DOCX ДЛЯ INGESTION")
    lines.append("")
    lines.append("ВАЖНО: этот скрипт ничего не меняет в DOCX, ingestion и БД.")
    lines.append("Он моделирует будущий слой подготовки raw DOCX без Word-макроса.")
    lines.append("")
    lines.append("ИТОГ")
    for key in (
        "total_files",
        "ok_files",
        "error_files",
        "official_start_found_files",
        "exact_core_1_1_1_files",
        "all_core_found_files",
        "trim_candidate_files",
        "tail_contains_core_table_files",
        "removed_before_official_start_total",
        "removed_consultant_noise_total",
        "trimmed_after_last_core_items_total",
        "trimmed_after_last_core_tables_total",
        "trimmed_after_last_core_chars_total",
        "kept_items_total",
        "kept_tables_total",
    ):
        lines.append(f"- {key}: {summary.get(key)}")
    lines.append(f"- core_count_patterns: {json.dumps(summary.get('core_count_patterns'), ensure_ascii=False)}")
    lines.append(f"- tail_table_type_counts: {json.dumps(summary.get('tail_table_type_counts'), ensure_ascii=False)}")
    lines.append(f"- tail_category_counts: {json.dumps(summary.get('tail_category_counts'), ensure_ascii=False)}")
    lines.append(f"- warning_counts: {json.dumps(summary.get('warning_counts'), ensure_ascii=False)}")
    lines.append("")

    risky = [
        r for r in reports
        if r.warnings or not r.trim_after_last_core_table_candidate or not r.has_exactly_one_each_core_table
    ]
    if risky:
        lines.append("ФАЙЛЫ ДЛЯ РУЧНОЙ ПРОВЕРКИ")
        for r in risky[:80]:
            lines.append(f"- {r.filename}")
            lines.append(f"  warnings: {', '.join(r.warnings) or '-'}")
            lines.append(f"  core: {r.core_table_counts}, indexes: {r.core_table_indexes}")
            lines.append(f"  trim: items={r.trimmed_after_last_core_count}, tables={r.trimmed_after_last_core_tables_count}")
        if len(risky) > 80:
            lines.append(f"... ещё {len(risky) - 80} файлов")
        lines.append("")

    top_tail = sorted(reports, key=lambda r: r.trimmed_after_last_core_count, reverse=True)[:15]
    lines.append("ТОП-15 ХВОСТОВ ПОСЛЕ ПОСЛЕДНЕЙ КЛЮЧЕВОЙ ТАБЛИЦЫ")
    for r in top_tail:
        lines.append(
            f"- {r.filename}: tail_items={r.trimmed_after_last_core_count}, "
            f"tail_tables={r.trimmed_after_last_core_tables_count}, "
            f"tail_chars={r.trimmed_after_last_core_chars_count}, "
            f"last_core={r.last_core_table_index}/{r.last_core_table_type}"
        )
        for sample in r.first_trimmed_samples[:3]:
            lines.append(
                f"  * {sample.get('kind')} #{sample.get('table_index') or '-'} "
                f"{sample.get('category')}: {sample.get('text')}"
            )
    lines.append("")

    if errors:
        lines.append("ОШИБКИ")
        for e in errors:
            lines.append(f"- {e.get('filename')}: {e.get('error')}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _run(input_root: Path, output_dir: Path, *, write_prepared_text_samples: bool) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = _collect_docx_files(input_root)
    preprocessor = ConsultantPlusDocxPreprocessor()
    reports: list[DocxPreprocessingReport] = []
    errors: list[dict[str, str]] = []

    samples_dir = output_dir / "prepared_text_samples"
    if write_prepared_text_samples:
        samples_dir.mkdir(parents=True, exist_ok=True)

    for path in files:
        try:
            result = preprocessor.analyze(path)
            reports.append(result.report)
            if write_prepared_text_samples:
                safe_name = path.stem[:120].replace("/", "_")
                (samples_dir / f"{safe_name}.txt").write_text(result.prepared_text, encoding="utf-8")
        except Exception as exc:
            errors.append({"filename": path.name, "path": str(path), "error": repr(exc)})

    summary = _build_summary(reports, errors)
    _write_json(output_dir / "docx_preprocessor_dry_run_report.json", {
        "summary": summary,
        "reports": [r.to_dict() for r in reports],
        "errors": errors,
    })
    _write_csv(output_dir / "docx_preprocessor_dry_run_summary.csv", reports)
    _write_review(output_dir / "docx_preprocessor_dry_run_review.txt", summary, reports, errors)

    print("ГОТОВО")
    print(f"DOCX всего: {summary['total_files']}")
    print(f"ошибок: {summary['error_files']}")
    print(f"официальное начало найдено: {summary['official_start_found_files']}")
    print(f"ключевые таблицы 1/1/1: {summary['exact_core_1_1_1_files']}")
    print(f"кандидаты на trim: {summary['trim_candidate_files']}")
    print(f"хвост после последней ключевой таблицы: {summary['trimmed_after_last_core_items_total']} элементов, {summary['trimmed_after_last_core_tables_total']} таблиц")
    print(f"типы таблиц хвоста: {json.dumps(summary['tail_table_type_counts'], ensure_ascii=False)}")
    print(f"категории хвоста: {json.dumps(summary['tail_category_counts'], ensure_ascii=False)}")
    print(f"JSON: {output_dir / 'docx_preprocessor_dry_run_report.json'}")
    print(f"CSV:  {output_dir / 'docx_preprocessor_dry_run_summary.csv'}")
    print(f"TXT:  {output_dir / 'docx_preprocessor_dry_run_review.txt'}")
    return 0 if not errors else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Dry-run программной подготовки raw/cleaned DOCX для ingestion без изменения файлов и БД."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-dir", help="Папка с DOCX.")
    source.add_argument("--input-zip", help="ZIP с DOCX.")
    parser.add_argument("--output-dir", required=True, help="Куда сохранить JSON/CSV/TXT отчёты.")
    parser.add_argument(
        "--write-prepared-text-samples",
        action="store_true",
        help="Дополнительно сохранить подготовленный plain text по каждому DOCX для ручной проверки.",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).resolve()
    with tempfile.TemporaryDirectory(prefix="rag2_docx_preprocessor_dry_run_") as tmp:
        if args.input_zip:
            input_root = Path(tmp) / "input"
            _safe_extract_zip(Path(args.input_zip).resolve(), input_root)
        else:
            input_root = Path(args.input_dir).resolve()
        return _run(
            input_root=input_root,
            output_dir=output_dir,
            write_prepared_text_samples=bool(args.write_prepared_text_samples),
        )


if __name__ == "__main__":
    raise SystemExit(main())
