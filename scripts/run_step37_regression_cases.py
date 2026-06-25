#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Регрессионный прогон пяти сценариев из second_step_37.zip через HTTP API RAG2.

Назначение:
- быстро проверить, как текущая версия RAG2 отвечает на выявленные в step37 случаи;
- сохранить полные JSON-ответы;
- сформировать краткую сводку для анализа practical ambiguity / citations.

Запуск:
    python scripts/run_step37_regression_cases.py \
      --url http://127.0.0.1:8000/api/v1/answer \
      --out-dir /home/logs/step38/step37_regression_$(date +%Y%m%d_%H%M%S) \
      --debug

Скрипт не зависит от внутренних модулей проекта и использует только стандартную библиотеку Python.
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


@dataclass(frozen=True)
class Case:
    case_id: str
    question_text: str
    expected_focus: str
    diagnostic_note: str
    expected_top_service_keys: tuple[str, ...] = ()
    require_expected_top1: bool = True
    allow_ambiguity: bool = False


CASES: List[Case] = [
    Case(
        case_id="disabled_child_device",
        question_text="я родитель ребёнка-инвалида, могу ли подать заявление на помощь на коляску или слуховой аппарат ребёнку-инвалиду?",
        expected_focus="АМП на кресло-каталку и слуховой аппарат для детей-инвалидов",
        expected_top_service_keys=("svc_kreslo_kolyaska_invalidam_ot_31_03_2025_n_75_n_475b57fb5eda",),
        diagnostic_note="Профильная услуга по кресло-каталке/слуховому аппарату для ребёнка-инвалида должна быть top1 или явно первой в уточняющем ответе.",
    ),
    Case(
        case_id="emergency_material_help",
        question_text="я пострадавший при чрезвычайной ситуации, могу ли подать заявление на материальную помощь пострадавшему при ЧС?",
        expected_focus="Выплата материальной помощи пострадавшим при чрезвычайных ситуациях",
        expected_top_service_keys=("svc_vyplaty_postradavshim_v_chs_ot_26_12_2025_n_226_n_9204152fb4be",),
        diagnostic_note="Фраза 'материальная помощь при ЧС' должна поднимать профильную ЧС-услугу выше выплат за утрату имущества, вред здоровью и смерть.",
    ),
    Case(
        case_id="emergency_property_loss",
        question_text="я человек, потерявший имущество при ЧС, могу ли подать заявление на выплату за утраченное имущество после ЧС?",
        expected_focus="Получение выплаты при утрате имущества в результате чрезвычайной ситуации",
        expected_top_service_keys=("svc_vyplaty_pri_utrate_imuschestva_v_chs_ot_26_12_2025_n_227_n_74bcecae3134",),
        diagnostic_note="Фраза 'утрата имущества' должна поднимать профильную ЧС-услугу про имущество.",
    ),
    Case(
        case_id="jku_evenkia_compensation",
        question_text="я льготник в Эвенкии, могу ли подать заявление на компенсацию ЖКУ льготнику в Эвенкии?",
        expected_focus="Компенсация или МСП по оплате ЖКУ в Эвенкии",
        expected_top_service_keys=(
            "svc_subsidii_evenkiya_ot_04_06_2025_n_113_n_dfe67a8d157a",
            "svc_msp_na_zhku_evenkiya_ot_14_05_2025_n_102_n_1ccb2c9ed8e8",
        ),
        diagnostic_note="Кандидаты должны быть про ЖКУ/Эвенкию; чужие citations недопустимы.",
        allow_ambiguity=True,
    ),
    Case(
        case_id="jku_general_compensation",
        question_text="я льготник, могу ли подать заявление на компенсацию за жильё и коммуналку льготнику?",
        expected_focus="ЖКУ/коммунальные услуги без ремонта жилья в верхних кандидатах",
        diagnostic_note="Широкий вопрос про ЖКУ должен просить уточнение категории/территории; ремонт жилого помещения не должен попадать в верхние кандидаты.",
        require_expected_top1=False,
        allow_ambiguity=True,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run step37 regression cases against RAG2 API.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/api/v1/answer", help="RAG2 answer API URL.")
    parser.add_argument("--out-dir", required=True, help="Directory for JSON responses and summary files.")
    parser.add_argument("--debug", action="store_true", help="Pass debug=true to API.")
    parser.add_argument("--channel", default="test_console", help="Channel value for API request.")
    parser.add_argument("--external-user-id", default="step37_regression_user", help="External user id.")
    parser.add_argument("--external-chat-id", default="step37_regression_chat", help="External chat id.")
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
        return {
            "ok": False,
            "status_code": int(exc.code),
            "elapsed_seconds": round(elapsed, 6),
            "response": None,
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


def get_nested(data: Dict[str, Any], path: Iterable[str], default: Any = None) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def first_candidates(response: Optional[Dict[str, Any]], limit: int = 8) -> List[Dict[str, Any]]:
    if not isinstance(response, dict):
        return []
    candidates = get_nested(response, ["service_resolution", "candidates"], [])
    return candidates[:limit] if isinstance(candidates, list) else []


def citation_names(response: Optional[Dict[str, Any]], limit: int = 5) -> List[str]:
    if not isinstance(response, dict):
        return []
    citations = response.get("citations") or []
    names: List[str] = []
    for item in citations[:limit]:
        if isinstance(item, dict):
            names.append(str(item.get("document_name") or item.get("display_label") or ""))
    return names


def normalize_for_match(value: str) -> str:
    text = str(value or "").lower().replace("ё", "е")
    replacements = {
        "слухового": "слухов",
        "слуховой": "слухов",
        "аппарата": "аппарат",
        "аппарат": "аппарат",
        "детей": "дет",
        "ребенка": "ребен",
        "ребёнка": "ребен",
        "инвалидов": "инвалид",
        "инвалида": "инвалид",
        "чрезвычайных": "чрезвычайн",
        "чрезвычайной": "чрезвычайн",
        "ситуациях": "ситуац",
        "ситуации": "ситуац",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return " ".join(text.split())


def detect_findings(case: Case, result: Dict[str, Any]) -> List[str]:
    response = result.get("response")
    findings: List[str] = []
    if not result.get("ok"):
        findings.append("api_error")
        return findings
    if not isinstance(response, dict):
        findings.append("empty_response")
        return findings

    answer_mode = response.get("answer_mode")
    answer_text = str(response.get("answer_text") or "")
    candidates = first_candidates(response, limit=8)
    candidate_names = [str(c.get("service_name_short") or "") for c in candidates if isinstance(c, dict)]
    candidate_keys = [str(c.get("service_key") or "") for c in candidates if isinstance(c, dict)]
    top1 = candidate_names[0] if candidate_names else ""
    top1_key = candidate_keys[0] if candidate_keys else ""

    if case.expected_top_service_keys:
        if not any(key in case.expected_top_service_keys for key in candidate_keys):
            findings.append("expected_service_key_absent_from_top_candidates")
        elif case.require_expected_top1 and top1_key not in case.expected_top_service_keys:
            findings.append("expected_service_key_not_top1")
    elif case.expected_focus and case.require_expected_top1:
        normalized_expected = normalize_for_match(case.expected_focus)
        normalized_candidates = [normalize_for_match(name) for name in candidate_names]
        normalized_top1 = normalize_for_match(top1)
        if not any(normalized_expected in name or name in normalized_expected for name in normalized_candidates):
            findings.append("expected_focus_absent_from_top_candidates")
        elif normalized_expected not in normalized_top1 and normalized_top1 not in normalized_expected:
            findings.append("expected_focus_not_top1")

    if "несколько похожих мер" in answer_text and answer_mode == "grounded_narrative":
        findings.append("ambiguity_text_with_grounded_narrative_mode")

    if case.allow_ambiguity and "несколько похожих мер" in answer_text:
        # Уточняющий ответ допустим. Главное — чтобы он не нес случайные citations
        # и не поднимал явно чужие услуги.
        pass

    if case.case_id.startswith("jku"):
        if any("ремонт" in name.lower() for name in candidate_names[:5]):
            findings.append("repair_service_in_top_jku_candidates")
        joined_citations = "\n".join(citation_names(response)).lower()
        if "зубн" in joined_citations or "земельн" in joined_citations:
            findings.append("irrelevant_citation_for_jku")
        if case.case_id == "jku_general_compensation" and "несколько похожих мер" not in answer_text:
            findings.append("broad_jku_question_should_ask_clarification")

    if case.case_id == "disabled_child_device":
        joined_citations = "\n".join(citation_names(response)).lower()
        if "земельн" in joined_citations:
            findings.append("irrelevant_citation_for_disabled_child_device")

    if response.get("answer_mode") == "safe_no_answer" and case.case_id in {"emergency_property_loss", "emergency_material_help", "disabled_child_device"}:
        findings.append("safe_no_answer_on_specific_question")

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
            "case_id": case.case_id,
            "expected_focus": case.expected_focus,
            "diagnostic_note": case.diagnostic_note,
            "request": payload,
            **result,
        }
        (out_dir / f"{case.case_id}.json").write_text(json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8")

        response = result.get("response") if isinstance(result.get("response"), dict) else {}
        candidates = first_candidates(response, limit=8)
        findings = detect_findings(case, result)
        rows.append(
            {
                "case_id": case.case_id,
                "ok": result.get("ok"),
                "status_code": result.get("status_code"),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "answer_mode": response.get("answer_mode"),
                "expected_focus": case.expected_focus,
                "top1": (candidates[0].get("service_name_short") if candidates else ""),
                "top1_service_key": (candidates[0].get("service_key") if candidates else ""),
                "top_candidates": " | ".join(str(c.get("service_name_short") or "") for c in candidates[:5]),
                "findings": ";".join(findings),
                "answer_start": str(response.get("answer_text") or "").replace("\n", " ")[:400],
            }
        )

    with (out_dir / "summary.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    findings_lines = ["# Step37 regression findings", ""]
    for row in rows:
        findings_lines.append(f"## {row['case_id']}")
        findings_lines.append(f"- ok: {row['ok']}")
        findings_lines.append(f"- answer_mode: {row['answer_mode']}")
        findings_lines.append(f"- expected_focus: {row['expected_focus']}")
        findings_lines.append(f"- top1: {row['top1']}")
        findings_lines.append(f"- findings: {row['findings'] or 'нет автоматических флагов'}")
        findings_lines.append("")
    (out_dir / "findings.md").write_text("\n".join(findings_lines), encoding="utf-8")

    print(f"Saved {len(rows)} cases to {out_dir}")
    failed = [r for r in rows if r["findings"]]
    print(f"Cases with findings: {len(failed)}")
    for row in rows:
        print(f"{row['case_id']}: mode={row['answer_mode']} top1={row['top1']} findings={row['findings'] or '-'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
