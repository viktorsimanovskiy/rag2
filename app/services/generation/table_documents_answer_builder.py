from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(slots=True)
class DocumentsAnswerItem:
    document_name: str
    submission_note: Optional[str] = None


@dataclass(slots=True)
class DocumentsAnswerBuildResult:
    can_answer: bool
    items: list[DocumentsAnswerItem]
    reason: Optional[str] = None


class TableDocumentsAnswerBuilder:
    """
    Deterministic builder for document-list questions based on table rows.

    Works only with already selected retrieval candidates.
    Does not query DB directly.
    """

    def build(
        self,
        *,
        candidates: list[Any],
        submission_channel: Optional[str],
    ) -> DocumentsAnswerBuildResult:
        items: list[DocumentsAnswerItem] = []
        seen_names: set[str] = set()

        for candidate in candidates:
            if getattr(candidate, "source_type", None) != "table_row":
                continue

            metadata = getattr(candidate, "metadata_json", None) or {}
            if str(metadata.get("table_semantic_type") or "").strip().lower() != "documents":
                continue

            cells = metadata.get("cells_by_semantic_key") or {}
            if not isinstance(cells, dict):
                continue

            document_name = self._clean(cells.get("document_name"))
            if not document_name:
                continue

            if self._is_service_value(document_name):
                continue

            normalized_name = document_name.lower()
            if normalized_name in seen_names:
                continue
            seen_names.add(normalized_name)

            submission_note = None
            if submission_channel:
                submission_note = self._extract_submission_note(
                    cells=cells,
                    submission_channel=submission_channel,
                )

            items.append(
                DocumentsAnswerItem(
                    document_name=document_name,
                    submission_note=submission_note,
                )
            )

        if not items:
            return DocumentsAnswerBuildResult(
                can_answer=False,
                items=[],
                reason="no_documents_rows",
            )

        return DocumentsAnswerBuildResult(
            can_answer=True,
            items=items,
            reason=None,
        )

    def render_text(
        self,
        *,
        result: DocumentsAnswerBuildResult,
        submission_channel: Optional[str],
    ) -> Optional[str]:
        if not result.can_answer or not result.items:
            return None

        if not submission_channel:
            lines = ["Для предоставления услуги потребуются следующие документы:"]
            for idx, item in enumerate(result.items, start=1):
                lines.append(f"{idx}. {item.document_name}")
            return "\n".join(lines)

        channel_label = self._channel_label(submission_channel)
        lines = [f"Для предоставления услуги при подаче {channel_label} потребуются:"]

        for idx, item in enumerate(result.items, start=1):
            if item.submission_note:
                lines.append(f"{idx}. {item.document_name} — {item.submission_note}")
            else:
                lines.append(f"{idx}. {item.document_name}")

        return "\n".join(lines)

    def _extract_submission_note(
        self,
        *,
        cells: dict[str, Any],
        submission_channel: Optional[str],
    ) -> Optional[str]:
        if submission_channel == "epgu":
            return self._clean(cells.get("epgu_submission"))
        if submission_channel == "regional_portal":
            return self._clean(
                cells.get("regional_portal_submission") or cells.get("epgu_submission")
            )
        if submission_channel == "in_person":
            return self._clean(cells.get("in_person_submission"))
        if submission_channel == "post":
            return self._clean(cells.get("post_submission"))
        if submission_channel == "mfc":
            return self._clean(cells.get("mfc_submission"))
        return None

    def _is_service_value(self, value: str) -> bool:
        text = value.lower()
        markers = [
            "документы (информация), необходимые",
            "исчерпывающий перечень документов",
            "способ подачи в уполномоченное учреждение",
            "наименование документа",
        ]
        return any(marker in text for marker in markers)

    def _channel_label(self, submission_channel: str) -> str:
        mapping = {
            "epgu": "через ЕПГУ",
            "regional_portal": "через РПГУ / краевой портал",
            "in_person": "лично",
            "post": "почтовым отправлением",
            "mfc": "через МФЦ",
        }
        return mapping.get(submission_channel, "указанным способом")

    def _clean(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = " ".join(str(value).strip().split())
        return text or None