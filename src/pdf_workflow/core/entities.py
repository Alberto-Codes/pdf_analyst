from datetime import datetime, timezone
from typing import Dict, List

from core.models import Citation
from pydantic import BaseModel, Field


class CitedEntity(BaseModel):
    """Base class for any entity that includes citations."""

    citations: List[Citation]
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_document: str = ""

    def to_csv_row(self) -> dict:
        """Convert entity data to a CSV-friendly row format using object attributes."""
        pages = ",".join(str(c.page_number) for c in self.citations)
        snippets = "; ".join(c.text_snippet for c in self.citations)
        avg_confidence = (
            sum(c.confidence_score for c in self.citations) / len(self.citations)
            if self.citations
            else 0
        )

        row = {
            "Source_Document": self.source_document,
            "Extracted_At": self.extracted_at.isoformat(),
            "Page_Numbers": pages,
            "Text_Snippets": snippets,
            "Average_Confidence": f"{avg_confidence:.2f}",
        }

        for field_name, value in self.__dict__.items():
            if field_name in {"citations", "extracted_at", "source_document"}:
                continue
            row[field_name] = value

        return row
