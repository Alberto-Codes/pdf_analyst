from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, List

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Represents a citation from the PDF document."""

    page_number: int
    text_snippet: str
    confidence_score: float


class ExtractionResult(BaseModel):
    """Represents the complete extraction result."""

    entities: List[Any] = Field(default_factory=list)
    raw_response: str
    extraction_timestamp: datetime = Field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "officers": [officer.to_dict() for officer in self.entities],
            "raw_response": self.raw_response,
            "extraction_timestamp": self.extraction_timestamp.isoformat(),
        }

    def export_to_csv(self, output_path: str) -> None:
        """Export officers data to a CSV file, one row per officer."""
        if not self.entities:
            raise ValueError("No officers data to export")

        # Get CSV-friendly rows
        rows = [officer.to_csv_row() for officer in self.entities]

        # Write to CSV
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "Name",
                "Age",
                "Title",
                "Source_Document",
                "Extracted_At",
                "Page_Numbers",
                "Text_Snippets",
                "Average_Confidence",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(rows)
