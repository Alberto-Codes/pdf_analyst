from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import List


@dataclass
class Citation:
    """Represents a citation from the PDF document."""

    page_number: int
    text_snippet: str
    confidence_score: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Officer:
    """Represents an officer with their details and citations."""

    name: str
    age: str
    title: str
    citations: List[Citation]
    extracted_at: datetime = datetime.now(timezone.utc)
    source_document: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "age": self.age,
            "title": self.title,
            "citations": [citation.to_dict() for citation in self.citations],
            "extracted_at": self.extracted_at.isoformat(),
            "source_document": self.source_document,
        }

    def to_csv_row(self) -> dict:
        """Convert officer data to a CSV-friendly row format."""
        # Get page numbers as comma-separated string
        pages = ",".join(str(c.page_number) for c in self.citations)

        # Get text snippets as semicolon-separated string
        snippets = "; ".join(c.text_snippet for c in self.citations)

        # Get average confidence score
        avg_confidence = (
            sum(c.confidence_score for c in self.citations) / len(self.citations)
            if self.citations
            else 0
        )

        return {
            "Name": self.name,
            "Age": self.age,
            "Title": self.title,
            "Source_Document": self.source_document,
            "Extracted_At": self.extracted_at.isoformat(),
            "Page_Numbers": pages,
            "Text_Snippets": snippets,
            "Average_Confidence": f"{avg_confidence:.2f}",
        }


@dataclass
class ExtractionResult:
    """Represents the complete extraction result."""

    officers: List[Officer]
    raw_response: str
    extraction_timestamp: datetime = datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        return {
            "officers": [officer.to_dict() for officer in self.officers],
            "raw_response": self.raw_response,
            "extraction_timestamp": self.extraction_timestamp.isoformat(),
        }

    def export_to_csv(self, output_path: str) -> None:
        """Export officers data to a CSV file, one row per officer."""
        if not self.officers:
            raise ValueError("No officers data to export")

        # Get CSV-friendly rows
        rows = [officer.to_csv_row() for officer in self.officers]

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
