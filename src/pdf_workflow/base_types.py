from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Generic, List, Type, TypeVar

from models import Citation

T = TypeVar("T")


@dataclass
class CitedEntity(Generic[T]):
    """Base class for any entity that includes citations."""

    citations: List[Citation]
    extracted_at: datetime = datetime.now(timezone.utc)
    source_document: str = ""

    def to_csv_row(self) -> dict:
        """Convert entity data to a CSV-friendly row format."""
        pages = ",".join(str(c.page_number) for c in self.citations)
        snippets = "; ".join(c.text_snippet for c in self.citations)
        avg_confidence = (
            sum(c.confidence_score for c in self.citations) / len(self.citations)
            if self.citations
            else 0
        )

        base_data = {
            "Source_Document": self.source_document,
            "Extracted_At": self.extracted_at.isoformat(),
            "Page_Numbers": pages,
            "Text_Snippets": snippets,
            "Average_Confidence": f"{avg_confidence:.2f}",
        }
        return base_data


@dataclass
class ExtractionTemplate:
    """Base template for extraction prompts."""

    entity_name: str
    fields: List[str]

    def get_prompt(self) -> str:
        fields_json = ", ".join(f'"{field}": "string"' for field in self.fields)
        return f"""
        Extract {self.entity_name} information and provide detailed citations.
        Format the response as a JSON object with the following structure:
        {{
            "{self.entity_name.lower()}s": [
                {{
                    {fields_json},
                    "citations": [
                        {{
                            "page_number": number,
                            "text_snippet": "string",
                            "confidence_score": number
                        }}
                    ]
                }}
            ]
        }}
        
        For each citation:
        - Include the page number where the information was found
        - Include a brief text snippet from the page (max 100 chars)
        - Provide a confidence score (0.0-1.0) for the citation
        
        Use empty string '' for missing values in any field.
        """
