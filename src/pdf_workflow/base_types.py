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

    # Class variable defining how fields map to CSV columns
    csv_field_mapping = {
        "source_document": "Source_Document",
        "extracted_at": "Extracted_At",
        "citations": ["Page_Numbers", "Text_Snippets", "Average_Confidence"],
    }

    @classmethod
    def get_csv_fields(cls) -> List[str]:
        """Get all CSV field names in the correct order."""
        fields = []
        for field_mapping in cls.csv_field_mapping.values():
            if isinstance(field_mapping, list):
                fields.extend(field_mapping)
            else:
                fields.append(field_mapping)
        return fields

    def to_csv_row(self) -> dict:
        """Convert entity data to a CSV-friendly row format."""
        pages = ",".join(str(c.page_number) for c in self.citations)
        snippets = "; ".join(c.text_snippet for c in self.citations)
        avg_confidence = (
            sum(c.confidence_score for c in self.citations) / len(self.citations)
            if self.citations
            else 0
        )

        # Map the raw fields to CSV fields using the mapping
        return {
            "Source_Document": self.source_document,
            "Extracted_At": self.extracted_at.isoformat(),
            "Page_Numbers": pages,
            "Text_Snippets": snippets,
            "Average_Confidence": f"{avg_confidence:.2f}",
        }


@dataclass
class ExtractionTemplate:
    """Base template for extraction prompts."""

    entity_name: str
    entity_type: Type[CitedEntity]  # Add this line to specify the entity class type
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
