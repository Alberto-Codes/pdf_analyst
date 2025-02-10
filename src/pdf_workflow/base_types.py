from datetime import datetime, timezone
from typing import Dict, List, Type

from models import Citation
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

    @classmethod
    def with_mapping(cls, field_mapping: Dict[str, str] = None) -> "CitedEntity":
        """Create a new instance with custom field mapping."""
        if field_mapping:
            return cls.construct(**field_mapping)  # ✅ Correct way to apply mapping
        return cls


class ExtractionTemplate(BaseModel):
    """Base template for extraction prompts."""

    entity_name: str
    entity_type: str  # ✅ Change from `Type[CitedEntity]` to `str`
    fields: List[str]
    is_singular: bool = False

    def model_post_init(self, __context):
        """Ensure field mapping and order are correctly initialized."""
        pass

    def get_prompt(self) -> str:
        """Generate appropriate prompt based on entity type."""
        fields_json = ", ".join(f'"{field}": "string"' for field in self.fields)
        entity_key = self.entity_name.lower()

        if self.is_singular:
            return f"""
            Extract {self.entity_name} information and provide detailed citation.
            Specifically, extract the total employee count and the corresponding year.
            
            Format the response as a JSON object with the following structure:
            {{
                "{entity_key}": {{
                    "count": "number",
                    "year": "string",
                    "citations": [
                        {{
                            "page_number": number,
                            "text_snippet": "string",
                            "confidence_score": number
                        }}
                    ]
                }}
            }}
            
            For the citation:
            - Include the page number where the information was found.
            - Include a brief text snippet from the page (max 100 chars).
            - Provide a confidence score (0.0-1.0) for the citation.
            
            Use empty string '' for missing values in any field.
            If the year is not explicitly mentioned, infer it from the surrounding context.
            """
        else:
            return f"""
            Extract {self.entity_name} information and provide detailed citations.
            Format the response as a JSON object with the following structure:
            {{
                "{entity_key}s": [
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
