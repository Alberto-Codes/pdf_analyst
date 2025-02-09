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

    # Base citation field mapping
    csv_field_mapping = {
        "source_document": "Source_Document",
        "extracted_at": "Extracted_At",
        "citations": ["Page_Numbers", "Text_Snippets", "Average_Confidence"],
    }

    def to_csv_row(self) -> dict:
        """Convert entity data to a CSV-friendly row format."""
        # Handle citation fields
        pages = ",".join(str(c.page_number) for c in self.citations)
        snippets = "; ".join(c.text_snippet for c in self.citations)
        avg_confidence = (
            sum(c.confidence_score for c in self.citations) / len(self.citations)
            if self.citations
            else 0
        )

        # Start with base fields
        row = {
            "Source_Document": self.source_document,
            "Extracted_At": self.extracted_at.isoformat(),
            "Page_Numbers": pages,
            "Text_Snippets": snippets,
            "Average_Confidence": f"{avg_confidence:.2f}",
        }

        # Add entity-specific fields using field mapping from instance
        if hasattr(self, "_field_mapping"):
            for field_name, csv_name in self._field_mapping.items():
                if hasattr(self, field_name):
                    row[csv_name] = getattr(self, field_name)

        return row

    @classmethod
    def with_mapping(cls, field_mapping: dict = None):
        """Create a new instance with custom field mapping."""
        if field_mapping:
            cls._field_mapping = field_mapping
        return cls


@dataclass
class ExtractionTemplate:
    """Base template for extraction prompts."""

    entity_name: str
    entity_type: Type[CitedEntity]
    fields: List[str]
    field_order: List[str] = None
    field_mapping: dict = None
    is_singular: bool = False  # New field to indicate single-value entities

    def __post_init__(self):
        """Initialize field order and mapping if not provided."""
        if self.field_mapping is None:
            self.field_mapping = {field: field.title() for field in self.fields}
            self.field_mapping.update(self.entity_type.csv_field_mapping)

        if self.field_order is None:
            base_fields = self.entity_type.get_csv_fields()
            entity_fields = [self.field_mapping[f] for f in self.fields]
            self.field_order = entity_fields + [
                f for f in base_fields if f not in entity_fields
            ]

    def get_prompt(self) -> str:
        """Generate appropriate prompt based on entity type."""
        fields_json = ", ".join(f'"{field}": "string"' for field in self.fields)

        if self.is_singular:
            return f"""
            Extract {self.entity_name} information and provide detailed citation.
            Look for mentions of total employee count, workforce size, or number of employees.
            Format the response as a JSON object with the following structure:
            {{
                "{self.entity_name.lower()}": {{
                    {fields_json},
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
            - Include the page number where the information was found
            - Include a brief text snippet from the page (max 100 chars)
            - Provide a confidence score (0.0-1.0) for the citation
            
            Use empty string '' for missing values in any field.
            """
        else:
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
