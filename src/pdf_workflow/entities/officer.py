from typing import List

from base_types import CitedEntity, ExtractionTemplate
from models import Citation
from pydantic import BaseModel


class Officer(BaseModel, CitedEntity):  # BaseModel MUST come first
    """Represents an officer with their details and citations."""

    name: str
    age: str
    title: str
    citations: List[Citation]

    @classmethod
    def get_csv_fields(cls):  # Explicitly add the missing method
        return [
            "Name",
            "Age",
            "Title",
            "Source_Document",
            "Extracted_At",
            "Page_Numbers",
            "Text_Snippets",
            "Average_Confidence",
        ]


OFFICER_TEMPLATE = ExtractionTemplate(
    entity_name="Officer",
    entity_type=Officer,
    fields=["name", "age", "title"],
    field_mapping={"name": "Name", "age": "Age", "title": "Title"},
)
