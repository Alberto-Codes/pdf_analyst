from dataclasses import dataclass

from base_types import CitedEntity, ExtractionTemplate


@dataclass(kw_only=True)
class Officer(CitedEntity):
    """Represents an officer with their details and citations."""

    name: str
    age: str
    title: str


OFFICER_TEMPLATE = ExtractionTemplate(
    entity_name="Officer",
    entity_type=Officer,
    fields=["name", "age", "title"],
    field_mapping={"name": "Name", "age": "Age", "title": "Title"},
    field_order=[
        "Name",
        "Age",
        "Title",
        "Source_Document",
        "Extracted_At",
        "Page_Numbers",
        "Text_Snippets",
        "Average_Confidence",
    ],
)
