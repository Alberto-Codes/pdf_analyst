from dataclasses import dataclass

from base_types import CitedEntity, ExtractionTemplate


@dataclass(kw_only=True)
class Officer(CitedEntity):
    """Represents an officer with their details and citations."""

    name: str
    age: str
    title: str

    # Define field ordering for Officer-specific fields first, then base fields
    field_order = [
        "Name",
        "Age",
        "Title",  # Entity-specific fields first
        "Source_Document",
        "Extracted_At",
        "Page_Numbers",
        "Text_Snippets",
        "Average_Confidence",  # Base fields last
    ]

    # Add field mapping for Officer-specific fields
    csv_field_mapping = {
        **CitedEntity.csv_field_mapping,  # Include parent class mappings
        "name": "Name",
        "age": "Age",
        "title": "Title",
    }

    def to_csv_row(self) -> dict:
        # Get base row from parent
        base_row = super().to_csv_row()

        # Add officer-specific fields
        officer_fields = {
            "Name": self.name,
            "Age": self.age,
            "Title": self.title,
        }

        # Create ordered dict based on field_order
        return {
            field: (officer_fields.get(field) or base_row.get(field))
            for field in self.field_order
        }


OFFICER_TEMPLATE = ExtractionTemplate(
    entity_name="Officer",
    entity_type=Officer,  # Add this line
    fields=["name", "age", "title"],
)
