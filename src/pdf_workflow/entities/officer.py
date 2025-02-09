from dataclasses import dataclass

from base_types import CitedEntity, ExtractionTemplate


@dataclass(kw_only=True)
class Officer(CitedEntity):
    """Represents an officer with their details and citations."""

    name: str
    age: str
    title: str

    def to_csv_row(self) -> dict:
        base_row = super().to_csv_row()
        base_row.update(
            {
                "Name": self.name,
                "Age": self.age,
                "Title": self.title,
            }
        )
        return base_row


OFFICER_TEMPLATE = ExtractionTemplate(
    entity_name="Officer", fields=["name", "age", "title"]
)
