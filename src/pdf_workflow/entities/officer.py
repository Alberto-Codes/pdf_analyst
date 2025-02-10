"""Officer entity definition."""

from core.entities import CitedEntity
from templates.extraction import ExtractionTemplate


class Officer(CitedEntity):
    """Represents an officer with their details and citations."""

    name: str
    age: str
    title: str


OFFICER_TEMPLATE = ExtractionTemplate(
    entity_name="Officer",
    entity_type="Officer",
    fields=["name", "age", "title"],
)
