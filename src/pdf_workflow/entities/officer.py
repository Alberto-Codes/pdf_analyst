from typing import List

from base_types import CitedEntity, ExtractionTemplate
from models import Citation


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
