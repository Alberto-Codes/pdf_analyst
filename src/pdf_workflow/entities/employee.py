"""Employee count entity definition."""

from core.entities import CitedEntity
from templates.extraction import ExtractionTemplate


class EmployeeCount(CitedEntity):
    """Represents the total employee count with citation."""

    count: str  # Using string to handle various formats (e.g., "approximately 5,000", "5,234")
    year: str  # Year the count represents


EMPLOYEE_TEMPLATE = ExtractionTemplate(
    entity_name="EmployeeCount",
    entity_type="EmployeeCount",
    fields=["count", "year"],
    is_singular=True,
)
