from base_types import CitedEntity, ExtractionTemplate
from pydantic import BaseModel


class EmployeeCount(CitedEntity):
    """Represents the total employee count with citation."""

    count: str  # Using string to handle various formats (e.g., "approximately 5,000", "5,234")
    year: str  # Year the count represents


EMPLOYEE_TEMPLATE = ExtractionTemplate(
    entity_name="EmployeeCount",
    entity_type="EmployeeCount",  # ✅ Pass class name as a string
    fields=["count", "year"],
    is_singular=True,
)
