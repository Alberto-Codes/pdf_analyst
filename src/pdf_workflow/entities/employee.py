from dataclasses import dataclass

from base_types import CitedEntity, ExtractionTemplate


@dataclass(kw_only=True)
class EmployeeCount(CitedEntity):
    """Represents the total employee count with citation."""

    count: str  # Using string to handle various formats (e.g., "approximately 5,000", "5,234")
    year: str  # Year the count represents


class EmployeeTemplate(ExtractionTemplate):
    """Custom template for employee count extraction."""

    def get_prompt(self) -> str:
        """Override to provide more specific employee count prompt."""
        return """
        Extract the total employee count or workforce size information.
        Look specifically for statements about:
        - Total number of employees
        - Total workforce size
        - Global employee count
        - Full-time equivalent employees
        
        Format the response as a JSON object with this exact structure:
        {
            "employeecount": {
                "count": "string with the number of employees",
                "year": "string with the year of the count",
                "citations": [
                    {
                        "page_number": number,
                        "text_snippet": "string",
                        "confidence_score": number
                    }
                ]
            }
        }

        For the citation:
        - Include the page number where the information was found
        - Include a brief text snippet from the page (max 100 chars)
        - Provide a confidence score (0.0-1.0) for the citation
        - Only include the most recent or most relevant employee count

        Example response:
        {
            "employeecount": {
                "count": "5,234",
                "year": "2023",
                "citations": [
                    {
                        "page_number": 12,
                        "text_snippet": "As of December 31, 2023, we had 5,234 full-time employees worldwide.",
                        "confidence_score": 0.95
                    }
                ]
            }
        }
        """


EMPLOYEE_TEMPLATE = EmployeeTemplate(
    entity_name="EmployeeCount",
    entity_type=EmployeeCount,
    fields=["count", "year"],
    field_mapping={"count": "Employee_Count", "year": "Year"},
    field_order=[
        "Employee_Count",
        "Year",
        "Source_Document",
        "Extracted_At",
        "Page_Numbers",
        "Text_Snippets",
        "Average_Confidence",
    ],
    is_singular=True,
)
