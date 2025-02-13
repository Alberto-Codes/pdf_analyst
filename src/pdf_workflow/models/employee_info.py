from models.doc_extraction_base import DocumentExtraction
from pydantic import Field


class EmployeeInfo(DocumentExtraction):
    """
    EmployeeInfo represents the information about employees extracted from a document.

    Attributes:
        total_employees (int): Total number of employees.
        year (int): Year of the employee count.
    """

    total_employees: int = Field(description="Total number of employees")
    year: int = Field(description="Year of the employee count")
