from pydantic import BaseModel, Field


class EmployeeInfo(BaseModel):
    """
    EmployeeInfo represents the information about employees extracted from a document.

    Attributes:
        total_employees (int): Total number of employees.
        year (int): Year of the employee count.
        source_page (int): Page number where information was found.
        source_text (str): Exact text snippet from the document.
        confidence (float): Confidence score of the extracted information.
    """

    total_employees: int = Field(description="Total number of employees")
    year: int = Field(description="Year of the employee count")
    source_page: int = Field(description="Page number where information was found")
    source_text: str = Field(description="Exact text snippet from document")
    confidence: float = Field(description="Confidence score", ge=0.0, le=1.0)
