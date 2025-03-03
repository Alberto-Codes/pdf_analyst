from pdf_workflow.models.doc_extraction_base import DocumentExtraction
from pydantic import Field


class CompanyInfo(DocumentExtraction):
    """
    A class representing company information extracted from a document.

    This class inherits from the `DocumentExtraction` base class and
    contains the essential fields related to company details.

    Attributes:
        company_name (str): The legal name of the company.
        ein (str): The Employer Identification Number (EIN) of the company.
    """

    company_name: str = Field(description="Legal name of the company")
    ein: str = Field(description="Employer Identification Number (EIN)")

