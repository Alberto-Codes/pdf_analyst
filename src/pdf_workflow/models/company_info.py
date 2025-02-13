from models.doc_extraction_base import DocumentExtraction
from pydantic import Field


class CompanyInfo(DocumentExtraction):
    company_name: str = Field(description="Legal name of the company")
    ein: str = Field(description="Employer Identification Number (EIN)")
