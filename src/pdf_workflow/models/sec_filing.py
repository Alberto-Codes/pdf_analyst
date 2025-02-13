from datetime import date
from typing import Dict, Union, Optional

from models.company_info import CompanyInfo
from models.officer_info import OfficerInfo
from models.employee_info import EmployeeInfo
from pydantic import BaseModel, HttpUrl


class OcrTags(BaseModel):
    """
    A class representing the OCR (Optical Character Recognition) tags
    for a company's filing.

    This class contains the extracted information about the company,
    officer, and employee from the document.

    Attributes:
        company (CompanyInfo): The company information extracted from the
            document.
        officer (OfficerInfo): The officer information extracted from the
            document.
        employee (EmployeeInfo): The employee information extracted from 
            the document.
    """

    company: CompanyInfo
    officer: OfficerInfo
    employee: EmployeeInfo


class SecFiling(BaseModel):
    """
    A class representing a SEC filing document with relevant metadata.

    This class holds information about the document's location, MIME type,
    and associated OCR tags that contain details about the company, officer, 
    and employee.

    Attributes:
        file_uri (HttpUrl): The URL pointing to the SEC filing document.
        mime_type (str): The MIME type of the document, default is 'application/pdf'.
        ocr_tags (OcrTags): The OCR tags associated with the document, 
            containing information about the company, officer, and employee.
            Default is None.
    """

    file_uri: HttpUrl
    mime_type: str = "application/pdf"
    ocr_tags: OcrTags = None
