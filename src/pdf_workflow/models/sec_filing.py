from datetime import date
from typing import Dict, Union, Optional

from models.company_info import CompanyInfo
from models.officer_info import OfficerInfo
from pydantic import BaseModel, HttpUrl


class OcrTags(BaseModel):
    company: CompanyInfo
    officer: OfficerInfo


class SecFiling(BaseModel):
    file_uri: HttpUrl
    mime_type: str="application/pdf"
    ocr_tags: OcrTags = None
