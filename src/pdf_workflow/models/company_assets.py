# src/pdf_workflow/models/company_assets.py
from typing import List
from decimal import Decimal
from models.doc_extraction_base import DocumentExtraction
from pydantic import BaseModel, Field




class Asset(DocumentExtraction):
    asset_type: str = Field(description="Type of asset (e.g., loans, deposits)")
    asset_amount: str = Field(
        description="Amount of the asset in USD",
        title="Asset Amount in USD",
    )


class CompanyAssets(BaseModel):
    assets: List[Asset] = Field(default_factory=list, description="List of company assets")
