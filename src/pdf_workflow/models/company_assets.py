from decimal import Decimal
from typing import List

from models.doc_extraction_base import DocumentExtraction
from pydantic import BaseModel, Field


class Asset(DocumentExtraction):
    """
    A class representing a single asset of the company.

    Attributes:
        asset_type (str): The type of asset (e.g., loans, deposits).
        asset_amount (str): The amount of the asset in USD.
    """

    asset_type: str = Field(description="Type of asset (e.g., loans, deposits)")
    asset_amount: str = Field(
        description="Amount of the asset in USD",
        title="Asset Amount in USD",
    )


class CompanyAssets(BaseModel):
    """
    A class representing a collection of company assets.

    This class holds a list of assets owned by the company.

    Attributes:
        assets (List[Asset]): A list of `Asset` objects representing the
            assets of the company.
    """

    assets: List[Asset] = Field(
        default_factory=list, description="List of company assets"
    )
