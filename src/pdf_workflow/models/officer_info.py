from models.doc_extraction_base import DocumentExtraction
from pydantic import Field, HttpUrl


class OfficerInfo(DocumentExtraction):
    """Extracted signer information from SEC filing.

    Attributes:
        signer_name (str): Name of signing officer
        signer_title (str): Title of signing officer
        date_signed (str): Document signature date
    """

    signer_name: str = Field(description="Name of signing officer")
    signer_title: str = Field(description="Title of signing officer")
    date_signed: str = Field(description="Date document was signed")