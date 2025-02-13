from pydantic import BaseModel, Field, confloat


class DocumentExtraction(BaseModel):
    """Represents AI-based document extraction results.

    This model serves as a standardized base class for information extracted
    from documents using Large Language Models (LLMs) or Optical Character
    Recognition (OCR). It includes source attribution details, confidence
    metrics, and optional bounding box coordinates.

    Attributes:
        page_number (int): The source page number in the document where the
            extracted information was found.
        context (str): The raw text extracted from the document, providing
            contextual information for the extracted data.
        confidence_score (float): The model's confidence score in the extraction,
            constrained between `0.0` (least confident) and `1.0` (most confident).
    """

    page_number: int = Field(description="Source page number in document")
    context: str = Field(description="Raw text context from source")
    confidence_score: float = Field(
        description="Model confidence in extraction", ge=0.0, le=1.0
    )
