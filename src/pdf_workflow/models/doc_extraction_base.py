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
        document_id (str | None): An optional identifier for the source document.
            Defaults to `None` if not provided.
        bbox (tuple[float, float, float, float] | None): Optional bounding box
            coordinates representing the location of the extracted text in
            the document. The coordinates are given as `(x1, y1, x2, y2)`,
            where `x1, y1` is the top-left corner and `x2, y2` is the bottom-right
            corner. Defaults to `None` if not applicable.
    """

    page_number: int = Field(description="Source page number in document")
    context: str = Field(description="Raw text context from source")
    confidence_score: confloat(ge=0.0, le=1.0) = Field(
        description="Model confidence in extraction"
    )
    document_id: str | None = Field(
        default=None, description="Source document identifier"
    )
    bbox: tuple[float, float, float, float] | None = Field(
        default=None, description="Bounding box coordinates (x1, y1, x2, y2)"
    )