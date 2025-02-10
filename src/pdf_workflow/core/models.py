from datetime import datetime
from typing import TYPE_CHECKING, List

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from core.entities import CitedEntity


class Citation(BaseModel):
    """Represents a citation with page number and confidence score."""

    page_number: int
    text_snippet: str
    confidence_score: float


class ExtractionResult(BaseModel):
    """Represents the complete extraction result with entities and metadata."""

    entities: List["CitedEntity"] = Field(default_factory=list)
    raw_response: str
    extraction_timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        """Pydantic model configuration."""

        arbitrary_types_allowed = True

    def to_dict(self) -> dict:
        """Convert extraction result to dictionary format."""
        return {
            "entities": [entity.dict() for entity in self.entities],
            "raw_response": self.raw_response,
            "extraction_timestamp": self.extraction_timestamp.isoformat(),
        }


# This is crucial - rebuild the model after all imports are done
from core.entities import CitedEntity

ExtractionResult.model_rebuild()
