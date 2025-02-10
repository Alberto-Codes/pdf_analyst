from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from document_config import DocumentConfig
from models import ExtractionResult
from pydantic import BaseModel, Field


class GraphState(BaseModel):
    """State tracking for the document extraction process."""

    document_path: str
    document_config: DocumentConfig
    output_path: str
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    raw_response: str = Field(default="")
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    extraction_result: Optional[ExtractionResult] = Field(default=None)

    class Config:
        """Pydantic model configuration."""

        arbitrary_types_allowed = True
