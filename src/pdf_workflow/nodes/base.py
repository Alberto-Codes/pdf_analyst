from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Generic, List, Optional, TypeVar
from uuid import uuid4

from document_config import DocumentConfig
from models import ExtractionResult
from pydantic import BaseModel, Field

StateT = TypeVar("StateT")
RunEndT = TypeVar("RunEndT")


class GraphState(BaseModel):
    """Pydantic-based state tracking for the extraction process."""

    run_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique ID for the processing run",
    )
    document_path: str = Field(..., description="Path to the document being processed")
    document_config: DocumentConfig
    raw_response: str = Field(
        default="", description="Raw response from the extraction process"
    )
    extracted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of extraction",
    )
    extraction_result: Optional[ExtractionResult] = Field(
        default=None, description="Extracted entities and metadata"
    )
    output_path: str = Field(
        default="data/extraction_export.csv", description="Path for exported CSV"
    )
    field_order: Optional[List[str]] = Field(
        default=None, description="Field order for CSV export"
    )


@dataclass
class End(Generic[RunEndT]):
    """Signals the end of graph execution."""

    data: RunEndT


@dataclass
class BaseNode(Generic[StateT]):
    """Base class for all nodes in the extraction workflow."""

    async def run(self, state: StateT) -> BaseNode[StateT] | End[ExtractionResult]:
        raise NotImplementedError
