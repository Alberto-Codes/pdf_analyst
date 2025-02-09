from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional
from uuid import uuid4

from document_config import DocumentConfig
from models import ExtractionResult
from pydantic_graph import GraphRunContext


@dataclass
class GraphState:
    """State tracking for the document extraction process."""

    run_id: str = field(default_factory=lambda: str(uuid4()))
    document_path: str = ""
    document_config: DocumentConfig = field(default_factory=DocumentConfig)
    raw_response: str = ""
    extracted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    extraction_result: Optional[ExtractionResult] = None
    output_path: str = "data/extraction_export.csv"
    field_order: Optional[List[str]] = None


@dataclass
class End:
    """Signals the end of graph execution."""

    data: ExtractionResult


@dataclass
class BaseNode:
    """Base class for all nodes in the extraction workflow."""

    async def run(self, ctx: GraphRunContext[GraphState]) -> BaseNode | End:
        raise NotImplementedError
