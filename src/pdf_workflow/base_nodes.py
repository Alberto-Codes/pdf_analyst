from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime, timezone
from typing import Generic, List, TypeVar

from document_config import DocumentConfig
from models import ExtractionResult
from utils import encode_file

StateT = TypeVar("StateT")
RunEndT = TypeVar("RunEndT")


@dataclass
class GraphState:
    """Holds the state of the extraction process."""

    document_path: str
    document_config: DocumentConfig
    raw_response: str = ""
    extracted_at: datetime = datetime.now(timezone.utc)
    extraction_result: ExtractionResult | None = None
    output_path: str = "data/extraction_export.csv"
    field_order: List[str] | None = None  # Added field_order to state


@dataclass
class End(Generic[RunEndT]):
    """Signals the end of graph execution."""

    data: RunEndT


@dataclass
class BaseNode(Generic[StateT]):
    """Base class for all nodes in the extraction workflow."""

    async def run(self, state: StateT) -> BaseNode[StateT] | End[ExtractionResult]:
        raise NotImplementedError
