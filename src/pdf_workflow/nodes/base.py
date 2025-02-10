from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional
from uuid import uuid4

from document_config import DocumentConfig
from models import ExtractionResult


@dataclass
class GraphState:
    """State tracking for the document extraction process."""

    document_path: str
    document_config: DocumentConfig
    output_path: str
    run_id: str = field(default_factory=lambda: str(uuid4()))
    raw_response: str = field(default="")
    extracted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    extraction_result: Optional[ExtractionResult] = field(default=None)
    field_order: Optional[List[str]] = field(default=None)
