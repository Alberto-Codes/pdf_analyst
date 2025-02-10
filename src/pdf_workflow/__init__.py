"""PDF workflow package for entity extraction using pydantic-graph framework."""

from pdf_workflow.config import GeminiConfig
from pdf_workflow.core.entities import CitedEntity
from pdf_workflow.core.models import Citation, ExtractionResult
from pdf_workflow.document_config import DocumentConfig
from pdf_workflow.graph import Graph
from pdf_workflow.nodes.base import GraphState
from pdf_workflow.nodes.export import ExportNode
from pdf_workflow.nodes.extract import ExtractNode
from pdf_workflow.nodes.parse import ParseNode
from pdf_workflow.templates.extraction import ExtractionTemplate

__all__ = [
    # Core framework components
    "Graph",
    "GraphState",
    # Node implementations
    "ExtractNode",
    "ParseNode",
    "ExportNode",
    # Configuration
    "GeminiConfig",
    "DocumentConfig",
    # Types and models
    "CitedEntity",
    "ExtractionTemplate",
    "Citation",
    "ExtractionResult",
]

__version__ = "0.1.0"
