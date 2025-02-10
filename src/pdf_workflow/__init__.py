"""PDF workflow package for entity extraction using pydantic-graph framework."""

from base_types import CitedEntity, ExtractionTemplate
from config import GeminiConfig
from document_config import DocumentConfig
from graph import Graph
from models import Citation, ExtractionResult
from nodes.base import GraphState
from nodes.export import ExportNode
from nodes.extract import ExtractNode
from nodes.parse import ParseNode

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
