from dataclasses import dataclass
from pathlib import Path
from typing import List

from base_types import ExtractionTemplate  # Added this import
from config import GeminiConfig
from document_config import DocumentConfig
from graph import Graph
from graph_nodes import ExportNode, GenericExtractNode, GraphState, ParseNode
from models import ExtractionResult


@dataclass
class BatchProcessor:
    """Processes multiple PDFs using the existing graph structure."""

    config: GeminiConfig
    doc_config: DocumentConfig
    template: ExtractionTemplate
    output_dir: str = "data/exports"

    def __post_init__(self):
        """Ensure output directory exists."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def _get_output_path(self, input_path: str) -> str:
        """Generate output path for a given input PDF."""
        input_name = Path(input_path).stem
        return str(Path(self.output_dir) / f"{input_name}_export.csv")

    async def process_documents(
        self, document_paths: List[str]
    ) -> List[ExtractionResult]:
        """Process multiple documents using the existing graph."""
        results = []
        workflow = Graph(nodes={GenericExtractNode, ParseNode, ExportNode})

        for doc_path in document_paths:
            # Create state for each document
            state = GraphState(
                document_path=doc_path,
                document_config=self.doc_config,
                output_path=self._get_output_path(doc_path),
            )

            # Initialize extraction node
            extract_node = GenericExtractNode(
                config=self.config, template=self.template
            )

            # Run workflow for current document
            result, history = await workflow.run(extract_node, state)
            results.append(result)

            print(f"\nProcessed {doc_path}:")
            print(f"Found {len(result.entities)} entities")
            print(f"Output saved to: {state.output_path}")
            print(
                "Workflow steps:",
                ", ".join(step.__class__.__name__ for step in history),
            )

        return results
