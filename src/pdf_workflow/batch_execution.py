import time
from dataclasses import dataclass
from pathlib import Path

from base_types import ExtractionTemplate
from config import GeminiConfig
from document_config import DocumentConfig
from graph import Graph
from models import ExtractionResult
from nodes.base import GraphState
from nodes.export import ExportNode
from nodes.extract import ExtractNode
from nodes.parse import ParseNode


@dataclass
class WorkflowExecutor:
    """Handles the execution of document processing workflows."""

    config: GeminiConfig
    doc_config: DocumentConfig
    template: ExtractionTemplate

    def _get_output_path(self, input_path: str, output_dir: str) -> str:
        """Generate output path for a given input PDF."""
        input_name = Path(input_path).stem
        return str(Path(output_dir) / f"{input_name}_export.csv")

    async def process(
        self, doc_path: str, output_dir: str
    ) -> tuple[str, ExtractionResult | None]:
        """Run the workflow for a single document."""
        start_time = time.time()

        try:
            if not doc_path:
                raise ValueError("Invalid document path provided.")

            output_path = self._get_output_path(doc_path, output_dir)
            state = GraphState(
                document_path=doc_path,
                document_config=self.doc_config,
                output_path=output_path,
            )

            start_node = ExtractNode(config=self.config, template=self.template)
            workflow = Graph(
                nodes={
                    "extract": ExtractNode,
                    "parse": ParseNode,
                    "export": ExportNode,
                }
            )

            result, history = await workflow.run(start_node, state)

            print(f"\nProcessed {doc_path}:")
            print(f"Entities extracted: {len(result.entities)}")
            print(f"Output saved to: {output_path}")
            print(f"Processing time: {time.time() - start_time:.2f} seconds")
            print(
                "Workflow steps:",
                ", ".join(step.__class__.__name__ for step in history),
            )

            return doc_path, result

        except Exception as e:
            print(f"Error processing {doc_path}: {str(e)}")
            return doc_path, None
