import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from base_types import ExtractionTemplate
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
    max_concurrent: int = field(default=3)  # Added as proper dataclass field

    def __post_init__(self):
        """Ensure output directory exists."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        self.workflow = Graph(nodes={GenericExtractNode, ParseNode, ExportNode})

    def _get_output_path(self, input_path: str) -> str:
        """Generate output path for a given input PDF."""
        input_name = Path(input_path).stem
        return str(Path(self.output_dir) / f"{input_name}_export.csv")

    async def _process_single_document(
        self, doc_path: str
    ) -> tuple[str, ExtractionResult]:
        """Process a single document with semaphore control."""
        async with self.semaphore:
            try:
                # Set up state for this document
                state = GraphState(
                    document_path=doc_path,
                    document_config=self.doc_config,
                    output_path=self._get_output_path(doc_path),
                )

                # Set up initial node with correct entity key based on template type
                entity_key = (
                    "employeecount" if self.template.is_singular else "officers"
                )
                extract_node = GenericExtractNode(
                    config=self.config, template=self.template
                )

                # Run workflow for this document
                result, history = await self.workflow.run(extract_node, state)

                print(f"\nProcessed {doc_path}:")
                print(f"Found {len(result.entities)} entities")
                print(f"Output saved to: {state.output_path}")
                print(
                    "Workflow steps:",
                    ", ".join(step.__class__.__name__ for step in history),
                )

                return doc_path, result
            except Exception as e:
                print(f"Error processing {doc_path}: {str(e)}")
                return doc_path, None

    async def process_documents(
        self, document_paths: List[str]
    ) -> List[ExtractionResult]:
        """Process multiple documents in parallel using asyncio."""
        print(
            f"\nStarting parallel processing with max {self.max_concurrent} concurrent tasks"
        )

        tasks = [self._process_single_document(doc_path) for doc_path in document_paths]

        results = await asyncio.gather(*tasks)

        successful_results = [
            (path, result) for path, result in results if result is not None
        ]
        failed_paths = [path for path, result in results if result is None]

        print("\nProcessing Summary:")
        print(f"Successfully processed: {len(successful_results)} documents")
        print(f"Failed to process: {len(failed_paths)} documents")
        if failed_paths:
            print("Failed documents:", failed_paths)

        total_entities = sum(len(result.entities) for _, result in successful_results)
        print(f"Total entities extracted: {total_entities}")

        return [result for _, result in successful_results]
