import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
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
    max_concurrent: int = field(default=5)  # Increased from 3 to 5
    chunk_size: int = field(default=3)  # Process documents in chunks

    def __post_init__(self):
        """Ensure output directory exists and initialize resources."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        self.workflow = Graph(nodes={GenericExtractNode, ParseNode, ExportNode})
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_concurrent)

    def _get_output_path(self, input_path: str) -> str:
        """Generate output path for a given input PDF."""
        input_name = Path(input_path).stem
        return str(Path(self.output_dir) / f"{input_name}_export.csv")

    async def _process_chunk(
        self, chunk: List[str]
    ) -> List[tuple[str, ExtractionResult | None]]:
        """Ensure all results are always a 2-tuple."""
        results = await asyncio.gather(
            *(self._process_single_document(doc) for doc in chunk)
        )

        # ✅ Ensure every result is correctly structured
        return [
            (doc, result) if isinstance(result, ExtractionResult) else (doc, None)
            for doc, result in results
        ]

    async def _process_single_document(
        self, doc_path: str
    ) -> tuple[str, ExtractionResult | None]:
        """Ensure it always returns a (path, ExtractionResult) tuple."""
        start_time = time.time()
        async with self.semaphore:
            try:
                if not doc_path:
                    raise ValueError("Invalid document path provided.")

                output_path = self._get_output_path(doc_path)
                if not output_path:
                    raise ValueError(f"Failed to determine output path for {doc_path}")

                state = GraphState(
                    document_path=doc_path,
                    document_config=self.doc_config,
                    output_path=output_path,
                )

                start_node = GenericExtractNode(
                    config=self.config, template=self.template
                )
                result, history = await self.workflow.run(start_node, state)

                print(f"\nProcessed {doc_path}:")
                print(f"Entities extracted: {len(result.entities)}")
                print(f"Output saved to: {state.output_path}")
                print(f"Processing time: {time.time() - start_time:.2f} seconds")
                print(
                    "Workflow steps:",
                    ", ".join(step.__class__.__name__ for step in history),
                )

                return doc_path, result  # ✅ Always return a tuple

            except Exception as e:
                print(f"Error processing {doc_path}: {str(e)}")
                return doc_path, None  # ✅ Ensure it's always a tuple

    async def process_documents(
        self, document_paths: List[str]
    ) -> List[ExtractionResult]:
        """Process multiple documents and ensure tuple structure is correct."""
        print(
            f"\nStarting parallel processing with max {self.max_concurrent} concurrent tasks"
        )
        start_time = time.time()

        results = []
        for i in range(0, len(document_paths), self.chunk_size):
            chunk = document_paths[i : i + self.chunk_size]
            chunk_results = await self._process_chunk(chunk)
            results.extend(
                chunk_results
            )  # ✅ Guaranteed to be a list of (path, result)

        successful_results = [
            (path, result) for path, result in results if result is not None
        ]
        failed_paths = [path for path, result in results if result is None]

        total_time = time.time() - start_time

        print("\nProcessing Summary:")
        print(f"Successfully processed: {len(successful_results)} documents")
        print(f"Failed to process: {len(failed_paths)} documents")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(
            f"Average time per document: {total_time/len(document_paths):.2f} seconds"
        )

        if failed_paths:
            print("Failed documents:", failed_paths)

        total_entities = sum(len(result.entities) for _, result in successful_results)
        print(f"Total entities extracted: {total_entities}")

        return [result for _, result in successful_results]
