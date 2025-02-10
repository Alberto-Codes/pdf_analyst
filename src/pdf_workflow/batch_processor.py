from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import List

from batch_execution import WorkflowExecutor
from config import GeminiConfig
from core.models import ExtractionResult
from document_config import DocumentConfig
from pydantic import BaseModel, Field
from templates.extraction import ExtractionTemplate


class BatchProcessor(BaseModel):
    """Processes multiple PDFs asynchronously."""

    config: GeminiConfig
    doc_config: DocumentConfig
    template: ExtractionTemplate
    output_dir: str = Field(default="data/exports")
    max_concurrent: int = Field(default=5)
    chunk_size: int = Field(default=3)
    executor: WorkflowExecutor = None
    semaphore: asyncio.Semaphore = None

    class Config:
        """Pydantic model configuration."""

        arbitrary_types_allowed = True

    def model_post_init(self, _):
        """Ensure output directory exists and initialize resources."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        self.executor = WorkflowExecutor(
            config=self.config, doc_config=self.doc_config, template=self.template
        )

    async def _process_chunk(
        self, chunk: List[str]
    ) -> List[tuple[str, ExtractionResult | None]]:
        """Process a chunk of documents."""
        results = await asyncio.gather(
            *(self._process_single_document(doc) for doc in chunk)
        )
        return [
            (doc, result) if isinstance(result, ExtractionResult) else (doc, None)
            for doc, result in results
        ]

    async def _process_single_document(
        self, doc_path: str
    ) -> tuple[str, ExtractionResult | None]:
        """Process a single document."""
        async with self.semaphore:
            return await self.executor.process(doc_path, self.output_dir)

    async def process_documents(
        self, document_paths: List[str]
    ) -> List[ExtractionResult]:
        """Process multiple documents in parallel."""
        print(
            f"\nStarting parallel processing with max {self.max_concurrent} concurrent tasks"
        )
        start_time = time.time()

        results = []
        for i in range(0, len(document_paths), self.chunk_size):
            chunk_results = await self._process_chunk(
                document_paths[i : i + self.chunk_size]
            )
            results.extend(chunk_results)

        total_time = time.time() - start_time
        successful_results = [result for _, result in results if result is not None]
        failed_paths = [path for path, result in results if result is None]

        print(f"\nSuccessfully processed: {len(successful_results)} documents")
        print(f"Failed to process: {len(failed_paths)} documents")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(
            f"Total entities extracted: {sum(len(result.entities) for result in successful_results)}"
        )

        return successful_results
