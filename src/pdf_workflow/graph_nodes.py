from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Generic, Type, TypeVar

from base_types import ExtractionTemplate
from config import GeminiConfig
from entities.officer import Officer
from google.genai import types
from models import Citation, ExtractionResult
from prompts import PromptTemplate
from utils import encode_pdf

StateT = TypeVar("StateT")
RunEndT = TypeVar("RunEndT")


@dataclass
class GraphState:
    """Holds the state of the extraction process."""

    pdf_path: str
    raw_response: str = ""
    extracted_at: datetime = datetime.now(timezone.utc)
    extraction_result: ExtractionResult | None = None
    output_path: str = "data/officers_export.csv"


@dataclass
class End(Generic[RunEndT]):
    """Signals the end of graph execution."""

    data: RunEndT


@dataclass
class BaseNode(Generic[StateT]):
    """Base class for all nodes in the extraction workflow."""

    async def run(self, state: StateT) -> BaseNode[StateT] | End[ExtractionResult]:
        raise NotImplementedError


@dataclass
class GenericExtractNode(BaseNode[GraphState]):
    """Generic node for extracting entities with citations."""

    config: GeminiConfig
    template: ExtractionTemplate

    async def run(self, state: GraphState) -> ParseNode | End[ExtractionResult]:
        try:
            encoded_pdf = encode_pdf(state.pdf_path)
            document = types.Part.from_bytes(
                data=encoded_pdf,
                mime_type="application/pdf",
            )

            contents = PromptTemplate.create_extraction_content(
                document, self.template.get_prompt()
            )

            response_text = ""
            for chunk in self.config.client.models.generate_content_stream(
                model=self.config.model,
                contents=contents,
                config=self.config.generate_config,
            ):
                response_text += chunk.text

            state.raw_response = response_text
            return ParseNode(entity_type=Officer, entity_key="officers")
        except Exception as e:
            raise Exception(f"Error in extraction: {str(e)}")


@dataclass
class ParseNode(BaseNode[GraphState]):
    """Node that parses the extraction results."""

    entity_type: Type  # The type of entity to parse (e.g., Officer)
    entity_key: str  # The key in the JSON response (e.g., "officers")

    async def run(self, state: GraphState) -> ExportNode | End[ExtractionResult]:
        try:
            result = json.loads(state.raw_response)
            entities = []

            for entity_data in result[self.entity_key]:
                citations = [
                    Citation(
                        page_number=cite["page_number"],
                        text_snippet=cite["text_snippet"],
                        confidence_score=cite["confidence_score"],
                    )
                    for cite in entity_data["citations"]
                ]

                # Remove citations from entity_data since we handle it separately
                entity_dict = {k: v for k, v in entity_data.items() if k != "citations"}

                # Create the entity instance with citations and source document
                entity = self.entity_type(
                    **entity_dict,
                    citations=citations,
                    source_document=state.pdf_path,
                )
                entities.append(entity)

            extraction_result = ExtractionResult(
                entities=entities,  # This field name should probably be made generic too
                raw_response=state.raw_response,
                extraction_timestamp=state.extracted_at,
            )
            state.extraction_result = extraction_result
            return ExportNode()
        except Exception as e:
            raise Exception(f"Error in parsing: {str(e)}")


@dataclass
class ExportNode(BaseNode[GraphState]):
    """Node that handles CSV export of the results."""

    async def run(self, state: GraphState) -> End[ExtractionResult]:
        try:
            if not state.extraction_result or not state.extraction_result.entities:
                raise ValueError("No officers data to export")

            # Get CSV-friendly rows
            rows = [
                officer.to_csv_row() for officer in state.extraction_result.entities
            ]

            # Write to CSV
            with open(state.output_path, "w", newline="", encoding="utf-8") as csvfile:
                fieldnames = [
                    "Name",
                    "Age",
                    "Title",
                    "Source_Document",
                    "Extracted_At",
                    "Page_Numbers",
                    "Text_Snippets",
                    "Average_Confidence",
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            print(f"\nExported officers data to: {state.output_path}")
            return End(state.extraction_result)
        except Exception as e:
            raise Exception(f"Error in CSV export: {str(e)}")
