from __future__ import annotations

import csv
import json
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from typing import Generic, Type, TypeVar

from base_types import CitedEntity, ExtractionTemplate
from config import GeminiConfig
from google.genai import types
from models import Citation, ExtractionResult
from prompts import PromptTemplate
from utils import encode_file

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
    mime_type: str = "application/pdf"  # Make file type configurable
    stream_response: bool = True  # Allow configuring whether to stream
    encoding: str = "utf-8"  # Make encoding configurable

    async def run(self, state: GraphState) -> ParseNode | End[ExtractionResult]:
        try:
            encoded_file = encode_file(
                state.pdf_path
            )  # This function name is still PDF-specific
            document = types.Part.from_bytes(
                data=encoded_file,
                mime_type=self.mime_type,
            )

            contents = PromptTemplate.create_extraction_content(
                document, self.template.get_prompt()
            )

            response_text = ""
            if self.stream_response:
                for chunk in self.config.client.models.generate_content_stream(
                    model=self.config.model,
                    contents=contents,
                    config=self.config.generate_config,
                ):
                    response_text += chunk.text
            else:
                response = self.config.client.models.generate_content(
                    model=self.config.model,
                    contents=contents,
                    config=self.config.generate_config,
                )
                response_text = response.text

            state.raw_response = response_text
            return ParseNode(
                entity_type=self.template.entity_type,
                entity_key=self.template.entity_name.lower() + "s",
            )
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
    """Node that handles CSV export for any type of CitedEntity."""

    async def run(self, state: GraphState) -> End[ExtractionResult]:
        try:
            if not state.extraction_result or not state.extraction_result.entities:
                raise ValueError("No entities data to export")

            # Get the first entity to determine the type
            first_entity = state.extraction_result.entities[0]
            if not isinstance(first_entity, CitedEntity):
                raise TypeError("Entities must inherit from CitedEntity")

            # Get field names from the entity type and capitalize them
            entity_fields = [
                field.name.capitalize()
                for field in fields(type(first_entity))
                if field.name not in fields(CitedEntity)
            ]

            # Combine with base CitedEntity CSV fields
            base_fields = [
                "Source_Document",
                "Extracted_At",
                "Page_Numbers",
                "Text_Snippets",
                "Average_Confidence",
            ]
            fieldnames = entity_fields + base_fields

            # Get CSV-friendly rows
            rows = [entity.to_csv_row() for entity in state.extraction_result.entities]

            # Write to CSV
            with open(state.output_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            print(f"\nExported {len(rows)} entities to: {state.output_path}")
            return End(state.extraction_result)
        except Exception as e:
            raise Exception(f"Error in CSV export: {str(e)}")
