from __future__ import annotations

import csv
import json
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from typing import Generic, Type, TypeVar

from base_types import CitedEntity, ExtractionTemplate
from config import GeminiConfig
from document_config import DocumentConfig
from google.genai import types
from models import Citation, ExtractionResult
from prompts import PromptTemplate
from utils import encode_file

StateT = TypeVar("StateT")
RunEndT = TypeVar("RunEndT")


@dataclass
class GraphState:
    """Holds the state of the extraction process."""

    document_path: str  # Changed from pdf_path
    document_config: DocumentConfig
    raw_response: str = ""
    extracted_at: datetime = datetime.now(timezone.utc)
    extraction_result: ExtractionResult | None = None
    output_path: str = "data/extraction_export.csv"


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
            encoded_file = encode_file(
                state.document_path, encoding=state.document_config.encoding
            )

            document = types.Part.from_bytes(
                data=encoded_file,
                mime_type=state.document_config.mime_type,
            )

            contents = PromptTemplate.create_extraction_content(
                document, self.template.get_prompt()
            )

            response_text = ""
            if state.document_config.stream_response:
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
                template=self.template,
            )
        except Exception as e:
            raise Exception(f"Error in extraction: {str(e)}")


@dataclass
class ParseNode(BaseNode[GraphState]):
    """Node that parses the extraction results."""

    entity_type: Type  # The type of entity to parse (e.g., Officer)
    entity_key: str  # The key in the JSON response (e.g., "officers")
    template: ExtractionTemplate  # Add template parameter

    async def run(self, state: GraphState) -> ExportNode | End[ExtractionResult]:
        try:
            result = json.loads(state.raw_response)
            entities = []

            # Apply field mapping from template
            entity_class = self.entity_type.with_mapping(self.template.field_mapping)

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
                entity = entity_class(
                    **entity_dict,
                    citations=citations,
                    source_document=state.document_path,
                )
                entities.append(entity)

            extraction_result = ExtractionResult(
                entities=entities,
                raw_response=state.raw_response,
                extraction_timestamp=state.extracted_at,
            )
            state.extraction_result = extraction_result
            return ExportNode(field_order=self.template.field_order)
        except Exception as e:
            raise Exception(f"Error in parsing: {str(e)}")


@dataclass
class ExportNode(BaseNode[GraphState]):
    """Node that handles CSV export for any type of CitedEntity."""

    field_order: List[str] = None  # Add field order parameter

    async def run(self, state: GraphState) -> End[ExtractionResult]:
        try:
            if not state.extraction_result or not state.extraction_result.entities:
                raise ValueError("No entities data to export")

            first_entity = state.extraction_result.entities[0]
            if not isinstance(first_entity, CitedEntity):
                raise TypeError("Entities must inherit from CitedEntity")

            # Use provided field order or get from entity
            fieldnames = self.field_order or getattr(
                type(first_entity), "field_order", None
            )

            if not fieldnames:
                # Fall back to default field order from CitedEntity
                fieldnames = first_entity.get_csv_fields()

            # Write to CSV
            with open(state.output_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(
                    entity.to_csv_row() for entity in state.extraction_result.entities
                )

            print(
                f"\nExported {len(state.extraction_result.entities)} entities to: {state.output_path}"
            )
            return End(state.extraction_result)
        except Exception as e:
            raise Exception(f"Error in CSV export: {str(e)}")
