from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from typing import Generic, List, Type, TypeVar

from base_nodes import BaseNode, End, GraphState
from base_types import CitedEntity, ExtractionTemplate
from config import GeminiConfig
from google.genai import types
from models import Citation, ExtractionResult
from prompts import PromptTemplate
from utils import encode_file


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
            state.field_order = (
                self.template.field_order
            )  # ✅ Store field order in state

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

    entity_type: Type
    entity_key: str
    template: ExtractionTemplate

    def _parse_citations(self, citations_data: List[dict]) -> List[Citation]:
        """Parse citation data into Citation objects."""
        return [
            Citation(
                page_number=cite["page_number"],
                text_snippet=cite["text_snippet"],
                confidence_score=cite["confidence_score"],
            )
            for cite in citations_data
        ]

    def _create_entity(
        self, entity_data: dict, entity_class: Type, source_document: str
    ) -> CitedEntity:
        """Create an entity instance from JSON extraction output."""
        citations = self._parse_citations(entity_data.pop("citations", []))
        return entity_class(
            **entity_data,
            citations=citations,
            source_document=source_document,
        )

    async def run(self, state: GraphState) -> ExportNode | End[ExtractionResult]:
        try:
            # Load and parse JSON response
            result = json.loads(state.raw_response)
            entity_class = self.entity_type.with_mapping(self.template.field_mapping)

            # Get the correct key based on template type
            key = "employeecount" if self.template.is_singular else "officers"

            if self.template.is_singular:
                # Handle single entity
                entity_data = result[key]
                entities = [
                    self._create_entity(entity_data, entity_class, state.document_path)
                ]
            else:
                # Handle list of entities
                entities = [
                    self._create_entity(entity_data, entity_class, state.document_path)
                    for entity_data in result.get(key, [])
                ]

            # Create extraction result
            extraction_result = ExtractionResult(
                entities=entities,
                raw_response=state.raw_response,
                extraction_timestamp=state.extracted_at,
            )
            state.extraction_result = extraction_result

            state.field_order = (
                self.template.field_order
            )  # Ensure field order is stored

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

            first_entity = state.extraction_result.entities[0]
            if not isinstance(first_entity, CitedEntity):
                raise TypeError("Entities must inherit from CitedEntity")

            fieldnames = state.field_order  # ✅ Use stored field order
            if not fieldnames:
                fieldnames = first_entity.get_csv_fields()

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
