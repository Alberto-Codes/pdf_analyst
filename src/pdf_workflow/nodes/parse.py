from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Type

from base_types import CitedEntity, ExtractionTemplate
from models import Citation, ExtractionResult
from nodes.base import BaseNode, End, GraphState
from nodes.export import ExportNode


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
