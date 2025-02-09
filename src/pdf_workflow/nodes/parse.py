from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Type

from base_types import CitedEntity, ExtractionTemplate
from models import Citation, ExtractionResult
from nodes.base import GraphState
from nodes.export import ExportNode
from pydantic_graph import BaseNode, End, GraphRunContext


@dataclass
class ParseNode(BaseNode):
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

    async def run(
        self, ctx: GraphRunContext[GraphState]
    ) -> ExportNode | End[ExtractionResult]:
        try:

            result = json.loads(ctx.state.raw_response)
            entity_class = self.entity_type.with_mapping(self.template.field_mapping)

            key = "employeecount" if self.template.is_singular else "officers"

            if self.template.is_singular:

                entity_data = result[key]
                entities = [
                    self._create_entity(
                        entity_data, entity_class, ctx.state.document_path
                    )
                ]
            else:

                entities = [
                    self._create_entity(
                        entity_data, entity_class, ctx.state.document_path
                    )
                    for entity_data in result.get(key, [])
                ]

            extraction_result = ExtractionResult(
                entities=entities,
                raw_response=ctx.state.raw_response,
                extraction_timestamp=ctx.state.extracted_at,
            )
            ctx.state.extraction_result = extraction_result

            ctx.state.field_order = self.template.field_order

            return ExportNode()
        except Exception as e:
            raise Exception(f"Error in parsing: {str(e)}")
