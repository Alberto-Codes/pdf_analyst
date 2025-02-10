from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

from base_types import CitedEntity, ExtractionTemplate
from entities.employee import EmployeeCount
from entities.officer import Officer
from models import Citation, ExtractionResult
from nodes.base import GraphState
from nodes.export import ExportNode
from pydantic_graph import BaseNode, End, GraphRunContext


@dataclass
class ParseNode(BaseNode[GraphState, None, ExtractionResult]):
    """Node that parses the extraction results."""

    entity_type: str
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

    def _create_entity(self, entity_data: dict, source_document: str) -> CitedEntity:
        """Create an entity instance from JSON extraction output."""
        citations = self._parse_citations(entity_data.pop("citations", []))

        entity_class = globals().get(self.entity_type)
        if not entity_class:
            raise ValueError(
                f"Invalid entity type: {self.entity_type}. Available: {list(globals().keys())}"
            )

        return entity_class.model_validate(
            {**entity_data, "citations": citations, "source_document": source_document}
        )

    async def run(
        self, ctx: GraphRunContext[GraphState]
    ) -> ExportNode | End[ExtractionResult]:
        result = json.loads(ctx.state.raw_response)

        if isinstance(result, list):
            if len(result) == 1 and isinstance(result[0], dict):
                result = result[0]
            else:
                raise ValueError(
                    f"Expected a JSON object, but got a list with multiple items: {result}"
                )

        key = self.template.entity_name.lower()
        key = key if self.template.is_singular else key + "s"

        if key not in result:
            raise ValueError(
                f"Expected key '{key}' in extraction result but got: {list(result.keys())}"
            )

        entity_data_list = result[key]

        if not isinstance(entity_data_list, list):
            entity_data_list = [entity_data_list]

        entities = [
            self._create_entity(entity_data, ctx.state.document_path)
            for entity_data in entity_data_list
        ]

        extraction_result = ExtractionResult(
            entities=entities,
            raw_response=ctx.state.raw_response,
            extraction_timestamp=ctx.state.extracted_at,
        )
        ctx.state.extraction_result = extraction_result

        return ExportNode()
