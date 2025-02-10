from __future__ import annotations

import csv
from pathlib import Path

from core.entities import CitedEntity
from core.models import ExtractionResult
from nodes.base import GraphState
from pydantic import BaseModel
from pydantic_graph import BaseNode, End, GraphRunContext


class ExportNode(BaseNode[GraphState, None, ExtractionResult], BaseModel):
    """Node that handles CSV export for any type of CitedEntity."""

    class Config:
        """Pydantic model configuration."""

        arbitrary_types_allowed = True

    async def run(self, ctx: GraphRunContext[GraphState]) -> End[ExtractionResult]:
        if not ctx.state.extraction_result or not ctx.state.extraction_result.entities:
            raise ValueError("No entities data to export")

        first_entity = ctx.state.extraction_result.entities[0]
        if not isinstance(first_entity, CitedEntity):
            raise TypeError("Entities must inherit from CitedEntity")

        all_fieldnames = set()
        for entity in ctx.state.extraction_result.entities:
            all_fieldnames.update(entity.to_csv_row().keys())
        all_fieldnames = sorted(list(all_fieldnames))

        output_path = Path(ctx.state.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=all_fieldnames)
            writer.writeheader()
            writer.writerows(
                entity.to_csv_row() for entity in ctx.state.extraction_result.entities
            )

        return End(ctx.state.extraction_result)
