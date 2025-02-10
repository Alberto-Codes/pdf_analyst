from __future__ import annotations

import csv
from dataclasses import dataclass

from base_types import CitedEntity
from models import ExtractionResult
from nodes.base import GraphState
from pydantic_graph import BaseNode, End, GraphRunContext


@dataclass
class ExportNode(BaseNode[GraphState, None, ExtractionResult]):
    """Node that handles CSV export for any type of CitedEntity."""

    async def run(self, ctx: GraphRunContext[GraphState]) -> End[ExtractionResult]:
        try:
            if (
                not ctx.state.extraction_result
                or not ctx.state.extraction_result.entities
            ):
                raise ValueError("No entities data to export")

            first_entity = ctx.state.extraction_result.entities[0]
            if not isinstance(first_entity, CitedEntity):
                raise TypeError("Entities must inherit from CitedEntity")

            # Collect all possible fieldnames from the entities
            all_fieldnames = set()
            for entity in ctx.state.extraction_result.entities:
                all_fieldnames.update(entity.to_csv_row().keys())
            all_fieldnames = sorted(list(all_fieldnames))

            with open(
                ctx.state.output_path, "w", newline="", encoding="utf-8"
            ) as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=all_fieldnames)
                writer.writeheader()
                writer.writerows(
                    entity.to_csv_row()
                    for entity in ctx.state.extraction_result.entities
                )

            print(
                f"\nExported {len(ctx.state.extraction_result.entities)} entities to: {ctx.state.output_path}"
            )
            return End(ctx.state.extraction_result)

        except Exception as e:
            raise Exception(f"Error in CSV export: {str(e)}")
