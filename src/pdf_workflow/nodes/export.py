from __future__ import annotations

import csv
from dataclasses import dataclass

from base_types import CitedEntity
from models import ExtractionResult
from nodes.base import BaseNode, End, GraphState


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
