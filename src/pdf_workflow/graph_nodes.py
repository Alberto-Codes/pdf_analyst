from __future__ import annotations

import json
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TypeVar, Generic
from google.genai import types

from models import Citation, Officer, ExtractionResult
from config import GeminiConfig
from prompts import PromptTemplate
from utils import encode_pdf

StateT = TypeVar('StateT')
RunEndT = TypeVar('RunEndT')

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
class ExtractNode(BaseNode[GraphState]):
    """Node that handles the extraction of officer information."""

    config: GeminiConfig

    async def run(self, state: GraphState) -> ParseNode | End[ExtractionResult]:
        try:
            # Use your existing GeminiPDFParser logic here
            encoded_pdf = encode_pdf(state.pdf_path)
            document = types.Part.from_bytes(
                data=encoded_pdf,
                mime_type="application/pdf",
            )

            contents = PromptTemplate.create_extraction_content(document)

            response_text = ""
            for chunk in self.config.client.models.generate_content_stream(
                model=self.config.model,
                contents=contents,
                config=self.config.generate_config,
            ):
                response_text += chunk.text

            state.raw_response = response_text
            return ParseNode()
        except Exception as e:
            raise Exception(f"Error in extraction: {str(e)}")


@dataclass
class ParseNode(BaseNode[GraphState]):
    """Node that parses the extraction results."""
    async def run(self, state: GraphState) -> ExportNode | End[ExtractionResult]:
        try:
            result = json.loads(state.raw_response)
            officers = []
            
            for officer_data in result["officers"]:
                citations = [
                    Citation(
                        page_number=cite["page_number"],
                        text_snippet=cite["text_snippet"],
                        confidence_score=cite["confidence_score"],
                    )
                    for cite in officer_data["citations"]
                ]

                officer = Officer(
                    name=officer_data["name"],
                    age=officer_data["age"],
                    title=officer_data["title"],
                    citations=citations,
                    source_document=state.pdf_path,
                )
                officers.append(officer)

            extraction_result = ExtractionResult(
                officers=officers,
                raw_response=state.raw_response,
                extraction_timestamp=state.extracted_at
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
            if not state.extraction_result or not state.extraction_result.officers:
                raise ValueError("No officers data to export")

            # Get CSV-friendly rows
            rows = [officer.to_csv_row() for officer in state.extraction_result.officers]

            # Write to CSV
            with open(state.output_path, "w", newline="", encoding="utf-8") as csvfile:
                fieldnames = [
                    "Name", "Age", "Title", "Source_Document", "Extracted_At",
                    "Page_Numbers", "Text_Snippets", "Average_Confidence",
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            print(f"\nExported officers data to: {state.output_path}")
            return End(state.extraction_result)
        except Exception as e:
            raise Exception(f"Error in CSV export: {str(e)}")