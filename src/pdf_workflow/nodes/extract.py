from __future__ import annotations

from dataclasses import dataclass

from base_types import ExtractionTemplate
from config import GeminiConfig
from google.genai import types
from models import ExtractionResult
from nodes.base import BaseNode, End, GraphState
from nodes.parse import ParseNode
from prompts import PromptTemplate
from pydantic_graph import GraphRunContext
from utils import encode_file


@dataclass
class ExtractNode(BaseNode):
    """Generic node for extracting entities with citations."""

    config: GeminiConfig
    template: ExtractionTemplate

    async def run(
        self, ctx: GraphRunContext[GraphState]
    ) -> ParseNode | End[ExtractionResult]:
        try:

            encoded_file = encode_file(
                ctx.state.document_path, encoding=ctx.state.document_config.encoding
            )

            document = types.Part.from_bytes(
                data=encoded_file,
                mime_type=ctx.state.document_config.mime_type,
            )

            contents = PromptTemplate.create_extraction_content(
                document, self.template.get_prompt()
            )

            response_text = ""
            if ctx.state.document_config.stream_response:
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

            # ✅ Assign extracted data back into ctx.state
            ctx.state.raw_response = response_text
            ctx.state.field_order = self.template.field_order

            return ParseNode(
                entity_type=self.template.entity_type,
                entity_key=self.template.entity_name.lower() + "s",
                template=self.template,
            )
        except Exception as e:
            raise Exception(f"Error in extraction: {str(e)}")
