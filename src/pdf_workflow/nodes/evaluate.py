from dataclasses import dataclass

from google.genai import types
from google.genai.types import GenerateContentResponse
from pydantic import BaseModel
from pydantic_graph import BaseNode, End, GraphRunContext
from states.hrp123 import Hrp123GraphState


@dataclass
class ExecuteAPI(BaseNode[Hrp123GraphState, None]):
    file_uri: str
    mime_type: str
    response_schema: BaseModel

    async def run(self, ctx: GraphRunContext[Hrp123GraphState]) -> "Evaluate":
        contents = []
        if self.file_uri:
            contents.append(
                types.Part.from_uri(
                    file_uri=str(self.file_uri),
                    mime_type=self.mime_type,
                )
            )
        contents.append(
            types.Part.from_text(text="""
You are a document extraction expert specialized in SEC filings. Analyze the provided document and extract information according to the schema tags provided. Focus on company details, officer information, and filing metadata.

For each extraction, provide:
- Complete context with citations for found information
- Standard null values when information isn't found (page: 0, context: "Information not found in document", confidence: 0.0)
- Strong confidence (>0.9) for exact matches of EINs, names, dates, and titles
- Location data (bbox) when available

Pay special attention to document sections typically containing:
- Company identifiers and legal names
- Officer signatures and titles
- Filing dates and attestations"""
            )
        )
        response = ctx.state.client.models.generate_content(
            model=ctx.state.model,
            contents=contents,
            config=ctx.state.config(response_schema=self.response_schema),
        )
        return Evaluate(response)


@dataclass
class Evaluate(BaseNode[Hrp123GraphState, None, None]):
    response: GenerateContentResponse

    async def run(self, ctx: GraphRunContext[Hrp123GraphState]) -> End | ExecuteAPI:
        if ctx.state.key_tag is None:

            return ExecuteAPI(
                file_uri=ctx.state.initial_document.file_uri._url,
                mime_type=ctx.state.initial_document.mime_type,
                response_schema=ctx.state.initial_document.ocr_tags,
            )
        elif self.response:
            print(self.response.text)
            return End()
        else:
            print("No response text")
            return End()
