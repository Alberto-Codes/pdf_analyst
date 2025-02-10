from dataclasses import dataclass

from config.state import GraphState
from execution.print_response import PrintResponse
from google import genai
from google.genai import types
from pydantic_graph import BaseNode, GraphRunContext


@dataclass
class ExecuteAPI(BaseNode[GraphState]):
    """Execute the content generation API request.

    This class is responsible for sending the configured prompt and
    generation settings to the Gemini API. It uses the provided client
    and configuration to generate content. The response from the API is
    passed to the next node for further processing. The class supports
    both document-based and prompt-based content generation by adding
    the appropriate content parts from the `GraphState`.

    Attributes:
        None directly, as the configuration and prompt are passed through
        the `GraphState` in the `run` method. The class relies on the context
        (`ctx.state`) to access the necessary parameters for content generation.
    """

    async def run(self, ctx: GraphRunContext[GraphState]) -> "PrintResponse":
        """Run the content generation API request.

        This method sends the content generation request to the Gemini API,
        using the prompt and configuration from the `GraphState`. If a document
        URL is provided, it is included in the request; otherwise, only the
        prompt text is used. The response from the API is stored in the state
        and passed to the next node, which will print the response.

        Args:
            ctx (GraphRunContext[GraphState]): The context containing the
                graph state, which includes the prompt, client, document URL,
                and configuration.

        Returns:
            PrintResponse: An instance of the `PrintResponse` node, which
                will print the response text from the Gemini API.
        """
        contents = []

        # Add document from URL if provided
        if ctx.state.document_url:
            contents.append(
                types.Part.from_uri(
                    file_uri=ctx.state.document_url,
                    mime_type=ctx.state.document_mime_type,
                )
            )

        # Add prompt text if provided
        if ctx.state.prompt:
            contents.append(types.Part.from_text(text=ctx.state.prompt))

        # Generate content using the provided Gemini client and configuration
        response = ctx.state.client.models.generate_content(
            model=ctx.state.model,
            contents=contents,
            config=ctx.state.config,
        )

        # Store the generated response in the state
        ctx.state.response_text = response.text

        return PrintResponse()
