from dataclasses import dataclass

from config.state import GraphState
from nodes.print_response import PrintResponse
from pydantic_graph import BaseNode, GraphRunContext


@dataclass
class ExecuteAPI(BaseNode[GraphState]):
    """Executes the Gemini API request for content generation.

    This node sends the prepared contents to the Gemini API and stores the
    generated response in `GraphState`.
    """

    async def run(self, ctx: GraphRunContext[GraphState]) -> PrintResponse:
        """Executes the API request to generate content.

        This method sends the compiled content to the Gemini API using the
        configured client and parameters. The generated response is then
        stored in `GraphState` for subsequent processing.

        Args:
            ctx (GraphRunContext[GraphState]): The execution context containing
                `GraphState`, which includes the API client, model name, and
                prepared contents.

        Returns:
            PrintResponse: The next node responsible for handling the response.
        """
        # Generate content using the Gemini client
        response = ctx.state.client.models.generate_content(
            model=ctx.state.model,
            contents=ctx.state.contents,
            config=ctx.state.config,
        )

        # Store the generated response in state
        ctx.state.response_text = response.text

        return PrintResponse()
