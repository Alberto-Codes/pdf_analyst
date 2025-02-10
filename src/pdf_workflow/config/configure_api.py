from dataclasses import dataclass

from config.state import GraphState
from execution.execute_api import ExecuteAPI
from google import genai
from google.genai import types
from pydantic_graph import BaseNode, GraphRunContext


@dataclass
class ConfigureAPI(BaseNode[GraphState]):
    """Configure the API request for generating content.

    This class is responsible for setting up the configuration for the
    Gemini API client. It uses the `GraphState` to initialize the necessary
    configurations for content generation. The `run` method configures the
    Gemini client and content generation parameters.

    Attributes:
        None directly, as all configuration is passed via the `GraphState`
        in the `run` method. The class relies on the context (`ctx.state`)
        to access the necessary parameters for the configuration.
    """

    async def run(self, ctx: GraphRunContext[GraphState]) -> ExecuteAPI:
        """Run the configuration setup for the Gemini API.

        Uses the `GraphState` from the context to set up the Gemini client
        and configure the content generation parameters. This includes
        temperature, top_p, and max_tokens, as well as safety settings for
        content filtering.

        Args:
            ctx (GraphRunContext[GraphState]): The context containing the
                graph state, including the parameters needed for configuration.

        Returns:
            ExecuteAPI: An instance of the `ExecuteAPI` node, which will perform
                the content generation based on the configured settings.
        """
        # Initialize the Gemini client with the state location
        ctx.state.client = genai.Client(vertexai=True, location=ctx.state.location)

        # Set up the content generation configuration
        ctx.state.config = types.GenerateContentConfig(
            temperature=ctx.state.temperature,
            top_p=ctx.state.top_p,
            max_output_tokens=ctx.state.max_tokens,
            response_modalities=["TEXT"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT", threshold="OFF"
                ),
            ],
            response_mime_type="application/json",
        )
        # Return an instance of ExecuteAPI to perform content generation
        return ExecuteAPI()
