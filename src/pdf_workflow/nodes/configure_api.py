from dataclasses import dataclass

from config.state import GraphState
from google import genai
from google.genai import types
from nodes.create_prompt import CreatePrompt
from pydantic_graph import BaseNode, GraphRunContext


@dataclass
class ConfigureAPI(BaseNode[GraphState]):
    """Configures the Gemini API client for content generation.

    This node initializes the Gemini API client and sets up the content
    generation parameters using values from `GraphState`. It also configures
    safety settings to filter out inappropriate content.

    Attributes:
        None directly, as all configurations are managed via `GraphState`
        and accessed through `ctx.state` during execution.
    """

    async def run(self, ctx: GraphRunContext[GraphState]) -> CreatePrompt:
        """Sets up the Gemini API client and generation parameters.

        This method initializes the Gemini client and configures the content
        generation settings, including temperature, top_p, response format,
        and safety settings. These configurations are stored in `GraphState`
        for later use in content generation.

        Args:
            ctx (GraphRunContext[GraphState]): The execution context containing
                `GraphState`, which holds configuration parameters.

        Returns:
            CreatePrompt: The next node responsible for prompt creation.
        """
        # Initialize the Gemini API client
        ctx.state.client = genai.Client(vertexai=True, location=ctx.state.location)

        # Configure content generation settings
        ctx.state.config = types.GenerateContentConfig(
            temperature=ctx.state.temperature,
            top_p=ctx.state.top_p,
            # max_output_tokens=ctx.state.max_tokens,
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
            response_mime_type=ctx.state.response_mime_type,
            response_schema=ctx.state.response_schema,
        )

        # Proceed to the next node for prompt creation
        return CreatePrompt()
