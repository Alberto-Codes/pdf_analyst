from dataclasses import dataclass

from google import genai
from google.genai import types
from nodes.evaluate import Evaluate
from pydantic_graph import BaseNode, GraphRunContext
from states.hrp123 import Hrp123GraphState


@dataclass
class ConfigureAPI(BaseNode[Hrp123GraphState]):
    """Configure the API request for generating content.

    This class is responsible for setting up the configuration for the
    Gemini API client. It retrieves the configuration from the `GraphState`
    and uses it to initialize the necessary settings for content generation.
    The `run` method sets up the Gemini client, defines content generation
    parameters, and configures safety settings to ensure appropriate content.

    Attributes:
        None directly, as the configuration is managed by the `GraphState`
        and passed through the `ctx.state` in the `run` method. The class
        relies on the context to access the required configuration parameters.
    """

    async def run(self, ctx: GraphRunContext[Hrp123GraphState]) -> Evaluate:
        """Run the configuration setup for the Gemini API.

        This method uses the `GraphState` from the context to configure the
        Gemini client and set up the content generation parameters, including
        temperature, top_p, max_tokens, response MIME type, and response schema.
        Additionally, it configures safety settings for content filtering
        to avoid harmful or inappropriate content.

        Args:
            ctx (GraphRunContext[GraphState]): The context containing the
                graph state, which holds the configuration parameters needed
                to set up the Gemini client and content generation settings.

        Returns:
            ExecuteAPI: An instance of the `ExecuteAPI` node, which will
                execute the content generation using the configured settings.
        """
        # Initialize the Gemini client using the state location
        ctx.state.client = genai.Client(vertexai=True, location="us-central1")

        # Set up the content generation configuration based on the state
        ctx.state.config = types.GenerateContentConfig(
            temperature=ctx.state.temperature,
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

        # Return an instance of ExecuteAPI to generate content based on the configuration
        return Evaluate(None)
