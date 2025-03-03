from dataclasses import dataclass
import os
import logging
import sys

from pdf_workflow.config.state import GraphState
from google import genai
from google.genai import types
from pdf_workflow.nodes.create_prompt import CreatePrompt
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
        # Check credentials file exists
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_path:
            if not os.path.isabs(creds_path):
                # Make absolute path if relative
                creds_path = os.path.abspath(creds_path)
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
            
            print(f"Using credentials file: {creds_path}")
            if not os.path.exists(creds_path):
                print(f"Error: Credentials file not found at {creds_path}")
                print("Cannot continue without valid credentials.")
                sys.exit(1)
        else:
            print("Error: GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
            print("Cannot continue without valid credentials.")
            sys.exit(1)

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

