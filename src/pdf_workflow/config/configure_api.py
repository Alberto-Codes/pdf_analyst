from dataclasses import dataclass

from execution.execute_api import ExecuteAPI
from google import genai
from google.genai import types
from pydantic_graph import BaseNode, GraphRunContext


@dataclass
class ConfigureAPI(BaseNode[None, None, str]):
    """Configure the API request for generating content.

    This class is responsible for setting up the configuration for the
    Gemini API client, including parameters like temperature, top_p,
    max_tokens, and safety settings. It initializes the necessary
    configurations for content generation.

    Attributes:
        prompt (str): The prompt to send to the API for content generation.
        location (str): The location of the API client (default is "us-central1").
        model (str): The model to use for content generation (default is "gemini-2.0-flash-001").
        temperature (float): Controls randomness in the generation (default is 0.7).
        top_p (float): Probability distribution for sampling (default is 0.95).
        max_tokens (int): The maximum number of tokens to generate (default is 8192).
    """

    prompt: str
    location: str = "us-central1"
    model: str = "gemini-2.0-flash-001"
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 8192

    async def run(self, ctx: GraphRunContext) -> "ExecuteAPI":
        """Run the configuration setup for the API.

        Initializes a Gemini client and sets up the generation configuration
        for the API based on the instance's attributes.

        Args:
            ctx (GraphRunContext): The context in which the graph is running.

        Returns:
            ExecuteAPI: An instance of the ExecuteAPI node to perform the
            content generation.
        """
        client = genai.Client(vertexai=True, location=self.location)
        generate_config = types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_tokens,
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
        return ExecuteAPI(prompt=self.prompt, client=client, config=generate_config)
