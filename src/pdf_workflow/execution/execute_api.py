from dataclasses import dataclass

from execution.print_response import PrintResponse
from google import genai
from google.genai import types
from pydantic_graph import BaseNode, GraphRunContext


@dataclass
class ExecuteAPI(BaseNode[None, None, str]):
    """Execute the content generation API request.

    This class sends the prompt and configuration to the Gemini API
    to generate content, using the provided client and configuration.

    Attributes:
        prompt (str): The prompt for content generation.
        client (genai.Client): The Gemini API client used for sending requests.
        config (types.GenerateContentConfig): The configuration for generating content.
    """

    prompt: str
    client: genai.Client
    config: types.GenerateContentConfig

    async def run(self, ctx: GraphRunContext) -> "PrintResponse":
        """Run the content generation API request.

        Sends the content generation request to the Gemini API with the
        provided prompt and configuration. The API response is then passed
        to the next node.

        Args:
            ctx (GraphRunContext): The context in which the graph is running.

        Returns:
            PrintResponse: An instance of the PrintResponse node to
            print the response from the API.
        """
        response = self.client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=[types.Part.from_text(text=self.prompt)],
            config=self.config,
        )
        return PrintResponse(response=response.text)
