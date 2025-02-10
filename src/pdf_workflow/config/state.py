from dataclasses import dataclass
from google import genai
from google.genai import types

@dataclass
class GeminiState:
    """State for the Gemini content generation workflow.

    This class holds the state necessary for content generation using the
    Gemini API. It stores the prompt, model information, response text, 
    the Gemini client, and the content generation configuration.

    Attributes:
        prompt (str): The prompt to be sent to the API for content generation.
        model (str): The model used for content generation (default is 
            "gemini-2.0-flash-001").
        response_text (str): The text response from the API once the content 
            generation is completed (default is an empty string).
        client (genai.Client): The Gemini API client used to make requests 
            (default is None).
        config (types.GenerateContentConfig): The configuration for generating 
            content, including parameters like temperature, top_p, and 
            max_tokens (default is None).
    """

    prompt: str
    model: str = "gemini-2.0-flash-001"
    response_text: str = ""
    client: genai.Client = None 
    config: types.GenerateContentConfig = None
