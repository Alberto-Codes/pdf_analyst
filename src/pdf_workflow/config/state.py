from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict


class GraphState(BaseModel):
    """State for the Gemini content generation workflow.

    This class holds the state necessary for content generation using the
    Gemini API. It stores the prompt, model information, response text,
    the Gemini client, and the content generation configuration. This state
    object can be used to pass information between the different stages of
    the content generation process.

    Attributes:
        prompt (str | None): The prompt to be sent to the API for content
            generation. Defaults to `None`.
        model (str): The model used for content generation. Defaults to
            "gemini-2.0-flash-001".
        location (str): The location for the Gemini API client. Defaults to
            "us-central1".
        temperature (float): Controls the randomness in content generation.
            Defaults to 0.7.
        top_p (float): Controls the probability distribution for sampling.
            Defaults to 0.95.
        max_tokens (int): The maximum number of tokens to generate. Defaults
            to 8192.
        response_text (str): The text response from the API once content
            generation is complete. Defaults to an empty string.
        client (genai.Client | None): The Gemini API client used to make
            requests. Defaults to `None`.
        config (types.GenerateContentConfig | None): The configuration for
            generating content, including parameters like temperature, top_p,
            and max_tokens. Defaults to `None`.

    Configuration:
        model_config (ConfigDict): Allows arbitrary types to be used in the
            model, enabling more flexibility in the data model.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt: str | None = None
    model: str = "gemini-2.0-flash-001"
    location: str = "us-central1"
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 8192
    response_text: str = ""
    client: genai.Client | None = None
    config: types.GenerateContentConfig | None = None
