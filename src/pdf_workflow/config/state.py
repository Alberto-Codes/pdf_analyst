from pathlib import Path

from google import genai
from google.genai import types
from google.genai.types import Schema
from pydantic import BaseModel, ConfigDict


class GraphState(BaseModel):
    """Maintains the state for the Gemini content generation workflow.

    This class stores essential parameters for content generation, such as
    the prompt, model configuration, API client, response data, and document
    processing details. The state is shared across different nodes in the
    workflow to ensure consistency.

    Attributes:
        prompt (str | None): The input prompt for Gemini API. Defaults to `None`.
        model (str): The model identifier for content generation.
            Defaults to `"gemini-2.0-flash-001"`.
        location (str): The regional endpoint for Gemini API.
            Defaults to `"us-central1"`.
        temperature (float): Controls randomness in generated content.
            Higher values (e.g., 1.0) increase creativity, while lower values
            (e.g., 0.0) make responses more deterministic. Defaults to `0.4`.
        top_p (float): Sampling probability used to control response diversity.
            Defaults to `0.95`.
        config (types.GenerateContentConfig | None): Content generation
            configuration including `temperature` and `top_p`. Defaults to `None`.
        client (genai.Client | None): The Gemini API client instance.
            Defaults to `None`.
        response_schema (Schema | None): Defines the expected response structure.
            Defaults to `None`.
        response_mime_type (str | None): MIME type of the API response.
            Defaults to `None`.
        response_text (str): The generated response text. Defaults to an empty string.
        document_url (str | None): URL of the document to be processed.
            Defaults to `None`.
        document_mime_type (str): The document's MIME type.
            Defaults to `"application/pdf"`.
        export_dir (Path): Directory for storing exported response files.
            Defaults to `"data"`.
        export_file_name (str): Base filename for exported responses.
            Defaults to `"exported_response"`.
        contents (list[types.Part]): List of content parts for the API request.
            Defaults to an empty list.

    Configuration:
        model_config (ConfigDict): Allows arbitrary types in the model to
            enable flexibility in handling different data types.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt: str | None = None
    model: str = "gemini-2.0-flash-001"
    location: str = "us-central1"
    temperature: float = 0.4
    top_p: float = 0.95
    config: types.GenerateContentConfig | None = None
    client: genai.Client | None = None
    response_schema: Schema | None = None
    response_mime_type: str | None = None
    response_text: str = ""
    document_url: str | None = None
    document_mime_type: str = "application/pdf"
    export_dir: Path = Path("data")
    export_file_name: str = "exported_response"
    contents: list[types.Part] = []
