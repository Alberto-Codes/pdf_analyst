from pathlib import Path

from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict


class GraphState(BaseModel):
    """Represents the state for the Gemini content generation workflow.

    This class maintains the state required for generating content using the
    Gemini API. It stores details such as the prompt, model information,
    response data, Gemini API client, and content generation configurations.
    The state is passed through different nodes in the workflow, ensuring
    that all relevant parameters and responses are accessible throughout the
    process.

    Attributes:
        prompt (str | None): The input prompt to be sent to the Gemini API
            for content generation. Defaults to `None`.
        model (str): The model identifier used for content generation.
            Defaults to `"gemini-2.0-flash-001"`.
        location (str): The regional endpoint for the Gemini API client.
            Defaults to `"us-central1"`.
        temperature (float): Controls randomness in generated content.
            Higher values (e.g., 1.0) increase creativity, while lower
            values (e.g., 0.0) make responses more deterministic. Defaults to `0.7`.
        top_p (float): Probability distribution for sampling, used to control
            diversity in generated responses. Defaults to `0.95`.
        max_tokens (int): The maximum number of tokens the API can generate
            in a single response. Defaults to `8192`.
        config (types.GenerateContentConfig | None): Configuration settings
            for content generation, including parameters such as `temperature`,
            `top_p`, and `max_tokens`. Defaults to `None`.
        client (genai.Client | None): The Gemini API client instance used to
            make requests. Defaults to `None`.
        response_schema (dict | None): Defines the expected structure of the
            API response, if applicable. Defaults to `None`.
        response_mime_type (str | None): The MIME type of the API response,
            typically `"application/json"`. Defaults to `None`.
        response_text (str): The generated response text received from the
            Gemini API. Defaults to an empty string.
        document_url (str | None): The URL of a document to be processed by
            the Gemini API. Defaults to `None`.
        document_mime_type (str): The MIME type of the document, typically
            `"application/pdf"` for PDF files. Defaults to `"application/pdf"`.
        export_dir (Path): The directory where exported response files will
            be stored. Defaults to `"data"`.
        export_file_name (str): The base filename for exported responses.
            Defaults to `"exported_response"`.

    Configuration:
        model_config (ConfigDict): Allows arbitrary types to be used in the
            model, enabling greater flexibility in handling different data types.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt: str | None = None
    model: str = "gemini-2.0-flash-001"
    location: str = "us-central1"
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 8192
    config: types.GenerateContentConfig | None = None
    client: genai.Client | None = None
    response_schema: dict | None = None
    response_mime_type: str | None = None
    response_text: str = ""
    document_url: str | None = None  # URL of the document to process
    document_mime_type: str = "application/pdf"  # Default MIME type for PDF
    export_dir: Path = Path("data")
    export_file_name: str = "exported_response"
