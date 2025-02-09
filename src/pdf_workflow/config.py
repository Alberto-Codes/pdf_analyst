# src/pdf_workflow/config.py
from dataclasses import dataclass

from google import genai
from google.genai import types


@dataclass
class GeminiConfig:
    """Configuration for Gemini API client."""

    location: str = "us-central1"
    model: str = "gemini-2.0-flash-001"
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 8192

    def __post_init__(self):
        """Initialize the client after instance creation."""
        self.client = genai.Client(vertexai=True, location=self.location)
        self.generate_config = self._create_generate_config()

    def _create_generate_config(self) -> types.GenerateContentConfig:
        """Create configuration for content generation."""
        return types.GenerateContentConfig(
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
