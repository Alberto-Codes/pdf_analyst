from dataclasses import dataclass

from google import genai
from google.genai import types


@dataclass
class GeminiConfig:
    """Configuration settings for Gemini AI model."""

    location: str = "us-central1"
    model: str = "gemini-2.0-flash-001"
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 8192

    def create_client(self) -> genai.Client:
        """Create and return a configured Gemini client."""
        return genai.Client(vertexai=True, location=self.location)

    def create_generate_config(self) -> types.GenerateContentConfig:
        """Create and return generation configuration."""
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
