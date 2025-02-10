"""Prompt templates for PDF extraction."""

from typing import Dict

from google.genai import types


class PromptTemplate:
    """Handles prompt generation and formatting for PDF extraction."""

    @classmethod
    def create_extraction_content(
        cls, document: types.Part, prompt_text: str
    ) -> list[types.Content]:
        """Create content for extraction."""
        return [
            types.Content(
                role="user",
                parts=[
                    document,
                    types.Part.from_text(text=prompt_text),
                ],
            )
        ]

    @classmethod
    def customize_template(cls, template: str, parameters: Dict[str, str]) -> str:
        """Allow customization of templates with parameters."""
        return template.format(**parameters)
