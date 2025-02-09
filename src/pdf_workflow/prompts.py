from typing import Dict

from google.genai import types


class PromptTemplate:
    """Handles prompt generation and formatting for PDF extraction."""

    OFFICER_EXTRACTION_TEMPLATE = """
    Extract officers' information and provide detailed citations.
    Format the response as a JSON object with the following structure:
    {
        "officers": [
            {
                "name": "string",
                "age": "string",
                "title": "string",
                "citations": [
                    {
                        "page_number": number,
                        "text_snippet": "string",
                        "confidence_score": number
                    }
                ]
            }
        ]
    }
    
    For each citation:
    - Include the page number where the information was found
    - Include a brief text snippet from the page (max 100 chars)
    - Provide a confidence score (0.0-1.0) for the citation
    
    Use empty string '' for missing values in name, age, or title.
    """

    @classmethod
    def create_extraction_content(cls, document: types.Part) -> list[types.Content]:
        """Create content for officer extraction."""
        return [
            types.Content(
                role="user",
                parts=[
                    document,
                    types.Part.from_text(text=cls.OFFICER_EXTRACTION_TEMPLATE),
                ],
            )
        ]

    @classmethod
    def customize_template(cls, template: str, parameters: Dict[str, str]) -> str:
        """Allow customization of templates with parameters."""
        return template.format(**parameters)
