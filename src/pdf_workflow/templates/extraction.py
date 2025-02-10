"""Template definitions for extraction prompts."""

from typing import List

from pydantic import BaseModel


class ExtractionTemplate(BaseModel):
    """Base template for extraction prompts."""

    entity_name: str
    entity_type: str
    fields: List[str]
    is_singular: bool = False

    def get_prompt(self) -> str:
        """Generate appropriate prompt based on entity type."""
        fields_json = ", ".join(f'"{field}": "string"' for field in self.fields)
        entity_key = self.entity_name.lower()
        plural_suffix = "" if self.is_singular else "s"

        return f"""
        Extract {self.entity_name} information and provide detailed citations.
        
        Format the response as a JSON object with the following structure:
        {{
            "{entity_key}{plural_suffix}": [
                {{
                    {fields_json},
                    "citations": [
                        {{
                            "page_number": number,
                            "text_snippet": "string",
                            "confidence_score": number
                        }}
                    ]
                }}
            ]
        }}
        
        For each citation:
        - Include the page number where the information was found
        - Include a brief text snippet from the page (max 100 chars)
        - Provide a confidence score (0.0-1.0) for the citation
        
        Use empty string '' for missing values in any field.
        """
