import json

from models import Citation, ExtractionResult, Officer


class ResponseHandler:
    """Handles parsing and conversion of Gemini API responses."""

    @staticmethod
    def parse_response(
        response_text: str, source_document: str = ""
    ) -> ExtractionResult:
        """
        Parse the raw response text into structured data.

        Args:
            response_text (str): Raw JSON response from Gemini
            source_document (str): Path to source PDF document

        Returns:
            ExtractionResult: Structured data result

        Raises:
            ValueError: If response cannot be parsed as valid JSON
        """
        try:
            result = json.loads(response_text)
            officers = []

            for officer_data in result["officers"]:
                citations = [
                    Citation(
                        page_number=cite["page_number"],
                        text_snippet=cite["text_snippet"],
                        confidence_score=cite["confidence_score"],
                    )
                    for cite in officer_data["citations"]
                ]

                officer = Officer(
                    name=officer_data["name"],
                    age=officer_data["age"],
                    title=officer_data["title"],
                    citations=citations,
                    source_document=source_document,
                )
                officers.append(officer)

            return ExtractionResult(officers=officers, raw_response=response_text)

        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Failed to parse response as JSON: {str(e)}")
