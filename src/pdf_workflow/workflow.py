from __future__ import annotations

import base64
import json

from google import genai
from google.api_core import retry
from google.genai import types
from models import Citation, ExtractionResult, Officer


class GeminiPDFParser:
    """A class to handle PDF parsing using Google's Gemini AI model."""

    def __init__(
        self,
        location: str = "us-central1",
        model: str = "gemini-2.0-flash-001",
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 8192,
    ):
        self.client = genai.Client(vertexai=True, location=location)
        self.model = model
        self.config = self._create_generate_config(temperature, top_p, max_tokens)

    def _create_generate_config(
        self, temperature: float, top_p: float, max_tokens: int
    ) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
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

    def _encode_pdf(self, pdf_path: str) -> str:
        try:
            with open(pdf_path, "rb") as file:
                pdf_data = file.read()
                return base64.b64encode(pdf_data).decode("utf-8")
        except FileNotFoundError:
            raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")
        except IOError as e:
            raise IOError(f"Error reading PDF file: {str(e)}")

    @retry.Retry(predicate=retry.if_transient_error)
    def extract_officers(self, pdf_path: str) -> ExtractionResult:
        """Extract officer information from a PDF document."""
        try:
            encoded_pdf = self._encode_pdf(pdf_path)

            document = types.Part.from_bytes(
                data=encoded_pdf,
                mime_type="application/pdf",
            )

            prompt = """
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

            contents = [
                types.Content(
                    role="user",
                    parts=[document, types.Part.from_text(text=prompt)],
                )
            ]

            response_text = ""
            for chunk in self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=self.config,
            ):
                response_text += chunk.text

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
                        source_document=pdf_path,
                    )
                    officers.append(officer)

                return ExtractionResult(officers=officers, raw_response=response_text)

            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"Failed to parse response as JSON: {str(e)}")

        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")


def main():
    """Example usage showing database-ready output and CSV export."""
    try:
        parser = GeminiPDFParser()
        result = parser.extract_officers("data/10k.pdf")

        # Export to CSV
        output_file = "data/officers_export.csv"
        result.export_to_csv(output_file)
        print(f"\nExported officers data to: {output_file}")

        # Preview the data
        print("\nExtracted Officers Preview:")
        for officer in result.officers:
            print(f"\n{officer.name} - {officer.title}")
            print("Citations:")
            for citation in officer.citations:
                print(f"  Page {citation.page_number}: {citation.text_snippet}")
                print(f"  Confidence: {citation.confidence_score:.2f}")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
