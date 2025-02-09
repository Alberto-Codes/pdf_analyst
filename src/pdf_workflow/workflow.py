from __future__ import annotations

import json

from config import GeminiConfig
from google.api_core import retry
from google.genai import types
from models import Citation, ExtractionResult, Officer
from prompts import PromptTemplate
from utils import encode_pdf


class GeminiPDFParser:
    """A class to handle PDF parsing using Google's Gemini AI model."""

    def __init__(self, config: GeminiConfig = None):
        self.config = config or GeminiConfig()
        self.client = self.config.create_client()
        self.model = self.config.model
        self.generate_config = self.config.create_generate_config()

    @retry.Retry(predicate=retry.if_transient_error)
    def extract_officers(self, pdf_path: str) -> ExtractionResult:
        """Extract officer information from a PDF document."""
        try:
            encoded_pdf = encode_pdf(pdf_path)  # Using the utility function
            document = types.Part.from_bytes(
                data=encoded_pdf,
                mime_type="application/pdf",
            )

            contents = PromptTemplate.create_extraction_content(document)

            response_text = ""
            for chunk in self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=self.generate_config,
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
