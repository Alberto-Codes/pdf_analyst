from __future__ import annotations

from config import GeminiConfig
from google.api_core import retry
from google.genai import types
from handlers import ResponseHandler
from models import ExtractionResult
from prompts import PromptTemplate
from utils import encode_pdf


class GeminiPDFParser:
    def __init__(self, config: GeminiConfig = None):
        self.config = config or GeminiConfig()
        self.client = self.config.create_client()
        self.model = self.config.model
        self.generate_config = self.config.create_generate_config()

    @retry.Retry(predicate=retry.if_transient_error)
    def extract_officers(self, pdf_path: str) -> ExtractionResult:
        """Extract officer information from a PDF document."""
        try:
            # Prepare document
            encoded_pdf = encode_pdf(pdf_path)
            document = types.Part.from_bytes(
                data=encoded_pdf,
                mime_type="application/pdf",
            )

            # Get response
            contents = PromptTemplate.create_extraction_content(document)
            response_text = ""
            for chunk in self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=self.generate_config,
            ):
                response_text += chunk.text

            # Parse response
            return ResponseHandler.parse_response(
                response_text=response_text, source_document=pdf_path
            )

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
