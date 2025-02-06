import sys

from agents.extraction_agent import process_document
from agents.knowledge_base import get_knowledge
from models.director_models import Directors
from processing.document_processing import extract_text
from utils.file_utils import DocumentRecord, cleanup_file, download_pdf
from utils.validation import is_valid_url


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <pdf_url>")
        sys.exit(1)
    
    pdf_url = sys.argv[1]
    
    if not is_valid_url(pdf_url):
        print("Error: Please provide a valid PDF URL")
        sys.exit(1)

    record = DocumentRecord(metadata={"url": pdf_url, "source_type": "url"})

    try:
        print("Downloading PDF...")
        pdf_path = download_pdf(pdf_url)
        record.metadata["local_path"] = pdf_path

        print("Loading vector store...")
        knowledge = get_knowledge(pdf_path)

        print("Processing document via knowledge base...")
        process_document(knowledge, Directors)

    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        record.status = "error"
    finally:
        cleanup_file(record.metadata.get("local_path"))


if __name__ == "__main__":
    main()
