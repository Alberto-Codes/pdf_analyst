import sys
from utils.validation import is_valid_url
from utils.file_utils import download_pdf, cleanup_file
from models.director_models import Directors
from processing.document_processing import extract_text
from agents.extraction_agent import process_document
from utils.file_utils import DocumentRecord

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

        print("Extracting text from PDF...")
        extracted_text = extract_text(pdf_path)
        record.metadata["extracted_text"] = extracted_text
        record.status = "text_extracted"

        process_document(extracted_text, Directors)

    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        record.status = "error"
    finally:
        cleanup_file(record.metadata.get("local_path"))

if __name__ == "__main__":
    main()
