from agno.agent import Agent
from agno.models.ollama import Ollama
import sys
import urllib.parse
import re
from docling.document_converter import DocumentConverter
import requests
import tempfile
import os

class DocumentRecord:
    def __init__(self, metadata: dict, status: str = "pending"):
        self.metadata = metadata
        self.status = status
    
    def __str__(self):
        return f"DocumentRecord(status={self.status}, metadata={self.metadata})"

def is_valid_url(url: str) -> bool:
    """Check if the provided string is a valid URL."""
    try:
        result = urllib.parse.urlparse(url)
        is_valid = all([result.scheme, result.netloc])
        is_pdf = url.lower().endswith('.pdf')
        return is_valid and is_pdf
    except:
        return False

def download_pdf(url: str) -> str:
    """Download PDF from URL and save to temporary file."""
    response = requests.get(url)
    response.raise_for_status()
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(response.content)
        return tmp_file.name

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <pdf_url>")
        sys.exit(1)
    
    pdf_url = sys.argv[1]
    
    if not is_valid_url(pdf_url):
        print("Error: Please provide a valid PDF URL")
        sys.exit(1)
        
    # Create a document record with the PDF URL
    record = DocumentRecord(metadata={
        "url": pdf_url,
        "source_type": "url"
    })
    
    try:
        # Download PDF to temporary file
        print("Downloading PDF...")
        pdf_path = download_pdf(pdf_url)
        record.metadata["local_path"] = pdf_path
        
        # Extract text from PDF
        print("Extracting text from PDF...")
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        extracted_text = result.document.export_to_markdown() if hasattr(result, 'document') else ""
        
        # Store extracted text in record
        record.metadata["extracted_text"] = extracted_text
        record.status = "text_extracted"
        
        # Initialize agent for PDF content extraction
        agent = Agent(
            model=Ollama(id="llama3.2"),
            description="You are a precise document analyzer skilled at extracting and summarizing information from PDFs. Focus on identifying key details, main topics, and important data points.",
            markdown=True
        )
        
        print(f"Processing document from URL: {record}")
        agent.print_response(f"Please describe the main topics and key information found in this text:\n\n{extracted_text}", stream=True)

    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        record.status = "error"
    finally:
        # Cleanup temporary file
        if "local_path" in record.metadata:
            try:
                os.unlink(record.metadata["local_path"])
            except:
                pass

if __name__ == "__main__":
    main()