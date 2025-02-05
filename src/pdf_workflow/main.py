from agno.agent import Agent
from agno.models.ollama import Ollama
import sys
import urllib.parse
import re

# Simple document record class as first step
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
    
    # Initialize agent for PDF content extraction
    agent = Agent(
        model=Ollama(id="llama3.2"),
        description="You are a precise document analyzer skilled at extracting and summarizing information from PDFs. Focus on identifying key details, main topics, and important data points.",
        markdown=True
    )
    
    print(f"Processing document from URL: {record}")
    agent.print_response("Please describe the main topics and key information found in this PDF document.", stream=True)

if __name__ == "__main__":
    main()