from agno.agent import Agent
from agno.models.ollama import Ollama
import sys
import urllib.parse
import re
from docling.document_converter import DocumentConverter
import requests
import tempfile
import os
import json
from typing import Type, Any
from pydantic import BaseModel, Field

# Define the structured data models
class Director(BaseModel):
    name: str = Field(..., description="Director's full name")
    age: int = Field(None, description="Director's age (if available)")
    title: str = Field(..., description="Director's title")

class Directors(BaseModel):
    directors: list[Director]

def generate_template(model_cls: Type[BaseModel]) -> str:
    """Generate a JSON template string from a Pydantic V2 model class."""
    def recurse(cls: Type[BaseModel]) -> Any:
        if hasattr(cls, "model_fields"):
            result = {}
            for field_name, field in cls.model_fields.items():
                # Get the type annotation
                field_type = field.annotation
                
                # Handle Optional types
                if hasattr(field_type, "__origin__") and field_type.__origin__ is list:
                    if hasattr(field_type.__args__[0], "model_fields"):
                        # It's a list of BaseModels
                        result[field_name] = [recurse(field_type.__args__[0])]
                    else:
                        # It's a list of simple types
                        result[field_name] = [field_name]
                elif hasattr(field_type, "model_fields"):
                    # It's a nested BaseModel
                    result[field_name] = recurse(field_type)
                else:
                    # It's a simple type
                    result[field_name] = field_name
            return result
        return str(cls)
    
    template_dict = recurse(model_cls)
    return json.dumps(template_dict, indent=2)

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

        # Generate template for structured data extraction
        template = generate_template(Directors)
        print(f"\nUsing template for extraction:\n{template}\n")
        
        # Initialize agent for PDF content extraction
        agent = Agent(
            model=Ollama(id="llama3.2-3b-instruct-fp16-32k"),
            description="You are a precise document analyzer skilled at extracting and structuring information from PDFs. Focus on identifying director information and organizing it according to the provided template.",
            markdown=True
        )
        
        print(f"Processing document from URL: {record}")
        extraction_prompt = f"""Extract the directors listed information from the text into the following JSON template exactly:
{template}
Do not output any extra text.

Text:
{extracted_text}"""
        
        agent.print_response(extraction_prompt, stream=True)

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