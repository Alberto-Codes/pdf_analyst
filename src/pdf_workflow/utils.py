import base64
from pathlib import Path


def encode_pdf(pdf_path: str) -> str:
    """
    Read and encode a PDF file to base64.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        str: Base64 encoded PDF content

    Raises:
        FileNotFoundError: If PDF file is not found
        IOError: If there's an error reading the file
    """
    try:
        with open(pdf_path, "rb") as file:
            pdf_data = file.read()
            return base64.b64encode(pdf_data).decode("utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")
    except IOError as e:
        raise IOError(f"Error reading PDF file: {str(e)}")
