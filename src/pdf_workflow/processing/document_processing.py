from docling.document_converter import DocumentConverter


def extract_text(pdf_path: str) -> str:
    """Extract text from a given PDF using DocumentConverter."""
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    return result.document.export_to_markdown() if hasattr(result, "document") else ""
