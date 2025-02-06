import os
import tempfile

import requests


class DocumentRecord:
    def __init__(self, metadata: dict, status: str = "pending"):
        self.metadata = metadata
        self.status = status

    def __str__(self):
        return f"DocumentRecord(status={self.status}, metadata={self.metadata})"


def download_pdf(url: str) -> str:
    """Download PDF from URL and save to temporary file."""
    response = requests.get(url)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(response.content)
        return tmp_file.name


def cleanup_file(filepath: str):
    """Remove temporary files after processing."""
    if filepath and os.path.exists(filepath):
        os.remove(filepath)
