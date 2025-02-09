from dataclasses import dataclass
from typing import Optional


@dataclass
class DocumentConfig:
    """Configuration for document processing."""

    mime_type: str = "application/pdf"
    encoding: str = "utf-8"
    chunk_size: Optional[int] = 1024 * 1024  # For potential streaming of large files
    stream_response: bool = True  # Whether to stream the API response

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.chunk_size is not None and self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive if specified")
