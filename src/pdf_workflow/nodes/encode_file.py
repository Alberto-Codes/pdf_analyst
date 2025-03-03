from dataclasses import dataclass
import os
from pathlib import Path
import logging
import time
import asyncio
from typing import Dict, Any, Optional

from pydantic_graph import BaseNode, GraphRunContext

from pdf_workflow.config.state import GraphState
from pdf_workflow.nodes.create_prompt import CreatePrompt


@dataclass
class EncodeFileNode(BaseNode[GraphState]):
    """Encodes a PDF file in a dedicated workflow node.
    
    This node is responsible for:
        1. Validating the file exists and is accessible
        2. Reading and encoding the file with appropriate strategy based on size
        3. Storing the encoded file content and metadata in GraphState
        4. Providing error handling for file-related issues
        
    This dedicated node isolates resource-intensive encoding operations from
    other processing steps, preparing the architecture for future asynchronous
    processing and better memory management during batch processing.
    """

    # File size threshold in MB for different encoding strategies
    SMALL_FILE_THRESHOLD_MB: float = 10.0
    
    # Timeout for encoding operations in seconds
    ENCODING_TIMEOUT_SECONDS: float = 60.0
    
    async def run(self, ctx: GraphRunContext[GraphState]) -> CreatePrompt:
        """Encodes a PDF file and stores the encoded content in GraphState.
        
        This method validates and reads the file from the path specified in
        GraphState, chooses an appropriate encoding strategy based on file size,
        and stores the encoded content in the state for subsequent nodes to use.
        
        Args:
            ctx (GraphRunContext[GraphState]): The execution context containing
                the shared state with document_path and other configuration.
                
        Returns:
            CreatePrompt: The next node for prompt creation.
        """
        # Initialize encoding metrics for tracking
        ctx.state.encoding_metrics = {
            "file_size_bytes": 0,
            "encoding_strategy": "none",
            "success": False,
            "errors": None,
            "duration_ms": 0,
            "start_time": time.time(),
            "end_time": 0,
            "timeout_seconds": self.ENCODING_TIMEOUT_SECONDS,
            "cancelled": False
        }
        
        # Skip if no document path provided
        if not ctx.state.document_path:
            logging.info("No document path provided, skipping file encoding")
            return CreatePrompt()
        
        # Validate file existence
        filepath = Path(ctx.state.document_path)
        if not filepath.exists():
            error_msg = f"File not found: {filepath}"
            logging.error(f"Error: {error_msg}")
            
            ctx.state.error = {
                "type": "file_error",
                "message": "File not found",
                "details": error_msg
            }
            ctx.state.encoding_metrics["errors"] = "file_not_found"
            self._finish_encoding_metrics(ctx.state.encoding_metrics)
            return CreatePrompt()
            
        # Check file size and choose encoding strategy
        file_size_bytes = filepath.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        ctx.state.encoding_metrics["file_size_bytes"] = file_size_bytes
        
        logging.info(f"Encoding file: {filepath.name} (Size: {file_size_mb:.2f} MB)")
        
        try:
            # For smaller files, read all at once
            if file_size_mb < self.SMALL_FILE_THRESHOLD_MB:
                ctx.state.encoding_metrics["encoding_strategy"] = "single_read"
                
                # Use asyncio.wait_for to implement timeout
                try:
                    file_bytes = await asyncio.wait_for(
                        self._encode_small_file(filepath), 
                        timeout=self.ENCODING_TIMEOUT_SECONDS
                    )
                except asyncio.TimeoutError:
                    ctx.state.encoding_metrics["cancelled"] = True
                    raise TimeoutError(f"Encoding operation timed out after {self.ENCODING_TIMEOUT_SECONDS} seconds")
            else:
                # For larger files, use optimized approach with cancellation support
                ctx.state.encoding_metrics["encoding_strategy"] = "large_file"
                
                # Use asyncio.wait_for to implement timeout
                try:
                    file_bytes = await asyncio.wait_for(
                        self._encode_large_file(filepath), 
                        timeout=self.ENCODING_TIMEOUT_SECONDS
                    )
                except asyncio.TimeoutError:
                    ctx.state.encoding_metrics["cancelled"] = True
                    raise TimeoutError(f"Encoding operation timed out after {self.ENCODING_TIMEOUT_SECONDS} seconds")
                
            # Store encoded content in state
            ctx.state.encoded_file = {
                "bytes": file_bytes,
                "mime_type": ctx.state.document_mime_type,
                "size": len(file_bytes),
                "path": str(filepath)
            }
            
            ctx.state.encoding_metrics["success"] = True
            self._finish_encoding_metrics(ctx.state.encoding_metrics)
            logging.info(f"Successfully encoded {len(file_bytes)} bytes from {filepath.name} "
                         f"in {ctx.state.encoding_metrics['duration_ms']:.2f}ms")
            
        except Exception as e:
            error_msg = f"Failed to encode file: {str(e)}"
            logging.error(f"Error: {error_msg}")
            
            error_type = "encoding_timeout" if isinstance(e, TimeoutError) else "encoding_error"
            
            ctx.state.error = {
                "type": error_type,
                "message": "Failed to encode file",
                "details": error_msg
            }
            ctx.state.encoding_metrics["errors"] = str(e)
            self._finish_encoding_metrics(ctx.state.encoding_metrics)
            
        return CreatePrompt()
    
    async def _encode_small_file(self, filepath: Path) -> bytes:
        """Encodes a small file by reading it all at once.
        
        This method supports cancellation via asyncio timeout mechanisms.
        
        Args:
            filepath: Path to the file to encode
            
        Returns:
            The file content as bytes
        """
        # Wrap synchronous operation in an async operation that can be cancelled
        return await asyncio.to_thread(filepath.read_bytes)
    
    async def _encode_large_file(self, filepath: Path) -> bytes:
        """Encodes a large file with optimizations for future async processing.
        
        This method supports cancellation via asyncio timeout mechanisms.
        Currently still reads the whole file at once, but provides a hook for
        future implementation of chunked reading or other optimizations.
        
        Args:
            filepath: Path to the file to encode
            
        Returns:
            The file content as bytes
        """
        # TODO: Implement chunked reading for better memory efficiency
        # This is a placeholder for future optimization with cancellation support
        return await asyncio.to_thread(filepath.read_bytes)
    
    def _finish_encoding_metrics(self, metrics: Dict[str, Any]) -> None:
        """Updates the encoding metrics with timing information.
        
        Args:
            metrics: The metrics dictionary to update
        """
        metrics["end_time"] = time.time()
        metrics["duration_ms"] = (metrics["end_time"] - metrics["start_time"]) * 1000 