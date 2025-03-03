"""Unit tests for the EncodeFileNode.

This module contains tests for the EncodeFileNode class, which is responsible 
for encoding PDF files in the workflow. Tests cover scenarios including:
- Processing small files (< 10MB)
- Processing large files (>= 10MB)
- Handling file validation errors
- Error handling for file encoding
- Metrics collection
"""

import os
import time
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import tempfile

from pydantic_graph import GraphRunContext

from pdf_workflow.config.state import GraphState
from pdf_workflow.nodes.encode_file import EncodeFileNode
from pdf_workflow.nodes.create_prompt import CreatePrompt


@pytest.fixture
def graph_state():
    """Creates a GraphState instance for testing."""
    return GraphState(
        document_path=None,
        document_mime_type="application/pdf"
    )


@pytest.fixture
def small_test_file():
    """Creates a small test file for encoding tests."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as f:
        # Write some content to make a small file (< 10MB)
        f.write(b"PDF content" * 1000)  # Just enough content for a small file
        file_path = f.name
    
    yield file_path
    
    # Clean up the file after the test
    if os.path.exists(file_path):
        os.unlink(file_path)


@pytest.fixture
def mock_large_file_path():
    """Creates a mock for a large file path without actually creating a large file."""
    # This avoids creating an actual large file by mocking the Path methods
    mock_path = MagicMock(spec=Path)
    mock_path.exists.return_value = True
    mock_path.stat.return_value.st_size = 15 * 1024 * 1024  # 15MB
    mock_path.name = "large_test_file.pdf"
    mock_path.__str__.return_value = "/path/to/large_test_file.pdf"
    return mock_path


class TestEncodeFileNode:
    """Tests for the EncodeFileNode class."""
    
    @pytest.mark.asyncio
    async def test_run_with_no_document_path(self, graph_state):
        """Test that node skips processing when no document path is provided."""
        # Arrange
        node = EncodeFileNode()
        ctx = GraphRunContext(state=graph_state, deps={})
        
        # Act
        result = await node.run(ctx)
        
        # Assert
        assert isinstance(result, CreatePrompt)
        assert ctx.state.encoding_metrics is not None
        assert ctx.state.encoding_metrics["encoding_strategy"] == "none"
        assert ctx.state.encoding_metrics["success"] is False
        assert ctx.state.encoded_file is None
    
    @pytest.mark.asyncio
    async def test_run_with_nonexistent_file(self, graph_state):
        """Test error handling when file does not exist."""
        # Arrange
        node = EncodeFileNode()
        graph_state.document_path = "/nonexistent/path/file.pdf"
        ctx = GraphRunContext(state=graph_state, deps={})
        
        # Act
        result = await node.run(ctx)
        
        # Assert
        assert isinstance(result, CreatePrompt)
        assert ctx.state.error is not None
        assert ctx.state.error["type"] == "file_error"
        assert ctx.state.encoding_metrics["errors"] == "file_not_found"
        assert ctx.state.encoded_file is None
    
    @pytest.mark.asyncio
    async def test_run_with_small_file(self, graph_state, small_test_file):
        """Test successful encoding of a small file."""
        # Arrange
        node = EncodeFileNode()
        graph_state.document_path = small_test_file
        ctx = GraphRunContext(state=graph_state, deps={})
        
        # Act
        result = await node.run(ctx)
        
        # Assert
        assert isinstance(result, CreatePrompt)
        assert ctx.state.encoded_file is not None
        assert ctx.state.encoded_file["mime_type"] == "application/pdf"
        assert len(ctx.state.encoded_file["bytes"]) > 0
        assert ctx.state.encoded_file["path"] == small_test_file
        assert ctx.state.encoding_metrics["success"] is True
        assert ctx.state.encoding_metrics["encoding_strategy"] == "single_read"
        assert ctx.state.encoding_metrics["duration_ms"] > 0
    
    @pytest.mark.asyncio
    async def test_run_with_large_file(self, graph_state, mock_large_file_path):
        """Test successful encoding of a large file."""
        # Arrange
        node = EncodeFileNode()
        graph_state.document_path = str(mock_large_file_path)
        ctx = GraphRunContext(state=graph_state, deps={})
        
        # Use patch to intercept calls to Path
        with patch('pdf_workflow.nodes.encode_file.Path', return_value=mock_large_file_path):
            # And patch the _encode_large_file method to return some bytes without reading a real file
            with patch.object(node, '_encode_large_file', new_callable=AsyncMock) as mock_encode:
                mock_encode.return_value = b"Large file content"
                
                # Act
                result = await node.run(ctx)
        
        # Assert
        assert isinstance(result, CreatePrompt)
        assert ctx.state.encoded_file is not None
        assert ctx.state.encoded_file["mime_type"] == "application/pdf"
        assert ctx.state.encoded_file["bytes"] == b"Large file content"
        assert ctx.state.encoding_metrics["success"] is True
        assert ctx.state.encoding_metrics["encoding_strategy"] == "large_file"
    
    @pytest.mark.asyncio
    async def test_run_with_file_read_error(self, graph_state, small_test_file):
        """Test error handling when file read fails."""
        # Arrange
        node = EncodeFileNode()
        graph_state.document_path = small_test_file
        ctx = GraphRunContext(state=graph_state, deps={})
        
        # Patch _encode_small_file to raise an exception
        with patch.object(node, '_encode_small_file', new_callable=AsyncMock) as mock_encode:
            mock_encode.side_effect = IOError("Mock file read error")
            
            # Act
            result = await node.run(ctx)
        
        # Assert
        assert isinstance(result, CreatePrompt)
        assert ctx.state.error is not None
        assert ctx.state.error["type"] == "encoding_error"
        assert "Mock file read error" in ctx.state.error["details"]
        assert ctx.state.encoding_metrics["success"] is False
        assert ctx.state.encoding_metrics["errors"] is not None
    
    @pytest.mark.asyncio
    async def test_metrics_calculation(self, graph_state, small_test_file):
        """Test that encoding metrics are calculated correctly."""
        # Arrange
        node = EncodeFileNode()
        graph_state.document_path = small_test_file
        ctx = GraphRunContext(state=graph_state, deps={})
        
        # Act
        with patch.object(time, 'time', side_effect=[100.0, 100.5]):  # Mock start and end times
            await node.run(ctx)
        
        # Assert
        assert ctx.state.encoding_metrics["start_time"] == 100.0
        assert ctx.state.encoding_metrics["end_time"] == 100.5
        assert ctx.state.encoding_metrics["duration_ms"] == 500.0  # 0.5 seconds = 500ms 