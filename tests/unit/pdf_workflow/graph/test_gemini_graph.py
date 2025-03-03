"""Integration tests for the Gemini workflow graph.

This module contains tests that verify the correct integration of various nodes
in the workflow graph, with a particular focus on the EncodeFileNode's interaction
with other nodes in the workflow.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import tempfile
import sys
from typing import Any, Dict, List, Optional

from pydantic_graph import GraphRunContext

from pdf_workflow.config.state import GraphState
from pdf_workflow.graph.gemini_graph import gemini_graph
from pdf_workflow.nodes.encode_file import EncodeFileNode
from pdf_workflow.nodes.create_prompt import CreatePrompt
from pdf_workflow.nodes.sanitize_prompt import SanitizePrompt


class MockGenAIClient:
    """Mock for the Google GenAI Client that passes type checking."""
    
    def __init__(self):
        # Mock the models property
        self.models = MagicMock()
        
        # Set up the generate_content method on models
        mock_response = MagicMock()
        mock_response.text = "Mocked API response"
        self.models.generate_content.return_value = mock_response


@pytest.fixture
def mock_test_file():
    """Creates a small test file for workflow testing."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as f:
        f.write(b"PDF test content for workflow" * 100)
        file_path = f.name
    
    yield file_path
    
    # Clean up the file after the test
    if os.path.exists(file_path):
        os.unlink(file_path)


@pytest.fixture
def mock_api_client():
    """Creates a mock for the Gemini API client that passes type checking."""
    with patch('google.genai.Client', return_value=MockGenAIClient()) as _:
        # Create an instance of our MockGenAIClient class
        mock_client = MockGenAIClient()
        
        # Patch the GraphState class to accept our mock client
        # This is a bit of a hack, but it allows us to test without modifying the GraphState class
        original_init = GraphState.__init__
        
        def patched_init(self, **kwargs):
            if 'client' in kwargs and isinstance(kwargs['client'], MockGenAIClient):
                # Convert the mock client to a regular MagicMock for the duration of the init call
                # This bypasses the type checking
                client = kwargs.pop('client')
                original_init(self, **kwargs)
                # Then restore the client after init
                self.client = client
            else:
                original_init(self, **kwargs)
        
        with patch.object(GraphState, '__init__', patched_init):
            yield mock_client
        

class TestGeminiGraph:
    """Tests for the Gemini workflow graph integration."""
    
    @pytest.mark.asyncio
    async def test_encode_file_integration(self, mock_test_file, mock_api_client):
        """Test that EncodeFileNode integrates correctly in the workflow."""
        # Create initial state with a test file
        state = GraphState(
            document_path=mock_test_file,
            document_mime_type="application/pdf",
            prompt="Test prompt",
            client=mock_api_client
        )
        
        # Create a context with environment variables for SanitizePrompt
        env_vars = {
            "MODEL_ARMOR_PROJECT_ID": "test-project",
            "MODEL_ARMOR_TEMPLATE_ID": "test-template"
        }
        
        # Mock components that would make external calls
        with patch.dict(os.environ, env_vars):
            # Mock SanitizePrompt to avoid calling the real Model Armor API
            with patch('pdf_workflow.nodes.sanitize_prompt.ModelArmorSanitizer.sanitize_prompt') as mock_sanitize:
                mock_sanitize.return_value = {
                    "sanitizationResult": {
                        "invocationResult": "SUCCESS",
                        "filterMatchState": "NO_MATCH_FOUND"
                    }
                }
                
                # Run the graph with the test state
                result = await gemini_graph.run(EncodeFileNode(), state=state)
        
        # Assert the workflow executed successfully
        assert result is not None
        assert result.state.error is None
        assert state.encoded_file is not None
        assert state.encoded_file["mime_type"] == "application/pdf"
        assert len(state.encoded_file["bytes"]) > 0
        assert state.encoding_metrics is not None
        assert state.encoding_metrics["success"] is True
        
        # Verify contents created properly for API call
        assert len(state.contents) == 2  # Document + Prompt
        
        # Verify response from mock API was stored
        assert state.response_text == "Mocked API response"
    
    @pytest.mark.asyncio
    async def test_workflow_with_missing_file(self, mock_api_client):
        """Test that workflow handles missing files gracefully."""
        # Create initial state with a non-existent file
        state = GraphState(
            document_path="/nonexistent/path/file.pdf",
            document_mime_type="application/pdf",
            prompt="Test prompt",
            client=mock_api_client
        )
        
        # Run the graph with the test state
        result = await gemini_graph.run(EncodeFileNode(), state=state)
        
        # Assert the workflow completed but with an error in the state
        assert result is not None
        assert result.state.error is not None
        assert state.error["type"] == "file_error"
        assert state.encoding_metrics is not None
        assert state.encoding_metrics["success"] is False 