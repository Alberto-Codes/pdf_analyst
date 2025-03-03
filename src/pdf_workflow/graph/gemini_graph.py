import os
from typing import Dict, Any

from pydantic_graph import Graph

from pdf_workflow.nodes.configure_api import ConfigureAPI
from pdf_workflow.nodes.create_prompt import CreatePrompt
from pdf_workflow.nodes.encode_file import EncodeFileNode
from pdf_workflow.nodes.execute_api import ExecuteAPI
from pdf_workflow.nodes.export import ExportToCSV
from pdf_workflow.nodes.print_response import PrintResponse
from pdf_workflow.nodes.sanitize_prompt import SanitizePrompt


# Default configuration for the Model Armor sanitizer
def get_sanitize_config() -> Dict[str, Any]:
    """Returns the configuration for the SanitizePrompt node."""
    return {
        "project_id": os.environ.get("MODEL_ARMOR_PROJECT_ID", "mock-project"),
        "location": os.environ.get("MODEL_ARMOR_LOCATION", "europe-west4"),
        "template_id": os.environ.get("MODEL_ARMOR_TEMPLATE_ID")
    }


"""Defines the Gemini API execution graph.

This creates an instance of a `Graph` that represents the sequence of nodes 
for processing content generation. The nodes are responsible for configuring 
the API, encoding the file, creating the prompt, executing the API request, 
printing the response, and exporting the result to a CSV file.
"""

# Create the Gemini graph with file encoding and sanitization
gemini_graph = Graph(
    nodes=[
        ConfigureAPI,      # Returns EncodeFileNode
        EncodeFileNode,    # Returns CreatePrompt
        CreatePrompt,      # Returns SanitizePrompt
        SanitizePrompt,    # Returns ExecuteAPI
        ExecuteAPI,        # Returns PrintResponse
        PrintResponse,     # Returns ExportToCSV
        ExportToCSV        # Returns End
    ]
)

