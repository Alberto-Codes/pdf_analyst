from pdf_workflow.nodes.configure_api import ConfigureAPI
from pdf_workflow.nodes.create_prompt import CreatePrompt
from pdf_workflow.nodes.execute_api import ExecuteAPI
from pdf_workflow.nodes.export import ExportToCSV
from pdf_workflow.nodes.print_response import PrintResponse
from pdf_workflow.nodes.sanitize_prompt import SanitizePrompt
from pydantic_graph import Graph

"""Defines the Gemini API execution graph.

This creates an instance of a `Graph` that represents the sequence of nodes 
for processing content generation. The nodes are responsible for configuring 
the API, executing the API request, printing the response, and exporting 
the result to a CSV file.
"""

# Create the Gemini graph with sanitization
gemini_graph = Graph(
    nodes=[
        ConfigureAPI,
        CreatePrompt,
        SanitizePrompt,  # Add sanitization step
        ExecuteAPI,
        PrintResponse,
        ExportToCSV
    ]
)

