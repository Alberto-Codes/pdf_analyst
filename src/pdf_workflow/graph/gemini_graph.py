from config.configure_api import ConfigureAPI
from execution.execute_api import ExecuteAPI
from execution.print_response import PrintResponse
from pydantic_graph import Graph

# Create the Gemini graph
gemini_graph = Graph(nodes=[ConfigureAPI, ExecuteAPI, PrintResponse])
"""
This creates an instance of a `Graph` which represents the flow of nodes
for content generation. The nodes are responsible for configuring the 
API, executing the API request, and printing the response.

Attributes:
    gemini_graph (Graph): The graph object that defines the node sequence 
    for generating and printing content from the Gemini API.
"""
