from nodes.configure_api import ConfigureAPI
from nodes.evaluate import Evaluate, ExecuteAPI
from nodes.export import ExportToCSV
from nodes.print_response import PrintResponse
from pydantic_graph import Graph, End

# Create the Gemini graph
gemini_graph = Graph(nodes=[ConfigureAPI, Evaluate, ExecuteAPI])

"""Defines the Gemini API execution graph.

This creates an instance of a `Graph` that represents the sequence of nodes 
for processing content generation. The nodes are responsible for configuring 
the API, executing the API request, printing the response, and exporting 
the result to a CSV file.

Attributes:
    gemini_graph (Graph): The graph object that defines the node sequence 
        for generating, processing, and exporting content from the Gemini API.
"""
