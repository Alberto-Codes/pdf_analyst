import logging
import os
import pathlib
import sys
import traceback
from typing import List, Optional

from pdf_workflow.config.prompt import prompt
from pdf_workflow.config.state import GraphState
from pdf_workflow.graph.gemini_graph import gemini_graph
from pdf_workflow.models.sec_filing import SecFiling
from pdf_workflow.nodes.configure_api import ConfigureAPI
from pdf_workflow.utils.schema_utils import create_vertex_schema


def configure_logging() -> None:
    """Configures the logging system for the application.
    
    Sets up logging with appropriate format, level, and handlers
    for console output and file logging.
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler
    try:
        # Get the project root directory
        current_dir = pathlib.Path(__file__).parent.resolve()
        project_root = current_dir.parent.parent
        
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(project_root, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # Add file handler
        file_handler = logging.FileHandler(
            os.path.join(logs_dir, "pdf_analyst.log"),
            mode='a'
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
    except Exception as e:
        logging.warning(f"Failed to set up file logging: {e}")


pydantic_schema = SecFiling.model_json_schema()
vertex_schema = create_vertex_schema(pydantic_schema)

if __name__ == "__main__":
    """Executes the Gemini graph synchronously with an initialized state.

    This script runs the `gemini_graph` synchronously, starting from the
    `ConfigureAPI` node. It initializes a `GraphState` object with necessary
    parameters, including the document URL, MIME type, prompt, and response schema.
    The execution of the graph results in content generation, with the generated
    response and execution history returned.

    The script extracts information from a PDF document using the Gemini API,
    focusing on retrieving employee count details.

    Process:
        1. Initializes `GraphState` with API parameters.
        2. Runs the Gemini graph synchronously, beginning with `ConfigureAPI`.
        3. Retrieves the generated response and execution history.

    Attributes:
        result (str): The generated content response from the Gemini API.
        history (list): The execution history of the graph, tracking all
            executed nodes in sequence.
    """
    # Configure logging
    configure_logging()
    
    # Get the absolute path to the project root directory
    current_dir = pathlib.Path(__file__).parent.resolve()  # pdf_workflow directory
    project_root = current_dir.parent.parent  # Root of the project
    
    # Set explicit absolute path for Google credentials
    creds_file = os.path.join(project_root, "google_application_credentials.json")
    if os.path.exists(creds_file):
        logging.info(f"Setting credentials path to: {creds_file}")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_file
    else:
        logging.warning(f"Credentials file not found at {creds_file}")
    
    # Construct the absolute path to the PDF document
    document_path = os.path.join(project_root, "data", "input", "10k_2022.pdf")
    
    # Verify the document exists
    if os.path.exists(document_path):
        logging.info(f"Document found: {document_path}")
    else:
        logging.warning(f"Document not found at {document_path}")
        # Try to look for it in different locations
        potential_paths: List[str] = [
            os.path.join(current_dir, "data", "input", "10k_2022.pdf"),
            os.path.join(project_root, "10k_2022.pdf"),
            os.path.join(current_dir, "10k_2022.pdf")
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                document_path = path
                logging.info(f"Found document at alternative location: {document_path}")
                break
            
    # Initialize the state for the content generation process
    state = GraphState(
        document_path=document_path,
        document_mime_type="application/pdf",
        prompt=prompt,
        response_mime_type="application/json",
        response_schema=vertex_schema,
        export_file_name=SecFiling.__name__.lower(),
        export_dir=pathlib.Path(os.path.join(project_root, "data"))  # Use absolute path to project root data directory
    )

    # Run the Gemini graph synchronously with the initialized state
    try:
        logging.info("Starting Gemini workflow execution")
        result = gemini_graph.run_sync(
            ConfigureAPI(), state=state  # Pass the state object with necessary parameters
        )
        
        gemini_graph.mermaid_save("gemini_graph_mermaid.png")
        logging.info("Workflow execution completed successfully")
        
    except Exception as e:
        logging.error(f"Error running Gemini graph: {e}")
        traceback.print_exc()

