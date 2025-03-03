import os
import pathlib

from pdf_workflow.config.prompt import prompt
from pdf_workflow.config.state import GraphState
from pdf_workflow.graph.gemini_graph import gemini_graph
from pdf_workflow.models.sec_filing import SecFiling
from pdf_workflow.nodes.configure_api import ConfigureAPI
from pdf_workflow.utils.schema_utils import create_vertex_schema

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
    # Get the absolute path to the project root directory
    current_dir = pathlib.Path(__file__).parent.resolve()  # pdf_workflow directory
    project_root = current_dir.parent.parent  # Root of the project
    
    # Set explicit absolute path for Google credentials
    creds_file = os.path.join(project_root, "google_application_credentials.json")
    if os.path.exists(creds_file):
        print(f"Setting credentials path to: {creds_file}")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_file
    else:
        print(f"Warning: Credentials file not found at {creds_file}")
    
    # Construct the absolute path to the PDF document
    document_path = os.path.join(project_root, "data", "input", "10k_2022.pdf")
    
    # Verify the document exists
    if os.path.exists(document_path):
        print(f"Document found: {document_path}")
    else:
        print(f"Warning: Document not found at {document_path}")
        # Try to look for it in different locations
        potential_paths = [
            os.path.join(current_dir, "data", "input", "10k_2022.pdf"),
            os.path.join(project_root, "10k_2022.pdf"),
            os.path.join(current_dir, "10k_2022.pdf")
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                document_path = path
                print(f"Found document at alternative location: {document_path}")
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
        result = gemini_graph.run_sync(
            ConfigureAPI(), state=state  # Pass the state object with necessary parameters
        )
        
        gemini_graph.mermaid_save("gemini_graph_mermaid.png")
        
    except Exception as e:
        print(f"Error running Gemini graph: {e}")
        import traceback
        traceback.print_exc()

