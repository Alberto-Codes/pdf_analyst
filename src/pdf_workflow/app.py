from config.configure_api import ConfigureAPI
from config.state import GraphState
from graph.gemini_graph import gemini_graph
from models.employee_info import EmployeeInfo
from utils.schema_utils import get_response_schema_from_model

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
    # Initialize the state for the content generation process
    state = GraphState(
        document_url="https://www.wellsfargo.com/assets/pdf/about/investor-relations/sec-filings/2023/10k.pdf",
        document_mime_type="application/pdf",
        prompt="You extract data from the attached pdf. How many employees?",
        response_mime_type="application/json",
        response_schema=get_response_schema_from_model(EmployeeInfo),
        export_file_name=EmployeeInfo.__name__.lower(),
    )

    # Run the Gemini graph synchronously with the initialized state
    result, history = gemini_graph.run_sync(
        ConfigureAPI(), state=state  # Pass the state object with necessary parameters
    )
