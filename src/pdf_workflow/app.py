from config.prompt import prompt
from config.state import GraphState
from graph.gemini_graph import gemini_graph
from models.sec_filing import SecFiling
from nodes.configure_api import ConfigureAPI
from utils.schema_utils import create_vertex_schema

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
    # Initialize the state for the content generation process
    state = GraphState(
        document_path="data/input/10k_2022.pdf",
        document_mime_type="application/pdf",
        prompt=prompt,
        response_mime_type="application/json",
        response_schema=vertex_schema,
        export_file_name=SecFiling.__name__.lower(),
    )

    # Run the Gemini graph synchronously with the initialized state
    result, history = gemini_graph.run_sync(
        ConfigureAPI(), state=state  # Pass the state object with necessary parameters
    )
    print(result)

    gemini_graph.mermaid_save("gemini_graph_mermaid.png")
