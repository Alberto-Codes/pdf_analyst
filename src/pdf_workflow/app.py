from config.configure_api import ConfigureAPI
from config.state import GraphState
from graph.gemini_graph import gemini_graph
from pydantic import BaseModel, Field


class EmployeeInfo(BaseModel):
    total_employees: int = Field(description="Total number of employees")
    year: int = Field(description="Year of the employee count")
    source_page: int = Field(description="Page number where information was found")
    source_text: str = Field(description="Exact text snippet from document")
    confidence: float = Field(description="Confidence score", ge=0.0, le=1.0)


def get_response_schema_from_model(model_class: type[BaseModel]) -> dict:
    """Convert a Pydantic model to Gemini API response schema format.

    This function converts a Pydantic model class into a format that can
    be used for the response schema in Gemini API. It extracts the model's
    JSON schema and constructs the required schema for the API, including
    the properties and the required fields.

    Args:
        model_class (type[BaseModel]): The Pydantic model class to extract
            the schema from.

    Returns:
        dict: The Gemini API-compatible schema, including properties
            and required fields.
    """
    schema = model_class.model_json_schema()

    return {
        "type": "object",
        "properties": schema["properties"],
        "required": schema.get("required", []),
    }


if __name__ == "__main__":
    """Run the Gemini graph synchronously with state.

    This block of code runs the `gemini_graph` synchronously, starting
    with the `ConfigureAPI` node. The node configures the Gemini API with
    a prompt and additional parameters such as temperature and top_p. It
    also initializes the state of the content generation process using
    a `GraphState` object, which contains information such as the prompt,
    response MIME type, and response schema. The execution of the graph
    results in content generation, and both the result and execution history
    are returned.

    Attributes:
        result (str): The generated content response from the Gemini API.
        history (list): The execution history of the graph, which includes
            all nodes that were executed in sequence, allowing tracking
            of the workflow.
    """
    # Initialize the state for the content generation process
    state = GraphState(
        document_url="https://www.wellsfargo.com/assets/pdf/about/investor-relations/sec-filings/2023/10k.pdf",
        document_mime_type="application/pdf",
        prompt="You extract data from the attached pdf. How many employees?",
        response_mime_type="application/json",
        response_schema=get_response_schema_from_model(EmployeeInfo),
    )

    # Run the Gemini graph synchronously with the initialized state
    result, history = gemini_graph.run_sync(
        ConfigureAPI(), state=state  # Pass the state object with necessary parameters
    )
