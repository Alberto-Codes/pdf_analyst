from config.configure_api import ConfigureAPI
from config.state import GraphState
from graph.gemini_graph import gemini_graph

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
    # Initialize state with prompt, MIME type, and response schema
    state = GraphState(
        prompt="What is the meaning of life?",
        response_mime_type="application/json",
        response_schema={
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "source": {"type": "string"},
            },
        },
    )

    # Run the graph with the initialized state and pass the state object
    result, history = gemini_graph.run_sync(
        ConfigureAPI(), state=state  # Pass the state object for context
    )
