from config.configure_api import ConfigureAPI
from config.state import GraphState
from graph.gemini_graph import gemini_graph

if __name__ == "__main__":
    """Run the Gemini graph synchronously with state.

    This block of code runs the `gemini_graph` synchronously using the
    `ConfigureAPI` node to configure the API with a prompt and additional
    parameters such as temperature and top_p. It also initializes the state
    of the content generation process using a `GraphState` object. The result
    of the graph execution and the execution history are returned.

    Attributes:
        result (str): The result of the content generation from the API.
        history (list): The execution history of the graph, including
        all nodes executed in sequence.
    """
    # Initialize state
    state = GraphState(
        prompt="What is the meaning of life?",
        response_text="",
        client=None,
        config=None,
    )

    # Run graph with state
    result, history = gemini_graph.run_sync(
        ConfigureAPI(), state=state  # Pass the state object
    )
