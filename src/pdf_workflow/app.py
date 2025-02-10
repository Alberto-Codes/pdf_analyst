from config.configure_api import ConfigureAPI
from graph.gemini_graph import gemini_graph

if __name__ == "__main__":
    """Run the Gemini graph synchronously.

    This block of code runs the `gemini_graph` synchronously using the
    `ConfigureAPI` node to configure the API with a prompt and additional
    parameters such as temperature and top_p. The result of the graph execution
    and the execution history are returned.

    Attributes:
        result (str): The result of the content generation from the API.
        history (list): The execution history of the graph, including
        all nodes executed in sequence.
    """
    result, history = gemini_graph.run_sync(
        ConfigureAPI(
            prompt="What is the meaning of life?",
            temperature=0.7,
            top_p=0.95,
        )
    )
