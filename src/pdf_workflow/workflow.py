from pydantic_ai import Agent
from pydantic_ai.models.vertexai import VertexAIModel


def main():
    """Creates an AI agent using VertexAIModel and executes a query.

    This function initializes a `VertexAIModel` instance with the
    "gemini-2.0-flash-001" model, sets up an `Agent` with a system
    prompt enforcing concise responses, and queries the agent
    synchronously.

    The result is printed to the console.
    """
    model = VertexAIModel("gemini-2.0-flash-001")

    agent = Agent(
        model,
        system_prompt="Be concise, reply with one sentence.",
    )

    result = agent.run_sync('Where does "hello world" come from?')
    print(result.data)


if __name__ == "__main__":
    main()
