import os
from typing import Annotated

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from utils.gcp_token import fetch_gcp_id_token


class State(TypedDict):
    """Defines the chatbot state.

    Attributes:
        messages (list): A list of messages passed in the conversation.
    """

    messages: Annotated[list, add_messages]


# Initialize the state graph
graph_builder = StateGraph(State)

# Fetch authentication token for secure API access
token = fetch_gcp_id_token()

# Set up the language model
llm = ChatOllama(
    model="llama3.2",
    base_url=os.getenv("OLLAMA_HOST"),
    client_kwargs={"headers": {"X-Serverless-Authorization": f"Bearer {token}"}},
)


def chatbot(state: State) -> State:
    """Processes chatbot responses using the LLM.

    Args:
        state (State): The conversation state containing previous messages.

    Returns:
        State: Updated conversation state with the assistant's response.
    """
    return {"messages": [llm.invoke(state["messages"])]}


# Add nodes and edges to the graph
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile the state graph
graph = graph_builder.compile()


def stream_graph_updates(user_input: str) -> None:
    """Streams responses from the chatbot in real-time.

    Args:
        user_input (str): The user's input message.

    Prints:
        The assistant's response to the terminal.
    """
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


# Interactive chat loop
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)

    except Exception as e:
        # Fallback for input errors (e.g., running in environments without `input()`)
        user_input = "What do you know about LangGraph?"
        print("User:", user_input)
        stream_graph_updates(user_input)
        print(f"Error encountered: {e}")
        break
