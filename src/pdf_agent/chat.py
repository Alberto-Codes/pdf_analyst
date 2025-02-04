"""
Chatbot with Search Integration using LangGraph and ChatOllama.

This script implements a chatbot using the `langgraph` library with a
stateful conversation model. It integrates DuckDuckGo search as an
external tool and utilizes the `ChatOllama` language model. The chatbot
processes user queries, searches for relevant information when needed,
and generates responses in real-time.

Modules:
    os: Accesses environment variables.
    typing: Supports type hints, including `Annotated`.
    langchain_community.tools: Includes the `DuckDuckGoSearchRun` tool.
    langchain_core.tools: Provides general tooling support.
    langchain_ollama: Integrates the `ChatOllama` LLM.
    langgraph.graph: Manages the state graph.
    langgraph.graph.message: Handles message processing.
    langgraph.prebuilt: Includes prebuilt graph components.
    langgraph.checkpoint.memory: Provides in-memory state saving.
    langgraph.types: Includes command handling and interruptions.
    typing_extensions: Provides additional typing utilities.
    utils.gcp_token: Fetches a secure GCP authentication token.

Classes:
    State:
        Represents the chatbot state, storing conversation messages.

Functions:
    human_assistance(query: str) -> str:
        Requests human intervention when needed.

    chatbot(state: State) -> State:
        Processes chatbot responses using the LLM.

    stream_graph_updates(user_input: str) -> None:
        Streams chatbot responses in real-time.

Usage:
    Run the script and enter queries interactively.
    Type "quit", "exit", or "q" to end the session.

Example:
    ```
    User: What is LangGraph?
    Assistant: LangGraph is a library for managing conversation state...
    ```
"""

import os
from typing import Annotated

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict
from utils.gcp_token import fetch_gcp_id_token


class State(TypedDict):
    """Represents the chatbot's conversation state.

    Attributes:
        messages (list): A list of messages exchanged in the conversation.
    """

    messages: Annotated[list, add_messages]


# Initialize the state graph
graph_builder = StateGraph(State)


@tool
def human_assistance(query: str) -> str:
    """Requests assistance from a human operator.

    This function is triggered when the chatbot determines that
    human intervention is required. It interrupts the chatbot flow
    and allows a human response.

    Args:
        query (str): The query requiring human assistance.

    Returns:
        str: The response provided by a human.
    """
    human_response = interrupt({"query": query})
    return human_response["data"]


# Initialize DuckDuckGo search tool
web_search = DuckDuckGoSearchRun(max_results=2)
tools = [web_search, human_assistance]

# Initialize memory saver for state persistence
memory = MemorySaver()

# Fetch authentication token for secure API access
token = fetch_gcp_id_token()

# Set up the language model with authentication
llm = ChatOllama(
    model="llama3.2_32k",
    base_url=os.getenv("OLLAMA_HOST"),
    client_kwargs={"headers": {"X-Serverless-Authorization": f"Bearer {token}"}},
)

# Bind tools to the language model
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State) -> State:
    """Processes chatbot responses using the LLM.

    This function takes the current conversation state, forwards
    it to the language model, and returns an updated state with
    the assistant's response.

    Args:
        state (State): The conversation state containing previous messages.

    Returns:
        State: Updated conversation state with the assistant's response.
    """
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# Add nodes to the state graph
graph_builder.add_node("chatbot", chatbot)

# Initialize tool node and add to the graph
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# Define conditional and sequential transitions
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")

# Set chatbot as the entry point
graph_builder.set_entry_point("chatbot")

# Compile the state graph with checkpointing
graph = graph_builder.compile(checkpointer=memory)

# Configuration settings for the chatbot session
config = {"configurable": {"thread_id": "1"}}


def stream_graph_updates(user_input: str) -> None:
    """Streams responses from the chatbot in real-time.

    This function takes user input, processes it through the chatbot
    pipeline, and prints the assistant's response.

    Args:
        user_input (str): The user's input message.

    Prints:
        The assistant's response to the terminal.
    """
    for event in graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
        stream_mode="values",
    ):
        event["messages"][-1].pretty_print()


# Visualize the conversation flow graph
print(graph.get_graph().draw_mermaid())

# Start an interactive chatbot session
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except Exception:
        # Fallback scenario if input() is unavailable
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
