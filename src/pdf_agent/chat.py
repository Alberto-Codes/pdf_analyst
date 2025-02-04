"""
Chatbot with Search Integration using LangGraph and ChatOllama.

This script implements a chatbot using the `langgraph` library with a
stateful conversation model. It integrates DuckDuckGo search as an
external tool and utilizes the `ChatOllama` language model. The chatbot
processes user queries, searches for relevant information when needed,
and generates responses in real-time.

Modules:
    json: Provides JSON serialization and deserialization.
    os: Accesses environment variables.
    typing: Supports type hints, including `Annotated`.
    langchain_community.tools: Includes the `DuckDuckGoSearchRun` tool.
    langchain_core.messages: Defines message structures.
    langchain_ollama: Integrates the `ChatOllama` LLM.
    langgraph.graph: Manages the state graph.
    langgraph.graph.message: Handles message processing.
    langgraph.prebuilt: Includes prebuilt graph components.
    typing_extensions: Provides additional typing utilities.
    utils.gcp_token: Fetches a secure GCP authentication token.

Classes:
    State:
        Defines the chatbot state, storing conversation messages.

Functions:
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
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
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

# Initialize DuckDuckGo search tool
tool = DuckDuckGoSearchRun(max_results=2)
tools = [tool]

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

    Args:
        state (State): The conversation state containing previous messages.

    Returns:
        State: Updated conversation state with the assistant's response.
    """
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# Add chatbot node to the graph
graph_builder.add_node("chatbot", chatbot)

# Add tool node for handling external search tools
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

# Define conditional edges based on tool invocation
graph_builder.add_conditional_edges("chatbot", tools_condition)

# Ensure the chatbot node is revisited after tool execution
graph_builder.add_edge("tools", "chatbot")

# Set the chatbot as the entry point of the conversation
graph_builder.set_entry_point("chatbot")

# Compile the graph for execution
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


# Start an interactive chatbot session
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # Fallback scenario if input() is unavailable
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
