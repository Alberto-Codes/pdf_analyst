import json
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


# Initialize DuckDuckGo search
tool = DuckDuckGoSearchRun(
    max_results=2,
)
tools = [tool]
# Fetch authentication token for secure API access
token = fetch_gcp_id_token()

# Set up the language model
llm = ChatOllama(
    model="llama3.2_32k",
    # model="deepseek-r1_32k",
    base_url=os.getenv("OLLAMA_HOST"),
    client_kwargs={"headers": {"X-Serverless-Authorization": f"Bearer {token}"}},
)

llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    """Processes chatbot responses using the LLM.

    Args:
        state (State): The conversation state containing previous messages.

    Returns:
        State: Updated conversation state with the assistant's response.
    """
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
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


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
