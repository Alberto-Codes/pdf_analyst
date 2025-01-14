"""PDF Knowledge Base Chatbot with Gemini and ChromaDb.

This script sets up a chatbot that utilizes a Gemini-based language model
and a vector database (ChromaDb) to interact with users based on a PDF
knowledge base. Users can query recipes or other information from a PDF.

Modules:
    typer: CLI utility for running the chatbot.
    phi.agent: Provides the chatbot's agent functionality.
    phi.embedder.google: Embeds text using the Gemini model.
    phi.knowledge.pdf: Handles knowledge base loading from PDFs.
    phi.model.vertexai: Represents the Gemini-based language model.
    phi.vectordb.chroma: Manages vector database storage with ChromaDb.
    rich.prompt: Enables rich-text user input prompts.

Attributes:
    embedder (GeminiEmbedder): Embeds text using Google's Gemini model.
    vector_db (ChromaDb): Stores and retrieves vector embeddings.
    knowledge_base (PDFUrlKnowledgeBase): Stores and retrieves PDF content.

Functions:
    pdf_agent(user: str = "user") -> None:
        Runs an interactive chatbot session with the PDF knowledge base.

Usage:
    Run this script to start an interactive chatbot session. Type queries
    based on the knowledge base (e.g., recipe-related questions). To exit,
    type 'exit' or 'bye'.

Example:
    ```
    python script.py
    ```
"""

__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import sqlite3
from typing import Optional

import typer
from phi.agent import Agent
from phi.embedder.google import GeminiEmbedder
from phi.knowledge.pdf import PDFKnowledgeBase, PDFUrlKnowledgeBase
from phi.model.vertexai import Gemini
from phi.vectordb.chroma import ChromaDb
from rich.prompt import Prompt

embedder = GeminiEmbedder()

vector_db = ChromaDb(collection="books", embedder=embedder, path="./tmp/chromadb")

# Create a knowledge base from a PDF URL
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=vector_db,
)

# Load the knowledge base (comment out after the first run)
knowledge_base.load(recreate=True)


def pdf_agent(user: str = "user") -> None:
    """Starts an interactive chatbot session with a PDF-based knowledge base.

    This function initializes an AI agent that interacts with the user using
    a pre-loaded PDF knowledge base. The chatbot session continues until
    the user exits.

    Args:
        user (str, optional): The user identifier. Defaults to "user".

    Example:
        ```
        python script.py
        ```
        The user can interact with the chatbot in a terminal environment.

    Commands:
        - Type queries to ask about the content of the PDF.
        - Type 'exit' or 'bye' to terminate the session.
    """
    run_id: Optional[str] = None

    agent = Agent(
        model=Gemini(),
        run_id=run_id,
        user_id=user,
        knowledge_base=knowledge_base,
        read_chat_history=True,
        add_chat_history_to_messages=True,
        num_history_responses=3,
        use_tools=True,
        show_tool_calls=True,
        markdown=True,
        debug_mode=True,
        stream=True,
        description="You are a senior NYT researcher answering questions about a book.",
        task="Answer the user's question using the knowledge base.",
        guidelines=["Answer the user's question using the knowledge base."],
        instructions=[
            "Search the knowledge base for relevant information.",
            "Answer the user's question concisely.",
            "Provide text from the knowledge base to support the answer.",
            "Format the response in Markdown.",
        ],
        reasoning=True,
        reasoning_min_steps=2,
        reasoning_max_steps=6,
        reasoning_model=Gemini(
            response_format=str,
            markdown=True,
            structured_outputs=True,
            debug_mode=True,
            stream=True,
        ),
        verbose=True,
    )

    if run_id is None:
        run_id = agent.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")

    while True:
        message = Prompt.ask(f"[bold] :sunglasses: {user} [/bold]")
        if message.lower() in ("exit", "bye"):
            break
        agent.print_response(message)


if __name__ == "__main__":
    typer.run(pdf_agent)
