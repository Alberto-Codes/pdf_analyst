"""PDF Knowledge Base Chatbot with Ollama and ChromaDb.

This script sets up a chatbot that utilizes an Ollama-based language model
and a vector database (ChromaDb) to interact with users based on a PDF
knowledge base. It enables users to query recipes from a provided PDF file.

Modules:
    typer: CLI utility for running the chatbot.
    phi.agent: Provides the chatbot's agent functionality.
    phi.embedder.ollama: Embeds text using the Ollama model.
    phi.knowledge.pdf: Handles knowledge base loading from PDFs.
    phi.model.ollama: Represents the Ollama-based language model.
    phi.vectordb.chroma: Manages vector database storage with ChromaDb.
    rich.prompt: Enables rich-text user input prompts.

Attributes:
    embedder (OllamaEmbedder): An instance for text embedding.
    vector_db (ChromaDb): A vector database instance for storing embeddings.
    knowledge_base (PDFUrlKnowledgeBase): A knowledge base built from a PDF.

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

from typing import Optional

import typer
from phi.agent import Agent
from phi.embedder.ollama import OllamaEmbedder
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.model.ollama import Ollama
from phi.vectordb.chroma import ChromaDb
from rich.prompt import Prompt

# Create embedder instance first
embedder = OllamaEmbedder(model="nomic-embed-text")

# Create ChromaDb instance with explicit embedder
vector_db = ChromaDb(
    collection="recipes",
    embedder=embedder,  # Explicitly pass the OllamaEmbedder
    persistent_client=True,  # Ensure data persistence
)

# Create a knowledge base from a PDF
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=vector_db,
)

# Load the knowledge base; comment out after the first run
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
        model=Ollama(id="llama3.2"),
        run_id=run_id,
        user_id=user,
        knowledge_base=knowledge_base,
        use_tools=True,
        show_tool_calls=True,
        debug_mode=True,
    )

    if run_id is None:
        run_id = agent.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")

    while True:
        message = Prompt.ask(f"[bold] :sunglasses: {user} [/bold]")
        if message in ("exit", "bye"):
            break
        agent.print_response(message)


if __name__ == "__main__":
    typer.run(pdf_agent)
