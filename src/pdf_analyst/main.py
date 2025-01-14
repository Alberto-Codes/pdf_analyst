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
from phi.knowledge.pdf import PDFKnowledgeBase, PDFUrlKnowledgeBase
from phi.model.ollama import Ollama
# from phi.vectordb.lancedb import LanceDb, SearchType
from phi.vectordb.chroma import ChromaDb
from rich.prompt import Prompt

embedder = OllamaEmbedder(model="nomic-embed-text", dimensions=2048)

# vector_db = LanceDb(
#     table_name="recipes",
#     uri="./tmp/lancedb",
#     search_type=SearchType.vector,
#     embedder=embedder,  # Explicitly pass the OllamaEmbedder
# )

vector_db=ChromaDb(collection="books", embedder=embedder, path="./tmp/chromadb")


# Create a knowledge base from a PDF
# knowledge_base = PDFUrlKnowledgeBase(
#     urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
#     vector_db=vector_db,
# )

knowledge_base = PDFKnowledgeBase(
    path="C:\\Users\\alber\\source\\repos\\pdf_analyst\\data\\books\\alice30.pdf",
    vector_db=vector_db, num_documents=5
)

# Load the knowledge base; comment out after the first run
# knowledge_base.load(recreate=True)



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

    # reasoning_agent = Agent(model=Ollama(id="llama3.2"), reasoning=True, markdown=True, structured_outputs=True)

    agent = Agent(
        model=Ollama(id="llama3.2"),
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
        description="You are a senior NYT researcher answering questions about a book",
        task="answer the user's question using the knowledge base",
        guidelines=["answer the user's question using the knowledge base"],
        instructions=["search the knowledge base for relevant information", "answer the user's question", "provide text from the knowledge base to support for your answer", "format your output in markdown"],
        reasoning=True,
        reasoning_min_steps=2,
        reasoning_max_steps=6,
        # reasoning_agent = Agent(model=Ollama(id="llama3.2"), markdown=True, structured_outputs=True, knowledge_base=knowledge_base),
        reasoning_model=Ollama(id="llama3.2", response_format=str, markdown=True, structured_outputs=True, debug_mode=True, stream=True),
        # verbose=True,

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
