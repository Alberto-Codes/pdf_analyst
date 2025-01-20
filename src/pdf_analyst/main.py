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
from rich.prompt import Prompt

from .services.knowledge_base import KnowledgeBaseService
from .agents.pdf_agent import PDFAgent

# Initialize and load the knowledge base
knowledge_base_service = KnowledgeBaseService(
    is_url=False,
    pdf_source="data/pdfs/CUAD_v1/full_contract_pdf/Part_I/Affiliate_Agreements/CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement.pdf",
)
knowledge_base_service.load_knowledge_base(recreate=True)

from sqlalchemy.orm import Session
from .db.session import get_db
from .db.models import Category  # Assuming you have a Category model


def fetch_categories(session: Session) -> list:
    return session.query(Category).all()


def pdf_agent(user: str = "user") -> None:
    agent = PDFAgent(knowledge_base=knowledge_base_service.knowledge_base)
    run_id = getattr(agent, "run_id", None)  # Initialize run_id

    if run_id is None:
        run_id = agent.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")

    with get_db() as session:
        categories = fetch_categories(session)
        for category in categories:
            question = category.description
            print(f"Question: {question}")
            agent.print_response(question)


if __name__ == "__main__":
    typer.run(pdf_agent)
