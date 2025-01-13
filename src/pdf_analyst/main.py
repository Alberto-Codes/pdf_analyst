from phi.agent import Agent
from phi.embedder.ollama import OllamaEmbedder
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.model.ollama import Ollama
from phi.vectordb.chroma import ChromaDb
from phi.embedder.ollama import OllamaEmbedder

# Create embedder instance first
embedder = OllamaEmbedder()

# Create ChromaDb instance with explicit embedder
vector_db = ChromaDb(
    collection="recipes",
    embedder=embedder,  # Explicitly pass the OllamaEmbedder
    path="data/chromadb",
    persistent_client=True  # Add this to ensure data persistence
)
# Create a knowledge base from a PDF
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=vector_db
)
# Comment out after first run as the knowledge base is loaded
knowledge_base.load()

agent = Agent(
    model=Ollama(id="llama3.2"),
    # Add the knowledge base to the agent
    knowledge=knowledge_base,
    show_tool_calls=True,
    markdown=True,
    task="Use knowledge_base to answer questions",
    guidelines=["Only use knowledge_base to answer questions"],
    instructions=["Only use knowledge_base to answer questions"],
    show_summary=True,
)
agent.print_response(
    "Can you give me a papaya recipe?", stream=True
)
