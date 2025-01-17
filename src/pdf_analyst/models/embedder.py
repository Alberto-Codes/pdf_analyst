import os

from phi.embedder.google import GeminiEmbedder
from phi.embedder.ollama import OllamaEmbedder

from pdf_analyst.utilities.config_loader import \
    load_config  # Use absolute import
from pdf_analyst.utilities.gcp_token import \
    fetch_gcp_id_token  # Add GCP token import


class EmbedderModel:
    def __init__(self):
        config = load_config()
        embedder_type = config["embedder"]  # Direct access since we know it exists

        if embedder_type == "ollama":
            id_token = fetch_gcp_id_token()
            extra_headers = {"X-Serverless-Authorization": f"Bearer {id_token}"}
            host = os.getenv("OLLAMA_HOST", "localhost")
            self.embedder = OllamaEmbedder(
                model="nomic-embed-text",
                host=host,
                client_kwargs={"headers": extra_headers},
            )
        elif embedder_type == "gemini":
            self.embedder = GeminiEmbedder()
        else:
            raise ValueError(f"Unsupported embedder type: {embedder_type}")
