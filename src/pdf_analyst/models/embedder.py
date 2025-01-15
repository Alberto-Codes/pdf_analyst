from phi.embedder.google import GeminiEmbedder
from phi.embedder.ollama import OllamaEmbedder
from pdf_analyst.utilities.config_loader import load_config  # Use absolute import

class EmbedderModel:
    def __init__(self):
        config = load_config()
        embedder_type = config["embedder"]  # Direct access since we know it exists
        
        if embedder_type == "ollama":
            self.embedder = OllamaEmbedder()
        elif embedder_type == "gemini":
            self.embedder = GeminiEmbedder()
        else:
            raise ValueError(f"Unsupported embedder type: {embedder_type}")