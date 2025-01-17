import os

from phi.agent import Agent
from phi.model.ollama import Ollama
from phi.model.vertexai import Gemini

from pdf_analyst.utilities.config_loader import \
    load_config  # Add config loader import
from pdf_analyst.utilities.gcp_token import \
    fetch_gcp_id_token  # Add GCP token import


class BaseAgent(Agent):
    def __init__(self):
        super().__init__()
        config = load_config()  # Use the shared config loader
        self.set_model(config)

    def set_model(self, config):
        model_type = config["model"]  # Direct access since we know it exists

        if model_type == "ollama":
            id_token = fetch_gcp_id_token()
            extra_headers = {"X-Serverless-Authorization": f"Bearer {id_token}"}
            host = os.getenv("OLLAMA_HOST", "localhost")
            self.model = Ollama(
                id=config["model_id"],
                host=host,
                port=8080,
                client_params={"headers": extra_headers},
            )
        elif model_type == "gemini":
            self.model = Gemini()
        else:
            raise ValueError(f"Unknown model: {model_type}")
