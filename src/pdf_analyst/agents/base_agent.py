from phi.agent import Agent
from phi.model.ollama import Ollama
from phi.model.vertexai import Gemini
from pdf_analyst.utilities.config_loader import load_config  # Add config loader import

class BaseAgent(Agent):
    def __init__(self):
        super().__init__()
        config = load_config()  # Use the shared config loader
        self.set_model(config)
    
    def set_model(self, config):
        model_type = config["model"]  # Direct access since we know it exists
        
        if model_type == "ollama":
            self.model = Ollama(id=config["model_id"])
        elif model_type == "gemini":
            self.model = Gemini()
        else:
            raise ValueError(f"Unknown model: {model_type}")