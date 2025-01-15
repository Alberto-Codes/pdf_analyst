import os
import yaml

def load_config():
    environment = os.getenv("ENVIRONMENT", "dev")
    config_file = f"src/pdf_analyst/config/{environment}_config.yml"
    
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    
    return config