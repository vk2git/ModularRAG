import yaml
import os

def load_config(config_path="config/settings.yaml"):
    """
    Loads the YAML configuration file
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    full_path = os.path.join(base_dir, config_path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Config file not found at: {full_path}")
    
    with open(full_path, "r") as file:
        try:
            config = yaml.safe_load(file)
            validate_config(config)
            return config
        except yaml.YAMLError as exc:
            raise ValueError(f"Error parsing YAML file: {exc}")

def validate_config(config):
    """
    Validates the configuration structure.
    """
    if not config:
        raise ValueError("Config file is empty")
        
    required_sections = ["llm", "embedding", "vector_db"]
    for section in required_sections:
        if section not in config:
            print(f"⚠️  Warning: Missing '{section}' section in config. Tool may not function correctly.")
            
    if "llm" in config:
        if "mode" not in config["llm"]:
             print("⚠️  Warning: Missing 'llm.mode' in config. Defaulting to 'local'.")
             
    if "vector_db" in config:
        if "provider" not in config["vector_db"]:
            print("⚠️  Warning: Missing 'vector_db.provider' in config. Defaulting to 'chroma'.")