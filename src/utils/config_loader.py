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


def load_architecture_config(architecture_name: str) -> dict:
    """
    Load architecture-specific configuration from config/architectures/<name>.yaml.
    
    Falls back to an empty dict if the file doesn't exist (architecture
    will use defaults from settings.yaml).
    
    Args:
        architecture_name: Name of the architecture (e.g., "naive", "advanced")
    
    Returns:
        Dict of architecture-specific settings
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    arch_path = os.path.join(base_dir, "config", "architectures", f"{architecture_name}.yaml")

    if not os.path.exists(arch_path):
        return {}

    with open(arch_path, "r") as file:
        try:
            config = yaml.safe_load(file) or {}
            return config
        except yaml.YAMLError as exc:
            print(f"⚠️  Error parsing architecture config '{architecture_name}': {exc}")
            return {}


def get_active_architecture(config: dict = None) -> str:
    """
    Get the active architecture name from config.
    
    Args:
        config: Optional pre-loaded config dict. Loads from disk if None.
    
    Returns:
        Architecture name string (e.g., "naive", "advanced")
    """
    if config is None:
        config = load_config()
    return config.get("architecture", {}).get("active", "naive")


def set_active_architecture(architecture_name: str, config_path="config/settings.yaml"):
    """
    Update the active architecture in settings.yaml.
    
    Args:
        architecture_name: Name of the architecture to activate
        config_path: Path to the settings file
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    full_path = os.path.join(base_dir, config_path)

    with open(full_path, "r") as f:
        config = yaml.safe_load(f)

    if "architecture" not in config:
        config["architecture"] = {}
    
    config["architecture"]["active"] = architecture_name

    with open(full_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


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