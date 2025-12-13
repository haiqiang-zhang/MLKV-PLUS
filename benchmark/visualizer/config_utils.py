import yaml
import os

CONFIG_PATH = "visualizer_config.yaml"

def load_config():
    path = os.path.join(os.path.dirname(__file__), CONFIG_PATH)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_config(config):
    # Load existing config
    existing_config = load_config()
    # Update only the provided fields
    existing_config.update(config)
    # Save the merged config
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(existing_config, f, allow_unicode=True, default_flow_style=False)
