from pathlib import Path
import yaml

def load_config(path: str | Path) -> dict:
    """
    Load a configuration from yaml file
    """
    path = Path(path)
    with path.open("r") as f:
        return yaml.safe_load(f) or {}