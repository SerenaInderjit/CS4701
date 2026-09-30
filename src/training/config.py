from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file into a dict."""
    with open(path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config file {path} must contain a mapping at the top level")
    return config


def save_config(config: Dict[str, Any], path: str) -> None:
    """Write a config dict to YAML (run directories keep a resolved copy)."""
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
