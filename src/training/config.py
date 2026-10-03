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


def resolve_device(requested: str = "auto") -> str:
    """Pick the best available device.

    Priority: requested (if not "auto") > MPS (Apple Silicon) > CUDA > CPU.
    This lets the same config run on any machine without changes.
    """
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def seed_everything(seed: int) -> None:
    """Seed torch, numpy and the stdlib RNG so runs are reproducible."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_commit_hash():
    """Current git commit, or None if unavailable (not a repo / git missing)."""
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
