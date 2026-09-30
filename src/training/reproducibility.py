import random
import subprocess
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed torch, numpy and the stdlib RNG so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_commit_hash() -> Optional[str]:
    """Current git commit, or None if unavailable (not a repo / git missing)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
