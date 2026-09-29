import os
import re
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: int = 0,
    metrics: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a checkpoint atomically (temp file + rename) so a crash can't corrupt it."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "metrics": metrics or {},
        "extra": extra or {},
    }
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location="cpu",
) -> Dict[str, Any]:
    """Restore model (and optimizer, if given) in place. Returns the full payload."""
    # weights_only=False: these are our own files and contain plain dicts.
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


class CheckpointManager:
    """Keeps the last `keep_last` periodic checkpoints plus a `best.pt`.

    Periodic files are named checkpoint_<step>.pt. If `score` is passed to
    `save` (e.g. mean distance from evaluation) and it beats the best so far,
    the checkpoint is also written to best.pt.
    """

    _PATTERN = re.compile(r"^checkpoint_(\d+)\.pt$")

    def __init__(self, directory: str = "checkpoints", keep_last: int = 3):
        self.directory = directory
        self.keep_last = keep_last
        self.best_score: Optional[float] = None
        os.makedirs(directory, exist_ok=True)

        # Resume best-score tracking if a previous run left a best.pt behind.
        if os.path.exists(self.best_path):
            payload = torch.load(self.best_path, map_location="cpu", weights_only=False)
            self.best_score = payload.get("extra", {}).get("score")

    @property
    def best_path(self) -> str:
        return os.path.join(self.directory, "best.pt")

    def _periodic_paths(self):
        found = []
        for name in os.listdir(self.directory):
            match = self._PATTERN.match(name)
            if match:
                found.append((int(match.group(1)), os.path.join(self.directory, name)))
        return [path for _, path in sorted(found)]

    def latest_path(self) -> Optional[str]:
        paths = self._periodic_paths()
        return paths[-1] if paths else None

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        step: int,
        metrics: Optional[Dict[str, Any]] = None,
        score: Optional[float] = None,
    ) -> str:
        path = os.path.join(self.directory, f"checkpoint_{step:09d}.pt")
        extra = {"score": score} if score is not None else {}
        save_checkpoint(path, model, optimizer, step=step, metrics=metrics, extra=extra)

        if score is not None and (self.best_score is None or score > self.best_score):
            self.best_score = score
            save_checkpoint(self.best_path, model, optimizer, step=step, metrics=metrics, extra=extra)

        for old_path in self._periodic_paths()[:-self.keep_last]:
            os.remove(old_path)
        return path

    def load_latest(self, model, optimizer=None, map_location="cpu") -> Optional[Dict[str, Any]]:
        path = self.latest_path()
        if path is None:
            return None
        return load_checkpoint(path, model, optimizer, map_location)

    def load_best(self, model, optimizer=None, map_location="cpu") -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.best_path):
            return None
        return load_checkpoint(self.best_path, model, optimizer, map_location)
