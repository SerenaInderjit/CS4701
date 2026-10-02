import os
import re
from typing import Any, Dict, Optional

import torch

from src.paths import CHECKPOINTS_DIR


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: int = 0,
    metrics: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
    additional_state: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a checkpoint atomically (temp file + rename) so a crash can't corrupt it.

    `additional_state` carries extra state dicts (e.g. a PPO value model and its
    optimizer) that don't fit the single-model signature above.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "metrics": metrics or {},
        "extra": extra or {},
        "additional_state": additional_state or {},
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
    """Per-run checkpoint storage.

    During training two live artifacts plus a bounded top-k candidate set:
      - latest.pt           : most recent checkpoint (for --resume)
      - best.pt             : highest-scoring checkpoint so far
      - checkpoint_<step>.pt: at most `keep_top` top-scoring periodic saves

    `finalize()` (called when training finishes) deletes latest.pt and
    prunes the periodic set to the top `keep_top` by score, leaving only
    the best model(s) on disk.
    """

    _PATTERN = re.compile(r"^checkpoint_(\d+)\.pt$")

    def __init__(self, directory: str = CHECKPOINTS_DIR, keep_top: int = 3):
        self.directory = directory
        self.keep_top = keep_top
        self.best_score: Optional[float] = None
        os.makedirs(directory, exist_ok=True)

        # Resume best-score tracking if a previous run left a best.pt behind.
        if os.path.exists(self.best_path):
            payload = torch.load(self.best_path, map_location="cpu", weights_only=False)
            self.best_score = payload.get("extra", {}).get("score")

    @property
    def best_path(self) -> str:
        return os.path.join(self.directory, "best.pt")

    @property
    def latest_checkpoint_path(self) -> str:
        return os.path.join(self.directory, "latest.pt")

    def _periodic_paths(self):
        found = []
        for name in os.listdir(self.directory):
            match = self._PATTERN.match(name)
            if match:
                found.append((int(match.group(1)), os.path.join(self.directory, name)))
        return [path for _, path in sorted(found)]

    def _score_of(self, path: str) -> Optional[float]:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            return payload.get("extra", {}).get("score")
        except Exception:
            return None

    def latest_path(self) -> Optional[str]:
        if os.path.exists(self.latest_checkpoint_path):
            return self.latest_checkpoint_path
        paths = self._periodic_paths()
        return paths[-1] if paths else None

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        step: int,
        metrics: Optional[Dict[str, Any]] = None,
        score: Optional[float] = None,
        additional_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        path = os.path.join(self.directory, f"checkpoint_{step:09d}.pt")
        extra = {"score": score} if score is not None else {}
        save_checkpoint(
            path, model, optimizer, step=step, metrics=metrics, extra=extra,
            additional_state=additional_state,
        )
        # Keep the resume target valid: latest.pt mirrors this save.
        save_checkpoint(
            self.latest_checkpoint_path, model, optimizer, step=step,
            metrics=metrics, extra=extra, additional_state=additional_state,
        )

        if score is not None and (self.best_score is None or score > self.best_score):
            self.best_score = score
            save_checkpoint(self.best_path, model, optimizer, step=step, metrics=metrics, extra=extra)

        # Bound periodic saves to the top `keep_top` by score (recency wins ties).
        scored = [
            (self._score_of(p) if self._score_of(p) is not None else float("-inf"), self._step_of(p), p)
            for p in self._periodic_paths()
        ]
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for _, _, old_path in scored[self.keep_top:]:
            os.remove(old_path)
        return path

    def _step_of(self, path: str) -> int:
        match = self._PATTERN.match(os.path.basename(path))
        return int(match.group(1)) if match else 0

    def finalize(self, keep_top: Optional[int] = None) -> None:
        """Training finished: drop the incremental latest.pt and keep only
        the top `keep_top` scored periodic checkpoints (default: keep_top)."""
        if os.path.exists(self.latest_checkpoint_path):
            os.remove(self.latest_checkpoint_path)
        keep = self.keep_top if keep_top is None else keep_top
        scored = [(self._score_of(p) if self._score_of(p) is not None else float("-inf"), self._step_of(p), p)
                  for p in self._periodic_paths()]
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for _, _, old_path in scored[keep:]:
            os.remove(old_path)

    def load_latest(self, model, optimizer=None, map_location="cpu") -> Optional[Dict[str, Any]]:
        path = self.latest_path()
        if path is None:
            return None
        return load_checkpoint(path, model, optimizer, map_location)

    def load_best(self, model, optimizer=None, map_location="cpu") -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.best_path):
            return None
        return load_checkpoint(self.best_path, model, optimizer, map_location)
