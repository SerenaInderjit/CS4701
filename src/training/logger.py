import json
import logging
import time
from typing import Any, Dict, List, Optional


def _json_default(value: Any) -> Any:
    # numpy scalars and similar
    if hasattr(value, "item"):
        return value.item()
    return str(value)


class Logger:
    """Records per-step data and per-episode statistics.

    Call `log_step` after every environment step and `end_episode` when an
    episode finishes (Runner does both). Per-episode stats are kept in
    `self.episodes`. Set keep_step_records=False for long training runs so
    per-step records don't accumulate in memory.
    """

    def __init__(self, keep_step_records: bool = True):
        self.records: List[Dict[str, Any]] = []
        self.episodes: List[Dict[str, Any]] = []
        self.keep_step_records = keep_step_records
        self.logger = logging.getLogger(__name__)
        self._episode_start = 0  # index into self.records where the current episode began
        self._episode_start_time = time.time()

    def log_step(self, action, reward, info):
        self.records.append({
            "action": action,
            "reward": reward,
            "info": info,
        })

    def end_episode(self) -> Optional[Dict[str, Any]]:
        """Summarise the steps since the last call and store them as one episode."""
        steps = self.records[self._episode_start:]
        now = time.time()
        if not steps:
            self._episode_start_time = now
            return None

        x_positions = [s["info"].get("x_pos", 0) for s in steps]
        stats = {
            "episode": len(self.episodes),
            "length": len(steps),
            "total_reward": float(sum(s["reward"] for s in steps)),
            "max_x_pos": max(x_positions),
            "final_x_pos": x_positions[-1],
            "flag_get": any(bool(s["info"].get("flag_get", False)) for s in steps),
            "duration_seconds": now - self._episode_start_time,
        }
        self.episodes.append(stats)

        if self.keep_step_records:
            self._episode_start = len(self.records)
        else:
            self.records.clear()
            self._episode_start = 0
        self._episode_start_time = now
        return stats

    def summary(self, last_n: Optional[int] = None) -> Dict[str, float]:
        """Mean statistics over all (or the last `last_n`) finished episodes.

        Values are coerced to native Python types so the result is JSON-serializable
        (episode stats can contain numpy scalars from the emulator info dict).
        """
        episodes = self.episodes[-last_n:] if last_n else self.episodes
        n = len(episodes)
        if n == 0:
            return {"num_episodes": 0}
        return {
            "num_episodes": n,
            "mean_reward": float(sum(e["total_reward"] for e in episodes) / n),
            "mean_distance": float(sum(e["max_x_pos"] for e in episodes) / n),
            "best_distance": int(max(e["max_x_pos"] for e in episodes)),
            "completion_rate": float(sum(e["flag_get"] for e in episodes) / n),
            "mean_episode_length": float(sum(e["length"] for e in episodes) / n),
        }

    def save_episodes(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.episodes, f, indent=2, default=_json_default)

    def info(self, message):
        self.logger.info(message)

    def error(self, message, exc_info=False):
        self.logger.error(message, exc_info=exc_info)
