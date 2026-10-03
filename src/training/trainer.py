import json
import os
from datetime import datetime
from typing import Optional

from src.training.checkpoint import CheckpointManager
from src.training.logger import Logger
from src.training.runner import Runner


class Trainer:
    """PPO training loop: collect rollout -> compute GAE -> update -> checkpoint.

    Each update fills the rollout buffer across episode boundaries (the Runner
    carries the in-progress episode over), bootstraps the final value, and runs
    the agent's minibatch updates. The buffer is cleared after every update.

    If `run_dir` is given, the run is self-contained there: one JSON line per
    update in metrics.jsonl and the episode log in episodes.json.
    """

    def __init__(
        self,
        env,
        agent,
        buffer,
        checkpoint_dir: Optional[str] = None,
        logger: Optional[Logger] = None,
        checkpoint_every: int = 10,
        run_dir: Optional[str] = None,
    ):
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.logger = logger or Logger(keep_step_records=False)
        self.runner = Runner(env, agent, self.logger)
        if checkpoint_dir is None:
            checkpoint_dir = (
                os.path.join(run_dir, "checkpoints") if run_dir else "checkpoints"
            )
        self.checkpoints = CheckpointManager(checkpoint_dir)
        self.checkpoint_every = checkpoint_every
        self.run_dir = run_dir
        if self.run_dir is not None:
            os.makedirs(self.run_dir, exist_ok=True)

    def train(self, updates: int, start: int = 1) -> None:
        """Run `updates` PPO updates, numbering them from `start` (for resume)."""
        self.logger.info(f"Starting PPO training for {updates} updates")
        for update in range(start, updates + 1):
            metrics = self._update()
            summary = self.logger.summary(last_n=10)
            self._log_update(update, updates, metrics, summary)
            if update % self.checkpoint_every == 0 or update == updates:
                self._save_checkpoint(update, metrics, summary)
        if self.run_dir is not None:
            self.logger.save_episodes(os.path.join(self.run_dir, "episodes.json"))
        self.logger.info("Training finished")
        self.checkpoints.finalize()

    def resume(self) -> int:
        """Restore the agent from the latest checkpoint. Returns the last completed update."""
        payload = self.checkpoints.load_latest(self.agent.policy_model, self.agent.policy_optimizer)
        if payload is None:
            raise FileNotFoundError(f"No checkpoint found in {self.checkpoints.directory}")
        self.agent.load_state_dicts(payload.get("additional_state") or {})
        self.logger.info(f"Resumed from {self.checkpoints.latest_path()} at update {payload['step']}")
        return payload["step"]

    def _update(self) -> dict:
        """Collect one rollout and run the PPO update on it."""
        self.runner.collect_rollout(self.buffer.capacity, self.buffer)
        data = self.buffer.get()

        last_value = self.agent.value_of(data["last_observation"])
        advantages, returns = self.agent.compute_gae(
            data["rewards"], data["dones"], data["values"], last_value
        )
        metrics = self.agent.update(
            data["observations"], data["actions"], data["log_probs"], advantages, returns
        )
        self.buffer.clear()
        return metrics

    def _log_update(self, update: int, updates: int, metrics: dict, summary: dict) -> None:
        self.logger.info(
            f"update {update}/{updates} | "
            f"policy_loss={metrics['policy_loss']:.4f} | "
            f"value_loss={metrics['value_loss']:.4f} | "
            f"episodes={summary['num_episodes']} | "
            f"mean_reward={summary.get('mean_reward', 0.0):.1f} | "
            f"mean_distance={summary.get('mean_distance', 0.0):.1f} | "
            f"best_distance={summary.get('best_distance', 0.0):.1f}"
        )
        if self.run_dir is not None:
            record = {
                "update": update,
                **metrics,
                "mean_reward": summary.get("mean_reward"),
                "mean_distance": summary.get("mean_distance"),
                "best_distance": summary.get("best_distance"),
                "timestamp": datetime.now().isoformat(),
            }
            with open(os.path.join(self.run_dir, "metrics.jsonl"), "a") as f:
                f.write(json.dumps(record) + "\n")

    def _save_checkpoint(self, update: int, metrics: dict, summary: dict) -> None:
        # The policy model is what evaluation/playback needs; the value model and
        # both optimizers ride along in additional_state so resume restores the
        # full agent.
        score = summary.get("best_distance")
        path = self.checkpoints.save(
            self.agent.policy_model,
            self.agent.policy_optimizer,
            step=update,
            metrics=metrics,
            score=score,
            additional_state=self.agent.state_dicts(),
        )
        self.logger.info(f"Saved checkpoint to {path}")
