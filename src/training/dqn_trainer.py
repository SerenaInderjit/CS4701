import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

from src.evaluation.evaluation import evaluate
from src.policies.dqn_policy import DQNPolicy
from src.training.checkpoint import CheckpointManager
from src.training.logger import Logger


class DQNTrainer:
    """DQN training loop: act -> store transition -> sample batch -> gradient step.

    How it differs from the PPO loop (src/training/trainer.py):

      * **Off-policy.** Transitions go into a replay buffer and are reused, so
        the behaviour that collected them does not have to match the policy
        being learned. This is what lets the agent learn from old data.
      * **One gradient step per environment step** after a burn-in, instead of
        `epochs` passes over a freshly collected rollout.
      * **A target network** that is resynced every `target_update_freq`
        gradient steps, so the TD target stays fixed while we regress on it.
      * **Epsilon-greedy exploration**, annealed linearly over the run.
      * **A greedy evaluation pass** every `eval_every` steps, run on a
        separate environment so it does not disturb the training episode.

    The run directory, if given, gets one JSON line per logging interval in
    metrics.jsonl and the episode log in episodes.json.
    """

    def __init__(
        self,
        env,
        agent,
        buffer,
        checkpoint_dir: Optional[str] = None,
        logger: Optional[Logger] = None,
        checkpoint_every: int = 10000,
        run_dir: Optional[str] = None,
        lr_anneal: bool = False,
        train_start: int = 1000,
        log_every: int = 1000,
        eval_every: int = 0,
        eval_episodes: int = 5,
        eval_env=None,
        eval_timeout: int = 2000,
    ):
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.logger = logger or Logger(keep_step_records=False)
        self.checkpoint_every = checkpoint_every
        self.run_dir = run_dir
        self.lr_anneal = lr_anneal
        self.train_start = train_start
        self.log_every = max(1, log_every)
        self.eval_every = eval_every
        self.eval_episodes = eval_episodes
        self.eval_env = eval_env
        self.eval_timeout = eval_timeout

        if checkpoint_dir is None:
            checkpoint_dir = (
                os.path.join(run_dir, "checkpoints") if run_dir else "checkpoints"
            )
        self.checkpoints = CheckpointManager(checkpoint_dir)
        self._base_lr = getattr(agent, "learning_rate", None)

        self._observation = None
        self._eval_policy = None
        self._last_metrics: Dict[str, float] = {}
        self._last_eval: Optional[Dict[str, Any]] = None
        self._eval_for_log: Optional[Dict[str, Any]] = None
        self._start_time = None

        if self.run_dir is not None:
            os.makedirs(self.run_dir, exist_ok=True)

    # ------------------------------------------------------------------ driving

    def train(self, total_steps: int, start: int = 1) -> None:
        """Run DQN until `total_steps` environment steps have been taken."""
        self.logger.info(f"Starting DQN training for {total_steps} steps")
        self._observation = self.env.reset()["observation"]
        self._start_time = time.time()

        for step in range(start, total_steps + 1):
            self._anneal_lr(step, total_steps)
            self._interact()
            self.agent.anneal_epsilon()

            if step >= self.train_start and self.buffer.is_ready(self.agent.batch_size):
                batch = self.buffer.sample(self.agent.batch_size)
                self._last_metrics = self.agent.train_step(batch)

            should_log = step % self.log_every == 0 or step == total_steps
            should_eval = bool(self.eval_every) and step % self.eval_every == 0

            # Evaluate before logging, so the row for this step carries the
            # evaluation numbers instead of the previous evaluation's.
            if should_eval:
                self._last_eval = self._evaluate()
                self._eval_for_log = self._last_eval

            # One row per step: a step that both logs and evaluates is logged
            # once, not twice.
            if should_log or should_eval:
                self._log_step(step, total_steps)

            if step % self.checkpoint_every == 0 or step == total_steps:
                self._save_checkpoint(step)

        if self.run_dir is not None:
            self.logger.save_episodes(os.path.join(self.run_dir, "episodes.json"))
        self.logger.info("Training finished")
        self.checkpoints.finalize()

    def resume(self) -> int:
        """Restore the agent from the latest checkpoint. Returns the last step."""
        payload = self.checkpoints.load_latest(self.agent.model, self.agent.optimizer)
        if payload is None:
            raise FileNotFoundError(
                f"No checkpoint found in {self.checkpoints.directory}. "
                "Point --run-dir and --checkpoint-dir at an existing run "
                "(the directory must contain latest.pt or checkpoint_<step>.pt), "
                "or drop --resume to start a new run."
            )
        self.agent.load_state_dicts(payload.get("additional_state") or {})
        self.logger.info(
            f"Resumed from {self.checkpoints.latest_path()} at step {payload['step']}"
        )
        return payload["step"]

    def _anneal_lr(self, step: int, total_steps: int) -> None:
        if not self.lr_anneal or self._base_lr is None:
            return
        if not hasattr(self.agent, "set_learning_rate"):
            return
        fraction = 1.0 - (step - 1) / total_steps
        self.agent.set_learning_rate(self._base_lr * fraction)

    def _interact(self) -> None:
        """Take one environment step and store the transition in the buffer."""
        observation = self._observation
        action = self.agent.act(observation)
        result = self.env.step(action)

        self.buffer.add(
            observation,
            action,
            result["reward"],
            result["episode_ended"],
            result["observation"],
        )

        self.logger.log_step(
            action=action,
            reward=result["reward"],
            info=result["info"],
        )

        if result["episode_ended"]:
            self.logger.end_episode()
            self._observation = self.env.reset()["observation"]
        else:
            self._observation = result["observation"]

    # ---------------------------------------------------------------- evaluation

    def _evaluate(self) -> Optional[Dict[str, Any]]:
        """Run greedy episodes on the evaluation environment."""
        if self.eval_env is None:
            return None
        if self._eval_policy is None:
            self._eval_policy = DQNPolicy(self.agent.model, device=str(self.agent.device), epsilon=0.0)
        results = evaluate(
            self.eval_env,
            self._eval_policy,
            num_episodes=self.eval_episodes,
            timeout=self.eval_timeout,
        )
        # evaluate() puts the network in eval mode; the agent trains on it.
        self.agent.model.train()
        return results

    # ------------------------------------------------------------------ logging

    def _log_step(self, step: int, total_steps: int) -> None:
        summary = self.logger.summary(last_n=10)
        metrics = self._last_metrics
        elapsed = time.time() - self._start_time if self._start_time else 0.0
        speed = step / elapsed if elapsed > 0 else 0.0

        self.logger.info(
            f"step {step}/{total_steps} | "
            f"loss={metrics.get('loss', float('nan')):.4f} | "
            f"td_error={metrics.get('td_error', float('nan')):.4f} | "
            f"q_mean={metrics.get('q_mean', float('nan')):.3f} | "
            f"epsilon={self.agent.epsilon:.3f} | "
            f"episodes={summary.get('num_episodes', 0)} | "
            f"mean_reward={summary.get('mean_reward', 0.0):.1f} | "
            f"mean_distance={summary.get('mean_distance', 0.0):.1f} | "
            f"best_distance={summary.get('best_distance', 0.0):.1f} | "
            f"{speed:.1f} steps/s"
        )

        if self.run_dir is None:
            return

        record = {
            "step": step,
            "update": self.agent.updates,
            "loss": metrics.get("loss"),
            "td_error": metrics.get("td_error"),
            "q_mean": metrics.get("q_mean"),
            "grad_norm": metrics.get("grad_norm"),
            "epsilon": self.agent.epsilon,
            "mean_reward": summary.get("mean_reward"),
            "mean_distance": summary.get("mean_distance"),
            "best_distance": summary.get("best_distance"),
            "completion_rate": summary.get("completion_rate"),
            "timestamp": datetime.now().isoformat(),
        }
        if self._eval_for_log is not None:
            record["eval_mean_distance"] = self._eval_for_log.get("mean_distance")
            record["eval_completion_rate"] = self._eval_for_log.get("completion_rate")
        with open(os.path.join(self.run_dir, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(record) + "\n")
        # The evaluation belongs to the row that was just written, not to the
        # next one.
        self._eval_for_log = None

    def _save_checkpoint(self, step: int) -> None:
        # Score = mean distance over the last 10 episodes (less noisy than the
        # single best episode). The full agent state rides along in
        # additional_state so a resume restores the target network too.
        summary = self.logger.summary(last_n=10)
        path = self.checkpoints.save(
            self.agent.model,
            self.agent.optimizer,
            step=step,
            metrics=self._last_metrics,
            score=summary.get("mean_distance"),
            additional_state=self.agent.state_dicts(),
        )
        self.logger.info(f"Saved checkpoint to {path}")
