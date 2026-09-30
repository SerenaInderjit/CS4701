import json
import time
from typing import Any, Dict, Optional

from src.training.logger import Logger
from src.training.runner import Runner


def evaluate(
    environment,
    policy,
    num_episodes: int = 10,
    timeout: Optional[int] = None,
    render: bool = False,
) -> Dict[str, Any]:
    """Play `num_episodes` episodes and return metrics that work for any policy.

    Every agent (random, PPO, DQN, ...) is evaluated through the same code path
    so results are directly comparable. Metrics:
      - mean_distance / best_distance: furthest x_pos reached per episode
      - completion_rate: fraction of episodes that reached the flag
      - mean_reward: mean episode return (shaped reward if the environment
        shapes it, so compare distance/completion across reward variants)
      - mean_episode_length: agent steps per episode (frame-skip applies)
      - steps_per_second / wall_time_seconds: runtime

    Note: gym-super-mario-bros ends an episode on every death, so "lives" is
    not a meaningful metric here; completion_rate and distance cover it.
    """
    logger = Logger(keep_step_records=False)
    runner = Runner(environment, policy, logger, render=render)

    start = time.time()
    for _ in range(num_episodes):
        runner.run_episode(timeout=timeout)
    wall_time = time.time() - start

    results = logger.summary()
    total_steps = sum(e["length"] for e in logger.episodes)
    results["wall_time_seconds"] = wall_time
    results["steps_per_second"] = total_steps / wall_time if wall_time > 0 else 0.0
    results["episodes"] = logger.episodes
    return results


def format_evaluation(name: str, results: Dict[str, Any]) -> str:
    return (
        f"{name}: "
        f"episodes={results['num_episodes']}  "
        f"mean_distance={results['mean_distance']:.1f}  "
        f"best_distance={results['best_distance']}  "
        f"completion_rate={results['completion_rate']:.0%}  "
        f"mean_reward={results['mean_reward']:.1f}  "
        f"mean_length={results['mean_episode_length']:.0f}  "
        f"steps/s={results['steps_per_second']:.0f}"
    )


def save_evaluation(path: str, all_results: Dict[str, Dict[str, Any]]) -> None:
    """Save {agent_name: results} to JSON for later plotting/comparison."""
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda v: v.item() if hasattr(v, "item") else str(v))
