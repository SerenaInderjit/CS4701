import json
import time
from typing import Any, Dict, Optional

from src.training.logger import Logger, _json_default
from src.training.runner import Runner

def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    return str(value)

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

    results["episodes"] = logger.episodes
    results["wall_time_seconds"] = wall_time
    results["steps_per_second"] = (
        total_steps / wall_time if wall_time > 0 else 0.0
    )

    return results


def format_evaluation(name: str, results: Dict[str, Any]) -> str:
    return (
        f"{name}: "
        f"episodes={results['num_episodes']}  "
        f"mean_reward={results['mean_reward']:.2f}  "
        f"mean_distance={results['mean_distance']:.2f}  "
        f"best_distance={results['best_distance']}  "
        f"completion_rate={results['completion_rate']:.2%}  "
        f"mean_episode_length={results['mean_episode_length']:.2f}  "
        f"wall_time={results['wall_time_seconds']:.2f}s"
    )


def save_evaluation(path: str, all_results: Dict[str, Dict[str, Any]], raw=False) -> None:
    if raw:
        all_episodes = []

        for policy_name, policy_results in all_results.items():
            for episode in policy_results["episodes"]:
                episode_data = episode.copy()
                episode_data["policy"] = policy_name
                all_episodes.append(episode_data)

        with open(path, "w") as f:
            json.dump(all_episodes, f, indent=2, default=_json_default)

    else:
        with open(path, "w") as f:
            json.dump(all_results, f, indent=2, default=_json_default)