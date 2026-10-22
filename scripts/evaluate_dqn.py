"""Evaluate a trained DQN checkpoint with the same harness as the baselines.

Usage:
    python scripts/evaluate_dqn.py --run data/runs/dqn_20261013_120000 --episodes 100

Evaluates the checkpoint twice and merges both into data/results/baselines.json
next to the baseline policies, so analyze_results.py, plot_results.py and the
dashboard compare them directly:

  * 'dqn_greedy'  : always takes the argmax action (epsilon = 0).
  * 'dqn_eps005'  : epsilon-greedy with epsilon = 0.05, the standard DQN
                     evaluation protocol. This is the honest performance
                     estimate, because the agent is trained with exploration.

The network architecture and environment settings are read from the run's
config.yaml, so they always match the checkpoint.
"""
import argparse
import json
import math
import os

from src.algorithms.cnn import ConvolutionalNeuralNetwork
from src.environment.mario_environment import make_training_environment
from src.environment.reward import reward_config_from_dict
from src.evaluation.evaluation import evaluate, format_evaluation, save_evaluation
from src.policies.dqn_policy import DQNPolicy
from src.training.config import load_config, resolve_device

# The standard DQN evaluation epsilon: 5% of the actions are random.
EVAL_EPSILON = 0.05


def wilson_interval(successes, n, z=1.96):
    """95% Wilson confidence interval for a completion rate."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="run directory, e.g. data/runs/dqn_...")
    parser.add_argument("--checkpoint", default=None, help="default: <run>/checkpoints/best.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=2000, help="max agent steps per episode")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="data/results/baselines.json")
    parser.add_argument("--name", default="dqn", help="prefix for the policy labels")
    args = parser.parse_args()

    checkpoint = args.checkpoint or os.path.join(args.run, "checkpoints", "best.pt")
    config = load_config(os.path.join(args.run, "config.yaml"))
    device = resolve_device(args.device)

    env = make_training_environment(
        frame_skip=config["frame_skip"],
        num_stack=config["num_stack"],
        frame_size=(config["frame_size"], config["frame_size"]),
        reward_config=reward_config_from_dict(config),
    )

    variants = (
        (f"{args.name}_greedy", 0.0),
        (f"{args.name}_eps{int(EVAL_EPSILON * 100):03d}", EVAL_EPSILON),
    )

    all_results = {}
    try:
        for label, epsilon in variants:
            model = ConvolutionalNeuralNetwork(
                output_dim=env.num_actions,
                conv_channels=tuple(config.get("conv_channels", [32, 64, 64])),
                hidden_dim=config.get("hidden_dim", 512),
                frame_size=config["frame_size"],
            )
            policy = DQNPolicy(model, checkpoint_path=checkpoint, device=device, epsilon=epsilon)
            results = evaluate(env, policy, num_episodes=args.episodes, timeout=args.timeout)
            all_results[label] = results
            print(format_evaluation(label, results))
            wins = sum(e["flag_get"] for e in results["episodes"])
            lo, hi = wilson_interval(wins, len(results["episodes"]))
            print(f"  completion {wins}/{len(results['episodes'])}  95% CI [{lo:.1%}, {hi:.1%}]")
    finally:
        env.close()

    keep = []
    if os.path.exists(args.output):
        try:
            with open(args.output) as f:
                existing = json.load(f)
            keep = [e for e in existing if e.get("policy") not in all_results]
        except (json.JSONDecodeError, OSError):
            keep = []
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_evaluation(args.output, all_results, raw=True, keep=keep)
    print(f"Merged results into {args.output} (checkpoint: {checkpoint})")


if __name__ == "__main__":
    main()
