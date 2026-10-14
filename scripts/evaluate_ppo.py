"""Evaluate a trained PPO checkpoint with the same harness as the baselines.

Usage:
    python scripts/evaluate_ppo.py --run data/runs/ppo_20261012_144448 --episodes 100

Evaluates the checkpoint twice -- 'ppo_sampled' (samples actions, as in
training) and 'ppo_greedy' (argmax action) -- and merges both into
data/results/baselines.json next to the baseline policies, so
analyze_results.py, plot_results.py and the dashboard compare them directly.

The network architecture and environment settings are read from the run's
config.yaml, so they always match what the checkpoint was trained with.
"""
import argparse
import json
import math
import os

from src.algorithms.cnn import ConvolutionalNeuralNetwork
from src.environment.mario_environment import make_training_environment
from src.environment.reward import RewardConfig
from src.evaluation.evaluation import evaluate, format_evaluation, save_evaluation
from src.policies.ppo_policy import PPOPolicy
from src.training.config import load_config, resolve_device


def wilson_interval(successes, n, z=1.96):
    """95% Wilson confidence interval for a completion rate."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def reward_config_from(config):
    base = RewardConfig()
    s = config.get("reward_scale", 1.0)
    return RewardConfig(
        x_progress_weight=config.get("x_progress_weight", base.x_progress_weight) * s,
        time_penalty_weight=config.get("time_penalty_weight", base.time_penalty_weight) * s,
        death_penalty=config.get("death_penalty", base.death_penalty) * s,
        flag_bonus=config.get("flag_bonus", base.flag_bonus) * s,
        reward_clip=config.get("reward_clip", base.reward_clip) * s,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="run directory, e.g. data/runs/ppo_...")
    parser.add_argument("--checkpoint", default=None, help="default: <run>/checkpoints/best.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=2000, help="max agent steps per episode")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="data/results/baselines.json")
    parser.add_argument("--name", default="ppo", help="prefix for the policy labels")
    args = parser.parse_args()

    checkpoint = args.checkpoint or os.path.join(args.run, "checkpoints", "best.pt")
    config = load_config(os.path.join(args.run, "config.yaml"))
    device = resolve_device(args.device)

    env = make_training_environment(
        frame_skip=config["frame_skip"],
        num_stack=config["num_stack"],
        frame_size=(config["frame_size"], config["frame_size"]),
        reward_config=reward_config_from(config),
    )

    all_results = {}
    try:
        for label, deterministic in ((f"{args.name}_sampled", False), (f"{args.name}_greedy", True)):
            model = ConvolutionalNeuralNetwork(
                output_dim=env.num_actions,
                conv_channels=tuple(config.get("conv_channels", [64, 128, 128])),
                hidden_dim=config.get("hidden_dim", 1024),
                frame_size=config["frame_size"],
            )
            policy = PPOPolicy(model, checkpoint_path=checkpoint, device=device, deterministic=deterministic)
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
