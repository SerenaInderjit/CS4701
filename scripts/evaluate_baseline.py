"""Evaluate the baseline policies (random, always-right) with the shared harness.

Usage: ./run.sh eval  (or: python scripts/evaluate_baseline.py --episodes 5 --render)
"""
import argparse
import json
import os

import yaml

from src.environment.mario_environment import make_training_environment
from src.evaluation.evaluation import evaluate, format_evaluation, save_evaluation
from src.policies.baselines import ConstantPolicy, RandomPolicy


def build_policies(config_path, num_actions):
    with open(config_path) as f:
        names = yaml.safe_load(f)["policies"]
    policies = {}
    for name in names:
        if name == "random":
            policies[name] = RandomPolicy(num_actions=num_actions, seed=0)
        elif name == "always_right":
            policies[name] = ConstantPolicy(action=1)
        elif name == "zero":
            policies[name] = ConstantPolicy(action=0)
        else:
            raise ValueError(f"Unknown baseline policy: {name}")
    return policies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=2000, help="max agent steps per episode")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", default="data/results/baselines.json")
    parser.add_argument("--config", default="configs/baselines.yaml")
    args = parser.parse_args()

    environment = make_training_environment()
    policies = build_policies(args.config, environment.num_actions)

    all_results = {}
    try:
        for name, policy in policies.items():
            results = evaluate(
                environment,
                policy,
                num_episodes=args.episodes,
                timeout=args.timeout,
                render=args.render,
            )
            all_results[name] = results
            print(format_evaluation(name, results))
    finally:
        environment.close()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    merge_existing = []
    if os.path.exists(args.output):
        try:
            with open(args.output) as f:
                existing = json.load(f)
            if isinstance(existing, list):
                reran = set(all_results)
                merge_existing = [e for e in existing if e.get("policy") not in reran]
        except (json.JSONDecodeError, OSError):
            merge_existing = []
    save_evaluation(args.output, all_results, raw=True, keep=merge_existing)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
