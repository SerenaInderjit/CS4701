"""Evaluate the baseline policies (random, always-right) with the shared harness.

Usage: ./run.sh eval  (or: python scripts/evaluate_baseline.py --episodes 5 --render)
"""
import argparse
import os

from src.environment.mario_environment import make_training_environment
from src.evaluation.evaluation import evaluate, format_evaluation, save_evaluation
from src.policies.baselines import ConstantPolicy, RandomPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=2000, help="max agent steps per episode")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", default="results/baselines.json")
    args = parser.parse_args()

    environment = make_training_environment()
    policies = {
        "random": RandomPolicy(num_actions=environment.num_actions, seed=0),
        "always_right": ConstantPolicy(action=1),
        "zero": ConstantPolicy(action=0),
    }

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
    save_evaluation(args.output, all_results, raw=True)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
