"""Run the whole baseline analysis pipeline end to end:
   evaluate baselines -> analyze statistics -> generate plots.

Usage:
    python scripts/full_analysis.py --episodes 20 --timeout 2000
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_step(cmd, env):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, env=env)
    if proc.returncode != 0:
        sys.exit(f"Error: step failed with exit code {proc.returncode}: {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=2000)
    parser.add_argument("--config", default="configs/baselines.yaml")
    args = parser.parse_args()

    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT

    run_step(
        [sys.executable, "scripts/evaluate_baseline.py",
         "--episodes", str(args.episodes), "--timeout", str(args.timeout),
         "--config", args.config],
        env,
    )
    run_step([sys.executable, "scripts/analyze_results.py"], env)
    run_step([sys.executable, "scripts/plot_results.py"], env)
    print("\nFull analysis complete: data/results/, data/plots/")


if __name__ == "__main__":
    main()
