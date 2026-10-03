"""Upload local checkpoints to Hugging Face Hub.

Layout on the Hub (one folder per run):

    <repo_id>/
        ppo_20260930_152258/
            config.yaml
            best.pt
            checkpoint_000001000.pt
        ...

Usage:
    python scripts/push_weights.py --repo-id <user>/mario-ppo [--run-id ppo_...]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import HfApi

from src.paths import RUNS_DIR


def best_distance_of(run_path):
    """Max best_distance across the run's metrics.jsonl; -inf if unknown."""
    import json

    metrics_file = os.path.join(run_path, "metrics.jsonl")
    best = float("-inf")
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            for line in f:
                try:
                    value = json.loads(line).get("best_distance")
                    if value is not None:
                        best = max(best, value)
                except json.JSONDecodeError:
                    continue
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="e.g. username/mario-ppo")
    parser.add_argument("--run-id", default=None, help="only push this run (default: all in data/runs)")
    parser.add_argument("--max-runs", type=int, default=10, help="top N runs by best distance")
    parser.add_argument("--include-top-k", action="store_true", help="also push the top-3 periodic checkpoints")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    uploads = []

    if args.run_id:
        run_ids = [args.run_id]
    else:
        candidates = []
        for run_id in os.listdir(RUNS_DIR):
            run_path = os.path.join(RUNS_DIR, run_id)
            if os.path.isdir(run_path):
                candidates.append((best_distance_of(run_path), run_id))
        # Rank runs by best distance; keep only the top --max-runs.
        candidates.sort(key=lambda item: item[0], reverse=True)
        run_ids = [run_id for _, run_id in candidates[: args.max_runs]]

    for run_id in run_ids:
        run_path = os.path.join(RUNS_DIR, run_id)
        ckpt_dir = os.path.join(run_path, "checkpoints")
        for meta in ("config.yaml", "run.json"):
            meta_path = os.path.join(run_path, meta)
            if os.path.exists(meta_path):
                api.upload_file(
                    path_or_fileobj=meta_path,
                    path_in_repo=f"{run_id}/{meta}",
                    repo_id=args.repo_id,
                    repo_type="model",
                )
                uploads.append(f"{run_id}/{meta}")

        if not os.path.isdir(ckpt_dir):
            continue
        # Always include best.pt; include latest.pt only while training is
        # unfinished (resumed value) to save space.
        wanted = ["best.pt"]
        if os.path.exists(os.path.join(ckpt_dir, "latest.pt")):
            wanted.append("latest.pt")
        for name in sorted(os.listdir(ckpt_dir)):
            if name in wanted or (args.include_top_k and name.startswith("checkpoint_")):
                api.upload_file(
                    path_or_fileobj=os.path.join(ckpt_dir, name),
                    path_in_repo=f"{run_id}/checkpoints/{name}",
                    repo_id=args.repo_id,
                    repo_type="model",
                )
                uploads.append(f"{run_id}/checkpoints/{name}")

    if not uploads:
        sys.exit(f"Nothing to upload — no runs with checkpoints in {RUNS_DIR}")
    print(f"Uploaded {len(uploads)} file(s) to https://huggingface.co/{args.repo_id}")
    for path in uploads:
        print(f"  {path}")


if __name__ == "__main__":
    main()
