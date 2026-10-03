"""Download checkpoints from Hugging Face Hub back into data/runs/<run_id>/.

Usage:
    python scripts/pull_weights.py --repo-id <user>/mario-ppo [--run-id ppo_...]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import snapshot_download

from src.paths import RUNS_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="e.g. username/mario-ppo")
    parser.add_argument("--run-id", default=None, help="only download this run's files")
    args = parser.parse_args()

    allow_patterns = [f"{args.run_id}/**"] if args.run_id else None
    local_dir = snapshot_download(repo_id=args.repo_id, allow_patterns=allow_patterns)

    moved = []
    for root, _dirs, files in os.walk(local_dir):
        for name in files:
            src_path = os.path.join(root, name)
            # Preserve the <run_id>/ folder structure from the Hub.
            rel_parts = os.path.relpath(src_path, local_dir).split(os.sep)
            run_id = rel_parts[0] if len(rel_parts) > 1 else "latest"
            if name.endswith((".pt", ".pth")):
                dst_dir = os.path.join(RUNS_DIR, run_id, "checkpoints")
            else:
                dst_dir = os.path.join(RUNS_DIR, run_id)
            os.makedirs(dst_dir, exist_ok=True)
            dst_path = os.path.join(dst_dir, name)
            os.replace(src_path, dst_path)
            moved.append(dst_path)

    print(f"Restored {len(moved)} file(s) under {RUNS_DIR}")
    for path in moved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
