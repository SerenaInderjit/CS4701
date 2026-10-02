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

from src.paths import CHECKPOINTS_DIR, RUNS_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="e.g. username/mario-ppo")
    parser.add_argument("--run-id", default=None, help="only push this run (default: all in data/runs)")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    checkpoints_dir = CHECKPOINTS_DIR
    uploads = []

    run_ids = [args.run_id] if args.run_id else sorted(
        d for d in os.listdir(checkpoints_dir)
        if os.path.isdir(os.path.join(checkpoints_dir, d))
    )
    for run_id in run_ids:
        run_ckpts = os.path.join(checkpoints_dir, run_id)
        if not os.path.isdir(run_ckpts):
            continue
        for name in sorted(os.listdir(run_ckpts)):
            src_path = os.path.join(run_ckpts, name)
            if name.endswith((".pt", ".pth")):
                api.upload_file(
                    path_or_fileobj=src_path,
                    path_in_repo=f"{run_id}/{name}",
                    repo_id=args.repo_id,
                    repo_type="model",
                )
                uploads.append(f"{run_id}/{name}")

        config_path = os.path.join(RUNS_DIR, run_id, "config.yaml")
        if os.path.exists(config_path):
            api.upload_file(
                path_or_fileobj=config_path,
                path_in_repo=f"{run_id}/config.yaml",
                repo_id=args.repo_id,
                repo_type="model",
            )
            uploads.append(f"{run_id}/config.yaml")

    if not uploads:
        sys.exit(f"Nothing to upload — no checkpoints found in {checkpoints_dir}")
    print(f"Uploaded {len(uploads)} file(s) to https://huggingface.co/{args.repo_id}")
    for path in uploads:
        print(f"  {path}")


if __name__ == "__main__":
    main()
