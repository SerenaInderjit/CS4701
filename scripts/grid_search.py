"""Run a hyper-parameter grid of short training runs, N at a time.

Usage:
    python scripts/grid_search.py --config configs/ppo.yaml --parallel 4 \
        --grid '{"learning_rate": [0.0003, 0.001], "epsilon": [0.1, 0.2]}' \
        --updates 50

Each combo writes a run dir named with a hyper-parameter slug:
    data/runs/ppo_lr0.0003_eps0.2_s0_20260101_120000/
Progress is recorded in data/grid_search_status.json.
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATUS_FILE = os.path.join(ROOT, "data", "grid_search_status.json")


def slug(combo):
    parts = []
    for key in sorted(combo):
        value = combo[key]
        short = {
            "learning_rate": "lr",
            "minibatch_size": "mbs",
            "rollout_size": "rs",
            "gamma": "g",
            "gae_lambda": "lam",
            "epsilon": "eps",
            "epochs": "ep",
        }.get(key, key.replace("_", ""))
        parts.append(f"{short}{value}")
    return "_".join(parts)


def run_combo(combo, base_config, updates):
    run_id = f"ppo_{slug(combo)}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = os.path.join(ROOT, "data", "runs", run_id)
    cmd = [sys.executable, "scripts/train.py", "--config", base_config,
           "--updates", str(updates), "--run-dir", run_dir]
    for key, value in combo.items():
        cmd += ["--" + key.replace("_", "-"), str(value)]
    log_path = run_dir + ".log"
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    return {"run_id": run_id, "combo": combo, "returncode": proc.returncode,
            "log": log_path}


def write_status(results):
    os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
    tmp = STATUS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, STATUS_FILE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ppo.yaml")
    parser.add_argument("--grid", required=True, help="JSON dict of param -> list")
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--parallel", type=int, default=4)
    args = parser.parse_args()

    grid = json.loads(args.grid)
    keys = list(grid)
    combos = [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]
    print(f"Grid search: {len(combos)} runs, {args.parallel} at a time")

    results = []
    write_status(results)
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(run_combo, combo, args.config, args.updates) for combo in combos]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            write_status(results)
            status = "ok" if result["returncode"] == 0 else f"FAILED ({result['returncode']})"
            print(f"[{status}] {result['run_id']} -> {result['log']}", flush=True)

    print(f"\nGrid search done. Status: {STATUS_FILE}")


if __name__ == "__main__":
    main()
