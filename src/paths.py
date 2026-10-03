"""Single root for all file-system "database" artifacts.

Override with the MARIO_DATA_DIR environment variable.

Layout:
    data/
        runs/<run_id>/          one full training run (self-contained)
            config.yaml
            run.json
            metrics.jsonl
            episodes.json
            checkpoints/        latest.pt, best.pt, checkpoint_<step>.pt
            plots/              run metric plots
        results/                baseline evaluation artifacts (baselines.json,
                                statistics.json)
        plots/                  baseline comparison plots (tracked in git)
"""
import os

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("MARIO_DATA_DIR", os.path.join(_root, "data"))
RUNS_DIR = os.path.join(DATA_DIR, "runs")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")


def run_dir(run_id: str) -> str:
    return os.path.join(RUNS_DIR, run_id)


def run_checkpoints_dir(run_id: str) -> str:
    return os.path.join(RUNS_DIR, run_id, "checkpoints")


def run_plots_dir(run_id: str) -> str:
    return os.path.join(RUNS_DIR, run_id, "plots")
