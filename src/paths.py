"""Single root for all file-system "database" artifacts.

Override with the MARIO_DATA_DIR environment variable.
"""
import os

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("MARIO_DATA_DIR", os.path.join(_root, "data"))
RUNS_DIR = os.path.join(DATA_DIR, "runs")
CHECKPOINTS_DIR = os.path.join(DATA_DIR, "checkpoints")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")
