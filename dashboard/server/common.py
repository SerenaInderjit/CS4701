"""Shared constants and template/static wiring."""
from pathlib import Path

from fastapi.templating import Jinja2Templates

from src.paths import PLOTS_DIR as _PLOTS_DIR
from src.paths import RESULTS_DIR as _RESULTS_DIR
from src.paths import RUNS_DIR as _RUNS_DIR

SERVER_DIR = Path(__file__).resolve().parent          # dashboard/server/
DASHBOARD_DIR = SERVER_DIR.parent                     # dashboard/
ROOT = DASHBOARD_DIR.parent                           # repo root

RUNS_DIR = Path(_RUNS_DIR)
CONFIGS_DIR = ROOT / "configs"
RESULTS_DIR = Path(_RESULTS_DIR)
PLOTS_DIR = Path(_PLOTS_DIR)
FRAME_DIR = DASHBOARD_DIR / "frames"

templates = Jinja2Templates(directory=DASHBOARD_DIR / "templates")
