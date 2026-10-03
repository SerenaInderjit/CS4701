"""FastAPI app: wiring + HTML page routes."""
from pathlib import Path

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request

from .common import CONFIGS_DIR, DASHBOARD_DIR, ROOT, RUNS_DIR, templates
from .gridsearch_api import router as gridsearch_router
from .play_api import router as play_router
from .results_api import router as results_router
from .runs_api import get_runs, router as runs_router
from .train_api import router as train_router

app = FastAPI(title="RL Training Dashboard")
app.mount("/static", StaticFiles(directory=DASHBOARD_DIR / "static"), name="static")

app.include_router(runs_router)
app.include_router(results_router)
app.include_router(train_router)
app.include_router(play_router)
app.include_router(gridsearch_router)


@app.get("/")
async def index(request: Request):
    """Overview page: list all runs."""
    return templates.TemplateResponse(request, "index.html", {"runs": get_runs()})


@app.get("/run/{run_id}")
async def run_detail(request: Request, run_id: str):
    """Run detail page: metrics, plots, config."""
    run = next((r for r in get_runs() if r["id"] == run_id), None)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return templates.TemplateResponse(request, "run.html", {"run": run})


@app.get("/train")
async def train_page(request: Request):
    """Training control page."""
    configs = []
    if CONFIGS_DIR.exists():
        for f in CONFIGS_DIR.glob("*.yaml"):
            with open(f) as fh:
                configs.append({"name": f.stem, "config": yaml.safe_load(fh)})
    return templates.TemplateResponse(request, "train.html", {"configs": configs})


@app.get("/gridsearch")
async def gridsearch_page(request: Request):
    """Grid search page: launch param sweeps."""
    return templates.TemplateResponse(request, "gridsearch.html", {})


@app.get("/play")
async def play_page(request: Request):
    """Play page with live game view."""
    checkpoints = []
    if RUNS_DIR.exists():
        for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
            ck_dir = run_dir / "checkpoints"
            if ck_dir.is_dir():
                for f in sorted(ck_dir.glob("*.pt")):
                    checkpoints.append(str(f.relative_to(ROOT)))
    return templates.TemplateResponse(request, "play.html", {"checkpoints": checkpoints})


@app.get("/results")
async def results_page(request: Request):
    """Results page: baseline statistics and plots."""
    return templates.TemplateResponse(request, "results.html", {})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
