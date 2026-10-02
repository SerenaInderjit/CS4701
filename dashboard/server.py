"""FastAPI dashboard for training monitoring and control.

Usage:
    cd dashboard
    uvicorn server:app --reload --port 8000

Then open http://localhost:8000
"""
import json
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
CHECKPOINTS_DIR = ROOT / "checkpoints"
CONFIGS_DIR = ROOT / "configs"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"

app = FastAPI(title="RL Training Dashboard")
app.mount("/static", StaticFiles(directory=Path(__file__).resolve().parent / "static"), name="static")
templates = Jinja2Templates(directory=Path(__file__).resolve().parent / "templates")

# Track running training processes
_training_process: Optional[subprocess.Popen] = None


def get_runs() -> List[Dict[str, Any]]:
    """List all runs with metadata."""
    runs = []
    if not RUNS_DIR.exists():
        return runs
    for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        run = {"id": run_dir.name, "path": str(run_dir)}
        run_json = run_dir / "run.json"
        if run_json.exists():
            with open(run_json) as f:
                run.update(json.load(f))
        config_yaml = run_dir / "config.yaml"
        if config_yaml.exists():
            with open(config_yaml) as f:
                run["config"] = yaml.safe_load(f)
        metrics_file = run_dir / "metrics.jsonl"
        if metrics_file.exists():
            metrics = []
            with open(metrics_file) as f:
                for line in f:
                    metrics.append(json.loads(line))
            run["metrics"] = metrics
            run["num_updates"] = len(metrics)
            if metrics:
                run["latest"] = metrics[-1]
                run["best_distance"] = max(
                    (m.get("best_distance", 0) for m in metrics if m.get("best_distance") is not None),
                    default=0,
                )
        episodes_file = run_dir / "episodes.json"
        if episodes_file.exists():
            with open(episodes_file) as f:
                run["episodes"] = json.load(f)
            run["finished"] = True
        else:
            run["finished"] = False
        runs.append(run)
    return runs


@app.get("/")
async def index(request: Request):
    """Overview page: list all runs."""
    runs = get_runs()
    return templates.TemplateResponse(request, "index.html", {"runs": runs})


@app.get("/run/{run_id}")
async def run_detail(request: Request, run_id: str):
    """Run detail page: metrics, plots, config."""
    runs = get_runs()
    run = next((r for r in runs if r["id"] == run_id), None)
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


@app.get("/play")
async def play_page(request: Request):
    """Play page with live game view."""
    checkpoints = []
    if CHECKPOINTS_DIR.exists():
        for f in CHECKPOINTS_DIR.glob("*.pt"):
            checkpoints.append(f.name)
    return templates.TemplateResponse(request, "play.html", {"checkpoints": checkpoints})


@app.get("/results")
async def results_page(request: Request):
    """Results page: baseline statistics and plots."""
    return templates.TemplateResponse(request, "results.html", {})


@app.get("/api/results/statistics")
async def api_results_statistics():
    """API: baseline statistics computed by analyze_results.py."""
    stats_file = RESULTS_DIR / "statistics.json"
    if not stats_file.exists():
        raise HTTPException(status_code=404, detail="statistics.json not found — run refresh")
    with open(stats_file) as f:
        return json.load(f)


@app.get("/api/results/episodes")
async def api_results_episodes():
    """API: raw per-episode baseline results."""
    raw_file = RESULTS_DIR / "baselines.json"
    if not raw_file.exists():
        raise HTTPException(status_code=404, detail="baselines.json not found — run eval first")
    with open(raw_file) as f:
        return json.load(f)


@app.get("/api/results/plots")
async def api_results_plots():
    """API: list available plot files."""
    if not PLOTS_DIR.exists():
        return []
    return sorted(f.name for f in PLOTS_DIR.glob("*.png"))


@app.get("/api/results/plot/{name}")
async def api_results_plot(name: str):
    """API: serve a generated plot image."""
    if "/" in name or ".." in name or not name.endswith(".png"):
        raise HTTPException(status_code=400, detail="Invalid plot name")
    plot_path = PLOTS_DIR / name
    if not plot_path.exists():
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(plot_path)


@app.post("/api/results/refresh")
async def api_results_refresh():
    """API: re-run analyze_results.py and plot_results.py."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    output = []
    for script in ("scripts/analyze_results.py", "scripts/plot_results.py"):
        proc = subprocess.run(
            [sys.executable, script],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        output.append(proc.stdout + proc.stderr)
        if proc.returncode != 0:
            clean = (proc.stderr or proc.stdout).strip().splitlines()
            message = clean[-1] if clean else "unknown error"
            raise HTTPException(status_code=400, detail=f"{script}: {message}")
    return {"status": "ok", "output": "\n".join(output)}


@app.get("/api/runs")
async def api_runs():
    """API: list all runs."""
    return get_runs()


@app.get("/api/run/{run_id}")
async def api_run(run_id: str):
    """API: get run details."""
    runs = get_runs()
    run = next((r for r in runs if r["id"] == run_id), None)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.get("/api/run/{run_id}/metrics")
async def api_metrics(run_id: str):
    """API: get metrics as JSON."""
    metrics_file = RUNS_DIR / run_id / "metrics.jsonl"
    if not metrics_file.exists():
        raise HTTPException(status_code=404, detail="Metrics not found")
    metrics = []
    with open(metrics_file) as f:
        for line in f:
            metrics.append(json.loads(line))
    return metrics


@app.get("/api/run/{run_id}/plot/{metric}")
async def api_plot(run_id: str, metric: str):
    """API: generate matplotlib plot for a metric."""
    metrics_file = RUNS_DIR / run_id / "metrics.jsonl"
    if not metrics_file.exists():
        raise HTTPException(status_code=404, detail="Metrics not found")
    metrics = []
    with open(metrics_file) as f:
        for line in f:
            metrics.append(json.loads(line))
    if not metrics:
        raise HTTPException(status_code=404, detail="No metrics available")

    values = [m.get(metric) for m in metrics if m.get(metric) is not None]
    if not values:
        raise HTTPException(status_code=404, detail=f"Metric '{metric}' not found")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(values)
    ax.set_xlabel("Update")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} — {run_id}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_dir = Path(__file__).resolve().parent / "plots"
    plot_dir.mkdir(exist_ok=True)
    plot_path = plot_dir / f"{run_id}_{metric}.png"
    fig.savefig(plot_path, dpi=100)
    plt.close(fig)
    return FileResponse(plot_path)


@app.post("/api/train")
async def api_train(config: Dict[str, Any]):
    """API: start training with a config."""
    global _training_process
    if _training_process is not None and _training_process.poll() is None:
        raise HTTPException(status_code=400, detail="Training already running")

    # Save config to a temp file
    config_path = CONFIGS_DIR / f"dashboard_{datetime.now():%Y%m%d_%H%M%S}.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

    # Start training
    log_file = open(RUNS_DIR / f"dashboard_train_{datetime.now():%Y%m%d_%H%M%S}.log", "w")
    _training_process = subprocess.Popen(
        [sys.executable, "scripts/train.py", "--config", str(config_path)],
        cwd=ROOT,
        stdout=log_file,
        stderr=log_file,
    )
    return {"status": "started", "pid": _training_process.pid}


@app.post("/api/train/stop")
async def api_train_stop():
    """API: stop training."""
    global _training_process
    if _training_process is None or _training_process.poll() is not None:
        raise HTTPException(status_code=400, detail="No training running")
    _training_process.send_signal(signal.SIGTERM)
    _training_process.wait(timeout=10)
    _training_process = None
    return {"status": "stopped"}


@app.get("/api/train/status")
async def api_train_status():
    """API: get training status."""
    if _training_process is None:
        return {"running": False}
    return {"running": _training_process.poll() is None, "pid": _training_process.pid}


@app.get("/api/configs")
async def api_configs():
    """API: list available configs."""
    configs = []
    if CONFIGS_DIR.exists():
        for f in CONFIGS_DIR.glob("*.yaml"):
            with open(f) as fh:
                configs.append({"name": f.stem, "config": yaml.safe_load(fh)})
    return configs


# Play state
_play_process: Optional[subprocess.Popen] = None
_play_results: str = ""
_play_frame_dir = Path(__file__).resolve().parent / "frames"


@app.post("/api/play/start")
async def api_play_start(config: Dict[str, Any]):
    """API: start playing with a checkpoint."""
    global _play_process, _play_results
    if _play_process is not None and _play_process.poll() is None:
        raise HTTPException(status_code=400, detail="Already playing")

    checkpoint = config.get("checkpoint", "checkpoints/best.pt")
    episodes = config.get("episodes", 3)
    fps = config.get("fps", 15)

    _play_frame_dir.mkdir(exist_ok=True)
    # Clear old frames
    for f in _play_frame_dir.glob("*.png"):
        f.unlink()

    _play_results = ""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    _play_process = subprocess.Popen(
        [sys.executable, "scripts/play_mario.py", "--policy", "ppo",
         "--checkpoint", checkpoint, "--episodes", str(episodes),
         "--device", "cpu", "--save-frames", str(_play_frame_dir)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {"status": "started", "pid": _play_process.pid}


@app.post("/api/play/stop")
async def api_play_stop():
    """API: stop playing."""
    global _play_process
    if _play_process is None or _play_process.poll() is not None:
        raise HTTPException(status_code=400, detail="Not playing")
    _play_process.send_signal(signal.SIGTERM)
    _play_process.wait(timeout=10)
    _play_process = None
    return {"status": "stopped"}


@app.get("/api/play/status")
async def api_play_status():
    """API: get play status."""
    if _play_process is None:
        return {"running": False, "results": _play_results}
    running = _play_process.poll() is None
    if not running and _play_process.stdout is not None:
        _play_results = _play_process.stdout.read().decode()
    return {"running": running, "results": _play_results}


@app.get("/api/play/frame")
async def api_play_frame():
    """API: get the latest game frame."""
    if not _play_frame_dir.exists():
        raise HTTPException(status_code=404, detail="No frames yet")
    frames = sorted(_play_frame_dir.glob("*.png"))
    if not frames:
        raise HTTPException(status_code=404, detail="No frames yet")
    return FileResponse(frames[-1])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
