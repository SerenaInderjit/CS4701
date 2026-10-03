"""Run inspection: get_runs() + /api/runs, /api/run/<id>, metrics, plots."""
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from .common import RUNS_DIR

router = APIRouter()


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


def _metrics(run_id: str) -> List[Dict[str, Any]]:
    metrics_file = RUNS_DIR / run_id / "metrics.jsonl"
    if not metrics_file.exists():
        raise HTTPException(status_code=404, detail="Metrics not found")
    metrics = []
    with open(metrics_file) as f:
        for line in f:
            metrics.append(json.loads(line))
    return metrics


@router.get("/api/runs")
async def api_runs():
    """API: list all runs."""
    return get_runs()


@router.get("/api/run/{run_id}")
async def api_run(run_id: str):
    """API: get run details."""
    run = next((r for r in get_runs() if r["id"] == run_id), None)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.get("/api/run/{run_id}/metrics")
async def api_metrics(run_id: str):
    """API: get metrics as JSON."""
    return _metrics(run_id)


@router.get("/api/run/{run_id}/plot/{metric}")
async def api_plot(run_id: str, metric: str):
    """API: generate matplotlib plot for a metric."""
    metrics = _metrics(run_id)
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

    plot_dir = RUNS_DIR / run_id / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / f"{run_id}_{metric}.png"
    fig.savefig(plot_path, dpi=100)
    plt.close(fig)
    return FileResponse(plot_path)
