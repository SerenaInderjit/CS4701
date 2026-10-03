"""Baseline results: statistics, episodes, plots, compare, full analysis."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from .common import PLOTS_DIR, RESULTS_DIR, ROOT, RUNS_DIR

router = APIRouter()


@router.get("/api/results/statistics")
async def api_results_statistics():
    """API: baseline statistics computed by analyze_results.py."""
    stats_file = RESULTS_DIR / "statistics.json"
    if not stats_file.exists():
        raise HTTPException(status_code=404, detail="statistics.json not found — run refresh")
    with open(stats_file) as f:
        return json.load(f)


@router.get("/api/results/episodes")
async def api_results_episodes():
    """API: raw per-episode baseline results."""
    raw_file = RESULTS_DIR / "baselines.json"
    if not raw_file.exists():
        raise HTTPException(status_code=404, detail="baselines.json not found — run eval first")
    with open(raw_file) as f:
        return json.load(f)


@router.get("/api/results/plots")
async def api_results_plots():
    """API: list available plot files."""
    if not PLOTS_DIR.exists():
        return []
    return sorted(
        str(f.relative_to(PLOTS_DIR))
        for f in PLOTS_DIR.rglob("*.png")
        if "compare" not in f.relative_to(PLOTS_DIR).parts
    )


@router.get("/api/results/plot/{name:path}")
async def api_results_plot(name: str):
    """API: serve a generated plot image."""
    if ".." in name or not name.endswith(".png"):
        raise HTTPException(status_code=400, detail="Invalid plot name")
    plot_path = (PLOTS_DIR / name).resolve()
    if not str(plot_path).startswith(str(PLOTS_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid plot name")
    if not plot_path.exists():
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(plot_path)


@router.post("/api/results/refresh")
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


@router.get("/api/results/available")
async def api_results_available():
    """API: labels usable in policy comparison (baseline policies + run ids)."""
    labels = []
    raw_file = RESULTS_DIR / "baselines.json"
    if raw_file.exists():
        try:
            with open(raw_file) as f:
                episodes = json.load(f)
            labels.extend(sorted({e.get("policy") for e in episodes if isinstance(e, dict)} - {None}))
        except (json.JSONDecodeError, AttributeError):
            pass
    if RUNS_DIR.exists():
        for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
            if (run_dir / "episodes.json").exists():
                labels.append(run_dir.name)
    return labels


@router.post("/api/results/compare")
async def api_results_compare(payload: Dict[str, Any]):
    """API: plot one metric's episode curve for each selected policy/run."""
    labels = payload.get("labels") or []
    metric = payload.get("metric", "max_x_pos")
    if len(labels) < 2:
        raise HTTPException(status_code=400, detail="Select at least 2 policies/runs")
    if metric not in ("max_x_pos", "total_reward", "length"):
        raise HTTPException(status_code=400, detail="Unknown metric")

    series = {}
    for label in labels:
        episodes = _episodes_for_label(label)
        if not episodes:
            raise HTTPException(status_code=404, detail=f"No episodes found for '{label}'")
        episodes = sorted(episodes, key=lambda e: e.get("episode", 0))
        series[label] = (
            [e.get("episode", i) for i, e in enumerate(episodes)],
            [e.get(metric) for e in episodes],
        )

    fig, ax = plt.subplots(figsize=(10, 4))
    for label, (x, y) in series.items():
        ax.plot(x, y, label=label)
    ax.set_xlabel("Episode")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by episode")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    digest = hashlib.md5(("|".join(sorted(labels)) + metric).encode()).hexdigest()[:8]
    compare_dir = PLOTS_DIR / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    plot_path = compare_dir / f"compare_{digest}.png"
    fig.savefig(plot_path, dpi=100)
    plt.close(fig)
    return FileResponse(plot_path)


def _episodes_for_label(label: str) -> List[Dict[str, Any]]:
    raw_file = RESULTS_DIR / "baselines.json"
    if raw_file.exists():
        try:
            with open(raw_file) as f:
                episodes = json.load(f)
            matches = [e for e in episodes if isinstance(e, dict) and e.get("policy") == label]
            if matches:
                return matches
        except (json.JSONDecodeError, AttributeError):
            pass
    episodes_file = RUNS_DIR / label / "episodes.json"
    if episodes_file.exists():
        with open(episodes_file) as f:
            return json.load(f)
    return []


@router.post("/api/results/full_analysis")
async def api_results_full_analysis(config: Dict[str, Any]):
    """API: run the full pipeline (eval -> analyze -> plot)."""
    episodes = int(config.get("episodes", 10))
    timeout = int(config.get("timeout", 2000))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.run(
        [sys.executable, "scripts/full_analysis.py",
         "--episodes", str(episodes), "--timeout", str(timeout)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
        raise HTTPException(status_code=400, detail="\n".join(tail))
    return {"status": "ok", "output": proc.stdout[-2000:]}
