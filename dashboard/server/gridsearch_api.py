"""Grid search control."""
import json
import os
import subprocess
import sys
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from .common import ROOT

router = APIRouter()


@router.get("/api/gridsearch/status")
async def api_gridsearch_status():
    status_file = ROOT / "data" / "grid_search_status.json"
    if not status_file.exists():
        return []
    with open(status_file) as f:
        return json.load(f)


@router.post("/api/gridsearch")
async def api_gridsearch(config: Dict[str, Any]):
    """API: launch a hyper-parameter grid search (n processes at a time)."""
    grid = config.get("grid")
    if not grid or not isinstance(grid, dict):
        raise HTTPException(status_code=400, detail="grid is required (JSON dict)")
    parallel = int(config.get("parallel", 4))
    updates = int(config.get("updates", 50))
    base_config = config.get("config", "configs/ppo.yaml")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    log_file = open(ROOT / "data" / "grid_search.log", "w")
    subprocess.Popen(
        [sys.executable, "scripts/grid_search.py", "--config", base_config,
         "--grid", json.dumps(grid), "--updates", str(updates),
         "--parallel", str(parallel)],
        cwd=ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT,
    )
    return {"status": "started", "parallel": parallel, "updates": updates}
