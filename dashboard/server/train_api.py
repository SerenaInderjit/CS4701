"""Training control: start/stop/status + config listing."""
import os
import signal
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import yaml
from fastapi import APIRouter, HTTPException

from .common import CONFIGS_DIR, ROOT, RUNS_DIR

router = APIRouter()

# Track running training process
_training_process: Optional[subprocess.Popen] = None


@router.post("/api/train")
async def api_train(config: Dict[str, Any]):
    """API: start training with a config."""
    global _training_process
    if _training_process is not None and _training_process.poll() is None:
        raise HTTPException(status_code=400, detail="Training already running")

    config_path = ROOT / "data" / f"dashboard_{datetime.now():%Y%m%d_%H%M%S}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

    log_file = open(RUNS_DIR / f"dashboard_train_{datetime.now():%Y%m%d_%H%M%S}.log", "w")
    _training_process = subprocess.Popen(
        [sys.executable, "scripts/train.py", "--config", str(config_path)],
        cwd=ROOT,
        stdout=log_file,
        stderr=log_file,
    )
    return {"status": "started", "pid": _training_process.pid}


@router.post("/api/train/stop")
async def api_train_stop():
    """API: stop training."""
    global _training_process
    if _training_process is None or _training_process.poll() is not None:
        raise HTTPException(status_code=400, detail="No training running")
    _training_process.send_signal(signal.SIGTERM)
    _training_process.wait(timeout=10)
    _training_process = None
    return {"status": "stopped"}


@router.get("/api/train/status")
async def api_train_status():
    """API: get training status."""
    if _training_process is None:
        return {"running": False}
    return {"running": _training_process.poll() is None, "pid": _training_process.pid}


@router.get("/api/configs")
async def api_configs():
    """API: list available configs."""
    configs = []
    if CONFIGS_DIR.exists():
        for f in CONFIGS_DIR.glob("*.yaml"):
            with open(f) as fh:
                configs.append({"name": f.stem, "config": yaml.safe_load(fh)})
    return configs
