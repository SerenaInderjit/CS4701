"""Play (live game view) control."""
import os
import signal
import subprocess
import sys
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from .common import FRAME_DIR, ROOT

router = APIRouter()

_play_process: Optional[subprocess.Popen] = None
_play_results: str = ""


@router.post("/api/play/start")
async def api_play_start(config: Dict[str, Any]):
    """API: start playing with a checkpoint."""
    global _play_process, _play_results
    if _play_process is not None and _play_process.poll() is None:
        raise HTTPException(status_code=400, detail="Already playing")

    checkpoint = config.get("checkpoint")
    if not checkpoint:
        raise HTTPException(status_code=400, detail="checkpoint is required")
    episodes = config.get("episodes", 3)

    FRAME_DIR.mkdir(exist_ok=True)
    for f in FRAME_DIR.glob("*.png"):
        f.unlink()

    _play_results = ""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    _play_process = subprocess.Popen(
        [sys.executable, "scripts/play_mario.py", "--policy", "ppo",
         "--checkpoint", checkpoint, "--episodes", str(episodes),
         "--device", "cpu", "--save-frames", str(FRAME_DIR)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {"status": "started", "pid": _play_process.pid}


@router.post("/api/play/stop")
async def api_play_stop():
    """API: stop playing."""
    global _play_process
    if _play_process is None or _play_process.poll() is not None:
        raise HTTPException(status_code=400, detail="Not playing")
    _play_process.send_signal(signal.SIGTERM)
    _play_process.wait(timeout=10)
    _play_process = None
    return {"status": "stopped"}


@router.get("/api/play/status")
async def api_play_status():
    """API: get play status."""
    if _play_process is None:
        return {"running": False, "results": _play_results}
    running = _play_process.poll() is None
    if not running and _play_process.stdout is not None:
        _play_results = _play_process.stdout.read().decode()
    return {"running": running, "results": _play_results}


@router.get("/api/play/frame")
async def api_play_frame():
    """API: get the latest game frame."""
    if not FRAME_DIR.exists():
        raise HTTPException(status_code=404, detail="No frames yet")
    frames = sorted(FRAME_DIR.glob("*.png"))
    if not frames:
        raise HTTPException(status_code=404, detail="No frames yet")
    return FileResponse(frames[-1])
