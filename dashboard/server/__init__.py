"""FastAPI dashboard for training monitoring and control.

Usage:
    uvicorn dashboard.server:app --port 8000
"""
from .app import app

__all__ = ["app"]
