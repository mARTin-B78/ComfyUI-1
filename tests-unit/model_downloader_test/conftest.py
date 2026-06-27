"""Shared fixtures for the model download manager tests.

These run in-process (no ComfyUI subprocess): a file-backed SQLite DB is
initialized once, a temp model folder is registered with ``folder_paths``, and
the shared aiohttp session is reset between tests so each async test gets a
session bound to its own event loop.
"""

from __future__ import annotations

import os
import tempfile

import pytest


@pytest.fixture(scope="session", autouse=True)
def _init_db():
    import app.database.db as db
    from comfy.cli_args import args

    db_path = tempfile.mktemp(suffix="-dlmgr-test.sqlite3")
    args.database_url = f"sqlite:///{db_path}"
    db.init_db()
    yield
    try:
        os.remove(db_path)
    except OSError:
        pass


@pytest.fixture(autouse=True)
def _reset_runtime():
    """Reset module singletons that hold event-loop-bound or cross-test state."""
    import app.model_downloader.net.session as ns
    from app.model_downloader.scheduler import SCHEDULER

    ns._session = None
    SCHEDULER._jobs.clear()
    SCHEDULER._tasks.clear()
    SCHEDULER._backoff_until.clear()
    SCHEDULER._started = False
    yield
    ns._session = None


@pytest.fixture
def model_root(tmp_path):
    """Register a temp 'loras' model folder and return its absolute path."""
    import folder_paths

    root = tmp_path / "loras"
    root.mkdir(parents=True, exist_ok=True)
    saved = folder_paths.folder_names_and_paths.get("loras")
    folder_paths.folder_names_and_paths["loras"] = (
        [str(root)],
        {".safetensors", ".sft", ".ckpt", ".pt", ".pth"},
    )
    yield str(root)
    if saved is not None:
        folder_paths.folder_names_and_paths["loras"] = saved
    else:
        folder_paths.folder_names_and_paths.pop("loras", None)
