"""Positioned, off-loop file writes (PRD section 4 + 5.2).

Network I/O stays on the event loop; every blocking disk op (preallocate,
positioned write, fsync) is run in a bounded thread pool via
``run_in_executor`` so downloads never stall inference or the web server.

A single file descriptor is opened for the whole download. Segments write to
their own offsets with ``os.pwrite`` — which is offset-addressed and atomic
per call, so concurrent segment writers need no extra locking. Per-chunk
fsync is avoided; we fsync once at completion.
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

# One shared, bounded pool for all download disk I/O.
_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="dl-writer")


class FileWriter:
    """Owns the ``.part`` file descriptor for one download."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._fd: Optional[int] = None

    def _open(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)

    async def open(self) -> None:
        await asyncio.get_running_loop().run_in_executor(_EXECUTOR, self._open)

    async def preallocate(self, size: int) -> None:
        """Grow the file to ``size`` so segments write to their offsets."""
        if self._fd is None or size <= 0:
            return
        await asyncio.get_running_loop().run_in_executor(
            _EXECUTOR, os.ftruncate, self._fd, size
        )

    async def truncate(self, size: int = 0) -> None:
        """Truncate the file to ``size`` bytes (default: empty it)."""
        if self._fd is None:
            return
        await asyncio.get_running_loop().run_in_executor(
            _EXECUTOR, os.ftruncate, self._fd, size
        )

    async def write_at(self, offset: int, data: bytes) -> None:
        assert self._fd is not None, "writer not opened"
        await asyncio.get_running_loop().run_in_executor(
            _EXECUTOR, os.pwrite, self._fd, data, offset
        )

    async def flush(self) -> None:
        if self._fd is None:
            return
        await asyncio.get_running_loop().run_in_executor(_EXECUTOR, os.fsync, self._fd)

    async def close(self) -> None:
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        await asyncio.get_running_loop().run_in_executor(_EXECUTOR, os.close, fd)
