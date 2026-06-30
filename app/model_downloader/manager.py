"""Public facade for the download manager.

This is the only object the server imports. It validates requests, owns the
:class:`Scheduler`, and exposes a small async API plus read models for status.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Callable, Optional

from app.model_downloader.constants import DownloadStatus
from app.model_downloader.database import queries
from app.model_downloader.scheduler import SCHEDULER
from app.model_downloader.security import paths
from app.model_downloader.security.allowlist import is_url_allowed
from app.model_downloader.security.paths import InvalidModelId

# Non-terminal statuses: an existing row in one of these blocks a re-enqueue.
_LIVE_STATUSES = (
    DownloadStatus.QUEUED,
    DownloadStatus.ACTIVE,
    DownloadStatus.PAUSED,
    DownloadStatus.VERIFYING,
)


class DownloadError(Exception):
    """A user-facing error with a stable machine-readable code."""

    def __init__(self, code: str, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = status


class DownloadManager:
    def __init__(self) -> None:
        self._scheduler = SCHEDULER
        self._notify_cb: Optional[Callable[[str], None]] = None

    def set_notify(self, cb: Optional[Callable[[str], None]]) -> None:
        self._notify_cb = cb
        self._scheduler.set_notify(cb)

    async def start(self) -> None:
        await self._scheduler.start()

    # ----- enqueue -----

    async def enqueue(
        self,
        url: str,
        model_id: str,
        *,
        priority: int = 0,
        expected_sha256: Optional[str] = None,
        allow_any_extension: bool = False,
        credential_id: Optional[str] = None,
    ) -> str:
        if not is_url_allowed(url, allow_any_extension):
            raise DownloadError(
                "URL_NOT_ALLOWED",
                "URL is not on the download allowlist (host/scheme/extension).",
            )
        try:
            paths.parse_model_id(model_id, allow_any_extension)
            dest_path, temp_path = paths.resolve_destination(model_id, allow_any_extension)
        except InvalidModelId as e:
            raise DownloadError("INVALID_MODEL_ID", str(e))

        if await asyncio.to_thread(
            paths.resolve_existing, model_id, allow_any_extension
        ):
            raise DownloadError(
                "ALREADY_AVAILABLE",
                f"Model already exists on disk: {model_id}",
                status=409,
            )
        if await self._has_live_download(model_id):
            raise DownloadError(
                "ALREADY_DOWNLOADING",
                f"A download for {model_id} is already in progress.",
                status=409,
            )

        download_id = str(uuid.uuid4())
        await asyncio.to_thread(
            queries.insert_download,
            {
                "id": download_id,
                "url": url,
                "model_id": model_id,
                "dest_path": dest_path,
                "temp_path": temp_path,
                "status": DownloadStatus.QUEUED,
                "priority": priority,
                "expected_sha256": expected_sha256,
                "credential_id": credential_id,
                "allow_any_extension": allow_any_extension,
            },
        )
        logging.info("[model_downloader] enqueued %s -> %s", url, model_id)
        await self._scheduler.pump()
        return download_id

    async def _has_live_download(self, model_id: str) -> bool:
        rows = await asyncio.to_thread(queries.list_downloads)
        return any(
            r.model_id == model_id and r.status in _LIVE_STATUSES for r in rows
        )

    # ----- control -----

    async def pause(self, download_id: str) -> None:
        job = self._scheduler.get_job(download_id)
        if job is not None:
            job.request_pause()
            return
        row = await asyncio.to_thread(queries.get_download, download_id)
        if row is None:
            raise DownloadError("NOT_FOUND", "No such download.", status=404)
        if row.status == DownloadStatus.QUEUED:
            await asyncio.to_thread(
                queries.update_download, download_id, status=DownloadStatus.PAUSED
            )

    async def resume(self, download_id: str) -> None:
        row = await asyncio.to_thread(queries.get_download, download_id)
        if row is None:
            raise DownloadError("NOT_FOUND", "No such download.", status=404)
        if row.status in (DownloadStatus.PAUSED, DownloadStatus.FAILED):
            await asyncio.to_thread(
                queries.update_download,
                download_id,
                status=DownloadStatus.QUEUED,
                error=None,
            )
            await self._scheduler.pump()

    async def cancel(self, download_id: str) -> None:
        job = self._scheduler.get_job(download_id)
        if job is not None:
            job.request_cancel()
            return
        row = await asyncio.to_thread(queries.get_download, download_id)
        if row is None:
            raise DownloadError("NOT_FOUND", "No such download.", status=404)
        if row.status in _LIVE_STATUSES:
            import os

            try:
                os.remove(row.temp_path)
            except OSError:
                pass
            await asyncio.to_thread(
                queries.update_download, download_id, status=DownloadStatus.CANCELLED
            )

    async def set_priority(self, download_id: str, priority: int) -> None:
        row = await asyncio.to_thread(queries.get_download, download_id)
        if row is None:
            raise DownloadError("NOT_FOUND", "No such download.", status=404)
        await asyncio.to_thread(
            queries.update_download, download_id, priority=priority
        )
        # Admission-order only (PRD section 13 default); a higher priority is
        # picked up the next time a slot frees. Pump in case a slot is free now.
        await self._scheduler.pump()

    # ----- read models -----

    def _view(self, row) -> dict:
        """Combine the persisted row with live in-memory progress, if running."""
        job = self._scheduler.get_job(row.id)
        bytes_done = row.bytes_done
        total = row.total_bytes
        speed = None
        eta = None
        segments = None
        if job is not None:
            st = job.state
            bytes_done = st.bytes_done
            total = st.total_bytes if st.total_bytes is not None else total
            speed = st.speed_bps
            eta = st.eta_seconds
            segments = [
                {"idx": s.idx, "bytes_done": s.bytes_done, "length": s.length}
                for s in st.segments
                if s.end >= s.start
            ]
        progress = (bytes_done / total) if total else None
        return {
            "download_id": row.id,
            "model_id": row.model_id,
            "url": row.url,
            "status": row.status,
            "priority": row.priority,
            "total_bytes": total,
            "bytes_done": bytes_done,
            "progress": progress,
            "speed_bps": speed,
            "eta_seconds": eta,
            "segments": segments,
            "error": row.error,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }

    def _view_from_state(self, job) -> dict:
        """Build a view purely from the live in-memory job state (no DB)."""
        st = job.state
        return {
            "download_id": st.download_id,
            "model_id": st.model_id,
            "url": st.url,
            "status": st.status,
            "priority": st.priority,
            "total_bytes": st.total_bytes,
            "bytes_done": st.bytes_done,
            "progress": st.progress,
            "speed_bps": st.speed_bps,
            "eta_seconds": st.eta_seconds,
            "segments": [
                {"idx": s.idx, "bytes_done": s.bytes_done, "length": s.length}
                for s in st.segments
                if s.end >= s.start
            ],
            "error": st.error,
        }

    def status_sync(self, download_id: str) -> Optional[dict]:
        """Synchronous status read for the websocket notify path.

        Uses live in-memory state when the job is running (no DB round-trip on
        the hot path); falls back to a quick DB read otherwise.
        """
        job = self._scheduler.get_job(download_id)
        if job is not None:
            return self._view_from_state(job)
        row = queries.get_download(download_id)
        return self._view(row) if row is not None else None

    async def status(self, download_id: str) -> Optional[dict]:
        row = await asyncio.to_thread(queries.get_download, download_id)
        return self._view(row) if row is not None else None

    async def list(self) -> list[dict]:
        rows = await asyncio.to_thread(queries.list_downloads)
        return [self._view(r) for r in rows]

    async def availability(self, models: dict[str, str]) -> dict[str, dict]:
        """Bulk per-id ``{state, progress, ...}`` for the frontend poll.

        ``state`` is ``available`` (on disk), ``downloading`` (live row), or
        ``missing``. Cheap: a path lookup plus an in-memory/DB status check.
        """
        rows = await asyncio.to_thread(queries.list_downloads)
        by_model: dict[str, object] = {}
        for r in rows:
            if r.status in _LIVE_STATUSES or r.model_id not in by_model:
                by_model[r.model_id] = r

        out: dict[str, dict] = {}
        for model_id, url in models.items():
            try:
                exists = await asyncio.to_thread(paths.resolve_existing, model_id)
            except InvalidModelId:
                out[model_id] = {"state": "missing", "url_allowed": is_url_allowed(url)}
                continue
            if exists:
                out[model_id] = {"state": "available", "url_allowed": is_url_allowed(url)}
                continue
            row = by_model.get(model_id)
            if row is not None and row.status in _LIVE_STATUSES:
                view = self._view(row)
                out[model_id] = {
                    "state": "downloading",
                    "url_allowed": is_url_allowed(url),
                    "download_id": view["download_id"],
                    "progress": view["progress"],
                    "bytes_done": view["bytes_done"],
                    "total_bytes": view["total_bytes"],
                    "speed_bps": view["speed_bps"],
                }
            else:
                out[model_id] = {"state": "missing", "url_allowed": is_url_allowed(url)}
        return out


DOWNLOAD_MANAGER = DownloadManager()
