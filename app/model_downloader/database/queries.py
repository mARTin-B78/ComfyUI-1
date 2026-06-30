"""Synchronous DB access for the download manager.

All functions open their own short-lived session via ``create_session`` and
commit before returning, mirroring ``app/assets`` usage. They are blocking
(SQLite) and should be called from async code through ``asyncio.to_thread``.
"""

from __future__ import annotations

import time
from typing import Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.database.db import create_session
from app.model_downloader.constants import DownloadStatus
from app.model_downloader.database.models import (
    Download,
    DownloadSegment,
    HostCredential,
)


# ----- downloads -----


def insert_download(values: dict) -> None:
    with create_session() as session:
        session.add(Download(**values))
        session.commit()


def get_download(download_id: str) -> Optional[Download]:
    with create_session() as session:
        row = session.get(Download, download_id)
        if row is not None:
            session.expunge_all()
        return row


def list_downloads() -> list[Download]:
    with create_session() as session:
        rows = list(
            session.execute(
                select(Download).order_by(Download.created_at.desc())
            ).scalars()
        )
        session.expunge_all()
        return rows


def list_segments(download_id: str) -> list[DownloadSegment]:
    with create_session() as session:
        rows = list(
            session.execute(
                select(DownloadSegment)
                .where(DownloadSegment.download_id == download_id)
                .order_by(DownloadSegment.idx)
            ).scalars()
        )
        session.expunge_all()
        return rows


def update_download(download_id: str, **fields) -> None:
    if not fields:
        return
    fields.setdefault("updated_at", int(time.time()))
    with create_session() as session:
        row = session.get(Download, download_id)
        if row is None:
            return
        for key, value in fields.items():
            setattr(row, key, value)
        session.commit()


def delete_download(download_id: str) -> None:
    with create_session() as session:
        row = session.get(Download, download_id)
        if row is not None:
            session.delete(row)
            session.commit()


def replace_segments(download_id: str, segments: list[dict]) -> None:
    """Atomically replace the segment plan for a download."""
    with create_session() as session:
        session.query(DownloadSegment).filter(
            DownloadSegment.download_id == download_id
        ).delete()
        for seg in segments:
            session.add(DownloadSegment(download_id=download_id, **seg))
        session.commit()


def update_segment_progress(download_id: str, idx: int, bytes_done: int) -> None:
    with create_session() as session:
        row = session.get(DownloadSegment, {"download_id": download_id, "idx": idx})
        if row is None:
            return
        row.bytes_done = bytes_done
        session.commit()


def list_queued_downloads() -> list[Download]:
    """Queued rows ordered for admission (priority desc, then FIFO)."""
    with create_session() as session:
        rows = list(
            session.execute(
                select(Download)
                .where(Download.status == DownloadStatus.QUEUED)
                .order_by(Download.priority.desc(), Download.created_at.asc())
            ).scalars()
        )
        session.expunge_all()
        return rows


def reconcile_live_downloads() -> list[Download]:
    """Reset any ``active``/``verifying`` rows left by a previous run.

    On a clean restart there can be no live worker, so anything still marked
    live is stale. Move it back to ``queued`` (offsets are preserved on the
    segment rows) so the scheduler re-admits it. Returns the rows that should
    be re-queued by the scheduler (queued + paused).
    """
    with create_session() as session:
        stale = list(
            session.execute(
                select(Download).where(
                    Download.status.in_([DownloadStatus.ACTIVE, DownloadStatus.VERIFYING])
                )
            ).scalars()
        )
        now = int(time.time())
        for row in stale:
            row.status = DownloadStatus.QUEUED
            row.updated_at = now
        session.commit()

        resumable = list(
            session.execute(
                select(Download)
                .where(Download.status == DownloadStatus.QUEUED)
                .order_by(Download.priority.desc(), Download.created_at.asc())
            ).scalars()
        )
        session.expunge_all()
        return resumable


# ----- host credentials -----


def get_credential(credential_id: str) -> Optional[HostCredential]:
    with create_session() as session:
        row = session.get(HostCredential, credential_id)
        if row is not None:
            session.expunge_all()
        return row


def get_credential_by_host(host: str) -> Optional[HostCredential]:
    with create_session() as session:
        row = (
            session.execute(
                select(HostCredential).where(HostCredential.host == host).limit(1)
            )
            .scalars()
            .first()
        )
        if row is not None:
            session.expunge_all()
        return row


def list_credentials() -> list[HostCredential]:
    with create_session() as session:
        rows = list(
            session.execute(
                select(HostCredential).order_by(HostCredential.host)
            ).scalars()
        )
        session.expunge_all()
        return rows


def list_subdomain_credentials() -> list[HostCredential]:
    """Credentials that opted into subdomain matching, for suffix checks."""
    with create_session() as session:
        rows = list(
            session.execute(
                select(HostCredential).where(HostCredential.match_subdomains.is_(True))
            ).scalars()
        )
        session.expunge_all()
        return rows


def upsert_credential(values: dict) -> HostCredential:
    """Insert or update a credential keyed by ``host``.

    Callers can target the same host concurrently (each runs in its own
    short-lived session on a separate connection), so the read-then-write here
    can race: two callers both see no existing row and both attempt an insert.
    The ``host`` column is uniquely indexed, so the loser's insert raises
    ``IntegrityError``. We recover by rolling back and retrying, at which point
    the now-committed row is found and updated in place, letting concurrent
    calls converge instead of failing or creating duplicates.
    """
    host = values["host"]
    now = int(time.time())
    last_error: IntegrityError | None = None
    for _ in range(2):
        with create_session() as session:
            row = (
                session.execute(
                    select(HostCredential).where(HostCredential.host == host).limit(1)
                )
                .scalars()
                .first()
            )
            if row is None:
                row = HostCredential(**values)
                row.created_at = now
                row.updated_at = now
                session.add(row)
            else:
                for key, value in values.items():
                    setattr(row, key, value)
                row.updated_at = now
            try:
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                last_error = exc
                continue
            session.refresh(row)
            session.expunge(row)
            return row
    assert last_error is not None
    raise last_error


def delete_credential(credential_id: str) -> bool:
    with create_session() as session:
        row = session.get(HostCredential, credential_id)
        if row is None:
            return False
        session.delete(row)
        session.commit()
        return True
