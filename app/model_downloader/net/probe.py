"""Pre-download probe.

Issues a tiny ranged GET (``Range: bytes=0-0``) — which doubles as a
range-support test — to discover ``Content-Length``, ``Accept-Ranges``,
``ETag``/``Last-Modified``, and the final post-redirect URL. For HuggingFace
LFS files the true size also appears in the non-standard ``X-Linked-Size``
header, which we read as a fallback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse, urlsplit

import aiohttp

from app.model_downloader.net.http import (
    filename_from_content_disposition,
    open_validated,
    redact_url,
)
from app.model_downloader.net.session import parse_int_header

_PROBE_TIMEOUT = aiohttp.ClientTimeout(total=60, sock_connect=30, sock_read=30)


@dataclass
class ProbeResult:
    ok: bool
    status: int
    final_url: Optional[str] = None
    total_bytes: Optional[int] = None
    accept_ranges: bool = False
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    gated: bool = False  # 401/403 — needs (or has wrong) credentials
    error: Optional[str] = None
    # HuggingFace's ``X-Error-Code`` header (e.g. ``GatedRepo``,
    # ``RepoNotFound``) when the host reports one. Lets us tell "this repo is
    # gated — request access" apart from "you just need a token".
    error_code: Optional[str] = None
    # Filename the server intends this response to be saved as: the
    # ``Content-Disposition`` name if present, else the post-redirect URL's
    # basename. Used to resolve the real extension for URLs (e.g. Civitai's
    # ``/api/download`` endpoints) that carry no extension in their path.
    filename: Optional[str] = None

    @property
    def is_gated_repo(self) -> bool:
        """True when the host says the repo is gated (access must be granted).

        Distinct from a plain missing/invalid token: even a valid credential
        won't help until the user accepts the model's terms on its page.
        """
        return (self.error_code or "").lower() == "gatedrepo"


def _total_from_content_range(value: Optional[str]) -> Optional[int]:
    # "bytes 0-0/12345" -> 12345 ; "bytes 0-0/*" -> None
    if not value or "/" not in value:
        return None
    total = value.rsplit("/", 1)[1].strip()
    return parse_int_header(total)


def _filename_from_response(
    content_disposition: Optional[str], final_url: Optional[str]
) -> Optional[str]:
    name = filename_from_content_disposition(content_disposition)
    if name:
        return name
    if final_url:
        base = urlsplit(final_url).path.rsplit("/", 1)[-1]
        if base:
            return base
    return None


async def probe(url: str, *, credential_id: Optional[str] = None) -> ProbeResult:
    """Probe ``url`` and return discovered metadata, failing soft."""
    try:
        async with open_validated(
            "GET",
            url,
            credential_id=credential_id,
            headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"},
            timeout=_PROBE_TIMEOUT,
        ) as (resp, final_url):
            if resp.status in (401, 403):
                error_code = resp.headers.get("X-Error-Code")
                error_message = resp.headers.get("X-Error-Message")
                return ProbeResult(
                    ok=False, status=resp.status, final_url=final_url, gated=True,
                    error_code=error_code,
                    error=(
                        error_message
                        or f"host returned {resp.status} (authentication required)"
                    ),
                )
            if resp.status not in (200, 206):
                return ProbeResult(
                    ok=False, status=resp.status, final_url=final_url,
                    error=f"probe returned HTTP {resp.status}",
                )

            headers = resp.headers
            accept_ranges = False
            total: Optional[int] = None
            if resp.status == 206:
                accept_ranges = True
                total = _total_from_content_range(headers.get("Content-Range"))
            else:  # 200: server ignored the range
                accept_ranges = headers.get("Accept-Ranges", "").lower() == "bytes"
                total = parse_int_header(headers.get("Content-Length"))

            if total is None:
                total = parse_int_header(headers.get("X-Linked-Size"))

            return ProbeResult(
                ok=True,
                status=resp.status,
                final_url=final_url,
                total_bytes=total,
                accept_ranges=accept_ranges,
                etag=headers.get("ETag"),
                last_modified=headers.get("Last-Modified"),
                filename=_filename_from_response(
                    headers.get("Content-Disposition"), final_url
                ),
            )
    except Exception as e:  # network / SSRF / timeout
        host = urlparse(url).netloc or "<unknown>"
        logging.debug("[model_downloader] probe failed for %s: %s", host, type(e).__name__)
        return ProbeResult(ok=False, status=0, error="probe failed: network error")


def gated_error_message(url: str, pr: ProbeResult) -> str:
    """Build a user-facing message for a gated/auth-required probe result.

    Distinguishes a *gated* repo (access must be requested/granted on the model
    page — a token alone is not enough) from a plain missing/invalid credential.
    """
    redacted = redact_url(url)
    if pr.is_gated_repo:
        detail = (pr.error or "access is restricted").rstrip()
        if detail and not detail.endswith((".", "!", "?")):
            detail += "."
        return (
            f"{redacted} is a gated model — {detail} Request access on the model's "
            f"page, add an API key for this host at /api/download/credentials, and retry."
        )
    return (
        f"{redacted} requires authentication. Add an API key for this host at "
        f"/api/download/credentials and retry."
    )
