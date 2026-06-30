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
from urllib.parse import urlparse

import aiohttp

from app.model_downloader.net.http import open_validated
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


def _total_from_content_range(value: Optional[str]) -> Optional[int]:
    # "bytes 0-0/12345" -> 12345 ; "bytes 0-0/*" -> None
    if not value or "/" not in value:
        return None
    total = value.rsplit("/", 1)[1].strip()
    return parse_int_header(total)


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
                return ProbeResult(
                    ok=False, status=resp.status, final_url=final_url, gated=True,
                    error=f"host returned {resp.status} (authentication required)",
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
            )
    except Exception as e:  # network / SSRF / timeout
        host = urlparse(url).netloc or "<unknown>"
        logging.debug("[model_downloader] probe failed for %s: %s", host, type(e).__name__)
        return ProbeResult(ok=False, status=0, error="probe failed: network error")
