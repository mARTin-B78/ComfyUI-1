"""Manual, validated redirect-following request opener.

Automatic redirects are disabled. We follow hops ourselves
so that on *every* hop we (a) re-validate scheme + reject credentials-in-URL,
(b) recompute which stored credential — if any — applies to that hop's host,
and (c) let the connector's resolver screen the IP. This is the single place
that attaches credentials, so a token can never ride a redirect to a CDN host.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional
from urllib.parse import urljoin, urlsplit, urlunsplit

import aiohttp

from app.model_downloader.credentials.resolver import resolve_auth_for_hop
from app.model_downloader.net.session import get_session
from app.model_downloader.security.ssrf import (
    MAX_REDIRECTS,
    SSRFError,
    check_redirect_hop,
)

_REDIRECT_CODES = {301, 302, 303, 307, 308}
DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=120)


def redact_url(url: str) -> str:
    """Drop the query string so a query-scheme secret is never logged/stored."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<unparseable-url>"
    return urlunsplit(parts._replace(query=""))


async def _resolve_final_response(
    method: str,
    url: str,
    credential_id: Optional[str],
    base_headers: dict[str, str],
    timeout: aiohttp.ClientTimeout,
) -> tuple[aiohttp.ClientResponse, str]:
    """Follow redirects manually until a non-redirect response.

    Each intermediate redirect response is released before the next hop.
    Returns the final ``(response, final_url)``; the caller owns releasing it.
    """
    session = await get_session()
    current = url
    hops = 0
    while True:
        check_redirect_hop(current)
        parts = urlsplit(current)
        auth = await resolve_auth_for_hop(
            parts.hostname or "", parts.scheme, explicit_credential_id=credential_id
        )
        req_headers = dict(base_headers)
        req_url = current
        if auth is not None:
            req_headers.update(auth.headers)
            req_url = auth.apply_to_url(current)

        resp = await session.request(
            method,
            req_url,
            allow_redirects=False,
            headers=req_headers,
            timeout=timeout,
        )
        if resp.status in _REDIRECT_CODES and resp.headers.get("Location"):
            next_url = urljoin(str(resp.url), resp.headers["Location"])
            await resp.release()
            hops += 1
            if hops > MAX_REDIRECTS:
                raise SSRFError(
                    f"too many redirects (> {MAX_REDIRECTS}) for {redact_url(url)}"
                )
            current = next_url
            continue
        return resp, redact_url(str(resp.url))


@asynccontextmanager
async def open_validated(
    method: str,
    url: str,
    *,
    credential_id: Optional[str] = None,
    headers: Optional[dict[str, str]] = None,
    timeout: aiohttp.ClientTimeout = DEFAULT_TIMEOUT,
) -> AsyncIterator[tuple[aiohttp.ClientResponse, str]]:
    """Open ``method url`` following redirects manually and validated.

    Yields ``(response, final_url)`` where ``final_url`` is redacted of any
    query string. The response is released automatically on exit.
    """
    resp, final_url = await _resolve_final_response(
        method, url, credential_id, dict(headers or {}), timeout
    )
    try:
        yield resp, final_url
    finally:
        try:
            await resp.release()
        except Exception:  # pragma: no cover - best-effort cleanup
            logging.debug("[model_downloader] response release error", exc_info=True)
