"""Turn a stored credential into a per-hop request modifier (PRD section 9.4.2).

The critical rule: a credential is only ever attached when *the current hop's
host* matches a stored credential, and only over https. This is recomputed
from scratch on every redirect hop, so a token bound to ``huggingface.co`` is
silently dropped when the request is redirected to a presigned CDN host —
which is exactly what these hubs expect.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlencode, urlsplit, urlunsplit

from app.model_downloader.constants import (
    AUTH_SCHEME_BEARER,
    AUTH_SCHEME_HEADER,
    AUTH_SCHEME_QUERY,
    ENV_TOKEN_HOSTS,
)
from app.model_downloader.credentials.store import normalize_host
from app.model_downloader.database import queries
from app.model_downloader.database.models import HostCredential


@dataclass
class RequestAuth:
    """How to modify a single request to carry a credential."""

    headers: dict[str, str] = field(default_factory=dict)
    query: dict[str, str] = field(default_factory=dict)

    def apply_to_url(self, url: str) -> str:
        if not self.query:
            return url
        parts = urlsplit(url)
        # Append only the credential params, leaving the original query string
        # (including any repeated keys and existing encoding) untouched.
        creds = urlencode(self.query)
        query = f"{parts.query}&{creds}" if parts.query else creds
        return urlunsplit(parts._replace(query=query))


def _matches(cred: HostCredential, hop_host: str) -> bool:
    cred_host = cred.host
    if hop_host == cred_host:
        return True
    if cred.match_subdomains:
        # Label-boundary suffix: api.example.com matches example.com, but
        # evil-example.com does NOT.
        return hop_host.endswith("." + cred_host)
    return False


def _build_auth(cred: HostCredential) -> RequestAuth:
    if cred.auth_scheme == AUTH_SCHEME_BEARER:
        return RequestAuth(headers={"Authorization": f"Bearer {cred.secret}"})
    if cred.auth_scheme == AUTH_SCHEME_HEADER:
        name = cred.header_name or "Authorization"
        return RequestAuth(headers={name: cred.secret})
    if cred.auth_scheme == AUTH_SCHEME_QUERY and cred.query_param:
        return RequestAuth(query={cred.query_param: cred.secret})
    return RequestAuth()


def _resolve_sync(
    host: str, scheme: str, explicit_credential_id: Optional[str]
) -> Optional[RequestAuth]:
    # Never attach a secret over a non-https hop (PRD section 9.4.2).
    if scheme.lower() != "https":
        return None
    hop_host = normalize_host(host)
    if not hop_host:
        return None

    if explicit_credential_id is not None:
        cred = queries.get_credential(explicit_credential_id)
        # An explicit credential is still subject to the per-hop host check —
        # it is not forced onto a non-matching host.
        if cred is None or not cred.enabled or not _matches(cred, hop_host):
            return None
        return _build_auth(cred)

    # Auto-resolve: exact host first, then any subdomain-matching credential.
    cred = queries.get_credential_by_host(hop_host)
    if cred is not None and cred.enabled:
        return _build_auth(cred)
    for sub in queries.list_subdomain_credentials():
        if sub.enabled and _matches(sub, hop_host):
            return _build_auth(sub)

    # Env fallback: only for an exact host match, and only after the DB lookups
    # miss, so a user-set credential always takes precedence. The token is never
    # persisted; it is read fresh from the environment on each hop.
    for var in ENV_TOKEN_HOSTS.get(hop_host, ()):
        token = os.environ.get(var)
        if token:
            return RequestAuth(headers={"Authorization": f"Bearer {token}"})
    return None


async def resolve_auth_for_hop(
    host: str, scheme: str, *, explicit_credential_id: Optional[str] = None
) -> Optional[RequestAuth]:
    """Resolve the credential (if any) to attach for one request hop."""
    return await asyncio.to_thread(
        _resolve_sync, host, scheme, explicit_credential_id
    )
