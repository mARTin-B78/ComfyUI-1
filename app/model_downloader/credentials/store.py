"""The credential store: one API key per host.

Secrets are write-only over the API — :class:`CredentialView` carries only
masked metadata (``secret_last4`` + scheme + label), never the secret itself.
At-rest protection for v1 is filesystem permissions on the shared DB (the DB
is the trust boundary); encryption-at-rest is a noted future seam.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlsplit

from app.model_downloader.constants import (
    AUTH_SCHEME_BEARER,
    AUTH_SCHEME_HEADER,
    AUTH_SCHEME_QUERY,
    AUTH_SCHEMES,
)
from app.model_downloader.database import queries
from app.model_downloader.database.models import HostCredential


def normalize_host(host: str) -> str:
    """Lowercase, strip port, IDNA-encode."""
    if not host:
        return ""
    host = host.strip()
    if "://" in host:  # a full URL was pasted — extract just the host
        host = urlsplit(host).hostname or ""
    host = host.lower()
    if host.startswith("[") and "]" in host:  # bracketed IPv6 literal
        host = host[1 : host.index("]")]
    elif host.count(":") == 1:  # host:port (not IPv6)
        host = host.split(":", 1)[0]
    try:
        host = host.encode("idna").decode("ascii")
    except (UnicodeError, ValueError):
        pass
    return host


@dataclass(frozen=True)
class CredentialView:
    """Masked, API-safe view of a credential — never includes the secret."""

    id: str
    host: str
    auth_scheme: str
    header_name: Optional[str]
    query_param: Optional[str]
    label: Optional[str]
    match_subdomains: bool
    enabled: bool
    secret_last4: Optional[str]
    created_at: int
    updated_at: int


def _to_view(row: HostCredential) -> CredentialView:
    return CredentialView(
        id=row.id,
        host=row.host,
        auth_scheme=row.auth_scheme,
        header_name=row.header_name,
        query_param=row.query_param,
        label=row.label,
        match_subdomains=row.match_subdomains,
        enabled=row.enabled,
        secret_last4=row.secret_last4,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class CredentialValidationError(ValueError):
    """A credential upsert had inconsistent fields."""


class CredentialStore:
    """Async facade over the ``host_credentials`` table.

    DB access is synchronous (SQLite) and offloaded via ``asyncio.to_thread``.
    """

    async def upsert(
        self,
        host: str,
        secret: str,
        *,
        auth_scheme: str = AUTH_SCHEME_BEARER,
        header_name: Optional[str] = None,
        query_param: Optional[str] = None,
        label: Optional[str] = None,
        match_subdomains: bool = False,
        enabled: bool = True,
    ) -> CredentialView:
        host = normalize_host(host)
        if not host:
            raise CredentialValidationError("host is required")
        if not secret:
            raise CredentialValidationError("secret is required")
        if auth_scheme not in AUTH_SCHEMES:
            raise CredentialValidationError(
                f"auth_scheme must be one of {AUTH_SCHEMES}, got {auth_scheme!r}"
            )
        if auth_scheme == AUTH_SCHEME_HEADER and not header_name:
            header_name = "Authorization"
        if auth_scheme == AUTH_SCHEME_QUERY and not query_param:
            raise CredentialValidationError(
                "query_param is required when auth_scheme='query'"
            )
        values = {
            "host": host,
            "secret": secret,
            "secret_last4": secret[-4:] if len(secret) > 4 else None,
            "auth_scheme": auth_scheme,
            "header_name": header_name,
            "query_param": query_param,
            "label": label,
            "match_subdomains": match_subdomains,
            "enabled": enabled,
        }
        row = await asyncio.to_thread(queries.upsert_credential, values)
        return _to_view(row)

    async def list(self) -> list[CredentialView]:
        rows = await asyncio.to_thread(queries.list_credentials)
        return [_to_view(r) for r in rows]

    async def get(self, credential_id: str) -> Optional[CredentialView]:
        row = await asyncio.to_thread(queries.get_credential, credential_id)
        return _to_view(row) if row is not None else None

    async def delete(self, credential_id: str) -> bool:
        return await asyncio.to_thread(queries.delete_credential, credential_id)


CREDENTIAL_STORE = CredentialStore()
