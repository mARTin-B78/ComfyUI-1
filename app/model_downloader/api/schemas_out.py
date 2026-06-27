"""Response helpers for the download manager API.

The download/status read models are plain dicts produced by the manager. This
module only needs to mask credentials for output (the secret is never returned).
"""

from __future__ import annotations

from app.model_downloader.credentials.store import CredentialView


def credential_to_dict(view: CredentialView) -> dict:
    """API-safe credential representation — never includes the secret."""
    return {
        "id": view.id,
        "host": view.host,
        "auth_scheme": view.auth_scheme,
        "header_name": view.header_name,
        "query_param": view.query_param,
        "label": view.label,
        "match_subdomains": view.match_subdomains,
        "enabled": view.enabled,
        "secret_last4": view.secret_last4,
        "created_at": view.created_at,
        "updated_at": view.updated_at,
    }
