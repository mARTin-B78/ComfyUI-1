"""Request schemas for the download manager API.

Pydantic enforces shape at the boundary; handlers operate only on validated
values past that point.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from app.model_downloader.constants import AUTH_SCHEME_BEARER


class EnqueueRequest(BaseModel):
    url: str
    model_id: str
    priority: int = 0
    expected_sha256: Optional[str] = None
    allow_any_extension: bool = False
    credential_id: Optional[str] = None


class PriorityRequest(BaseModel):
    priority: int


class AvailabilityRequest(BaseModel):
    """``{model_id: url}`` — the URLs declared in the workflow JSON."""

    models: dict[str, str] = Field(default_factory=dict)


class CredentialUpsertRequest(BaseModel):
    host: str
    secret: str
    auth_scheme: str = AUTH_SCHEME_BEARER
    header_name: Optional[str] = None
    query_param: Optional[str] = None
    label: Optional[str] = None
    match_subdomains: bool = False
    enabled: bool = True


__all__ = [
    "EnqueueRequest",
    "PriorityRequest",
    "AvailabilityRequest",
    "CredentialUpsertRequest",
]
