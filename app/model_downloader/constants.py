"""Shared constants for the download manager.

Status values are persisted as TEXT in the ``downloads`` table; keep them
stable. The lifecycle is:

    queued -> active -> verifying -> completed
       |        |-> paused -> (resume) -> active
       |        |-> failed (network, retryable) -> queued (backoff)
       |-> cancelled
"""

from __future__ import annotations

# Auth schemes for HostCredential
AUTH_SCHEME_BEARER = "bearer"
AUTH_SCHEME_HEADER = "header"
AUTH_SCHEME_QUERY = "query"
AUTH_SCHEMES = (AUTH_SCHEME_BEARER, AUTH_SCHEME_HEADER, AUTH_SCHEME_QUERY)

# Hosts for which a bearer token can be sourced from the environment when no
# stored credential matches. Values are the env var names to try, in order.
# Only consulted during auto-resolve for an exact host match over https, so the
# same per-hop boundary rules apply (e.g. the token is dropped on a redirect to
# a CDN host). Kept here so the host->env-var mapping lives in one place.
ENV_TOKEN_HOSTS = {
    "huggingface.co": ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"),
}


class DownloadStatus:
    QUEUED = "queued"
    ACTIVE = "active"
    PAUSED = "paused"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    #: States from which a worker is doing (or about to do) network I/O.
    LIVE = (QUEUED, ACTIVE, VERIFYING)
    #: Terminal states — the job will not transition again on its own.
    TERMINAL = (COMPLETED, FAILED, CANCELLED)


# Default temp-file suffix. Distinctive so the startup orphan sweep only
# removes files THIS subsystem created, never unrelated *.tmp files.
TMP_SUFFIX = ".comfy-download.part"
