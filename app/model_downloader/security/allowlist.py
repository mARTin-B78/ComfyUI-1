"""URL allowlist for server-side model fetches.

Default-deny. A URL is downloadable only when its parsed host + scheme are
allowlisted AND (unless explicitly relaxed) its final filename ends in a
known model extension.

The built-in host defaults mirror the frontend's ``isModelDownloadable``
allowlist so the two flows agree on what is eligible; ``--download-allowed-hosts``
extends it for self-hosted mirrors. Matching is done on ``urlparse().hostname``
(never a raw string prefix) so userinfo tricks like
``http://127.0.0.1@169.254.169.254/x.safetensors`` — whose real host is the
metadata IP — cannot slip past.
"""

from __future__ import annotations

from urllib.parse import urlparse

from comfy.cli_args import args

# host -> set of allowed schemes. Frontend parity (HuggingFace / Civitai /
# localhost). Extra hosts from --download-allowed-hosts are https-only.
_DEFAULT_ALLOWED_HOSTS: dict[str, set[str]] = {
    "huggingface.co": {"https"},
    "civitai.com": {"https"},
    "localhost": {"http", "https"},
    "127.0.0.1": {"http", "https"},
}

# Hosts for which loopback addresses are intentionally permitted (the localhost
# "download a local model" feature). Every other host's loopback resolution is
# rejected by the SSRF resolver.
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

# Known model file extensions (frontend parity). Checked on the final filename.
ALLOWED_MODEL_EXTENSIONS = (
    ".safetensors",
    ".sft",
    ".ckpt",
    ".pth",
    ".pt",
    ".gguf",
    ".bin",
)


def _allowed_hosts() -> dict[str, set[str]]:
    hosts = {h: set(s) for h, s in _DEFAULT_ALLOWED_HOSTS.items()}
    for extra in getattr(args, "download_allowed_hosts", []) or []:
        host = extra.strip().lower()
        if host:
            hosts.setdefault(host, set()).add("https")
    return hosts


def is_host_allowed(host: str | None, scheme: str | None) -> bool:
    """True iff ``host`` is allowlisted for ``scheme``.

    Used both for the initial URL and re-checked on every redirect hop,
    so a whitelisted URL cannot 30x into an off-list host.
    """
    if not host or not scheme:
        return False
    allowed = _allowed_hosts().get(host.lower())
    return allowed is not None and scheme.lower() in allowed


def has_allowed_extension(path: str, allow_any_extension: bool = False) -> bool:
    if allow_any_extension:
        return True
    return path.lower().endswith(ALLOWED_MODEL_EXTENSIONS)


def is_url_allowed(url: str, allow_any_extension: bool = False) -> bool:
    """Check whether ``url`` is permitted as a server-side download source."""
    if not isinstance(url, str) or not url:
        return False
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if not is_host_allowed(parsed.hostname, parsed.scheme):
        return False
    return has_allowed_extension(parsed.path, allow_any_extension)
