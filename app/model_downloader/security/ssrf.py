"""SSRF / exfiltration defenses.

Two cooperating layers:

1. :class:`ValidatingResolver` is installed on the shared connector. Every
   connection — the initial probe and every segment GET, including ones made
   after a redirect — resolves its host through this resolver, which rejects
   any address that lands on a private / special-use IP range. Because the
   resolve and the connect happen together inside the connector, there is no
   check-then-connect window for DNS rebinding to exploit.

2. :func:`check_redirect_hop` re-validates every redirect hop. The host
   allowlist gates only the *initial* user-supplied URL (anti-SSRF for
   arbitrary input); legitimate downloads from allowlisted origins redirect
   to presigned CDN hosts that are deliberately NOT on the allowlist (HF ->
   ``cdn-lfs*.huggingface.co``, Civitai -> signed Cloudflare/S3), so hops are
   instead screened for scheme, embedded credentials, and — via the resolver
   above — private IPs. Credentials are only ever attached when a hop's host
   exactly matches a stored credential, so they are dropped on the CDN hop.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

from aiohttp.abc import AbstractResolver
from aiohttp.resolver import DefaultResolver

from app.model_downloader.security.allowlist import LOOPBACK_HOSTS

# Cap the redirect chain length a hop may use.
MAX_REDIRECTS = 5


class SSRFError(Exception):
    """A hop failed an SSRF / allowlist check."""


def is_scheme_allowed(scheme: str | None, host: str | None) -> bool:
    """True iff ``scheme`` is permitted for ``host`` on a download hop.

    https is always allowed; plain http only for loopback/approved dev hosts.
    """
    if not scheme:
        return False
    scheme = scheme.lower()
    if scheme == "https":
        return True
    if scheme == "http":
        return bool(host) and host.lower() in LOOPBACK_HOSTS
    return False


def is_blocked_ip(ip_str: str) -> bool:
    """True for any address we refuse to connect to.

    Covers loopback, link-local (incl. 169.254.169.254 cloud metadata),
    RFC1918 private ranges, unique-local (ULA), unspecified (0.0.0.0/::),
    multicast and other reserved ranges.
    """
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True  # unparseable -> refuse
    # On CPython before the gh-113171 fix (backported to 3.12.4/3.11.9/
    # 3.10.14/3.9.19) the is_* properties don't see through IPv4-mapped IPv6
    # (e.g. ::ffff:169.254.169.254), so resolve and re-check the embedded IPv4
    # to keep mapped metadata/private addresses from slipping past the filter.
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


class ValidatingResolver(AbstractResolver):
    """Delegating resolver that drops blocked IPs from every resolution.

    If a hostname resolves only to blocked addresses, the connection fails
    closed with an :class:`OSError`, which aiohttp surfaces as a connection
    error to the caller.
    """

    def __init__(self) -> None:
        self._inner = DefaultResolver()

    async def resolve(self, host, port=0, family=socket.AF_INET):
        infos = await self._inner.resolve(host, port, family)
        # localhost/127.0.0.1 are an explicit, opt-in allowlist feature.
        if isinstance(host, str) and host.lower() in LOOPBACK_HOSTS:
            return infos
        safe = [info for info in infos if not is_blocked_ip(info["host"])]
        if not safe:
            raise OSError(
                f"refusing to connect to {host!r}: resolves only to "
                f"private/special-use addresses"
            )
        return safe

    async def close(self) -> None:
        await self._inner.close()


def check_redirect_hop(url: str) -> str:
    """Validate one redirect hop's URL.

    Returns the URL unchanged on success; raises :class:`SSRFError` otherwise.
    Requires https for external hosts (http only for loopback/approved dev
    hosts) and forbids credentials-in-URL. The host is NOT re-checked against
    the allowlist (CDN redirect targets are off-list by design); private-IP
    protection is provided by the connector's resolver, and credential leakage
    is prevented by exact host matching at attach time. The landing filename's
    extension is gated separately by the caller.
    """
    try:
        parsed = urlparse(url)
    except ValueError as e:
        raise SSRFError(f"unparseable redirect URL {url!r}: {e}") from e
    if not parsed.hostname:
        raise SSRFError(f"redirect URL has no host: {url!r}")
    if not is_scheme_allowed(parsed.scheme, parsed.hostname):
        raise SSRFError(
            f"redirect to disallowed scheme {parsed.scheme!r} for host "
            f"{parsed.hostname!r} (https required for external hosts)"
        )
    if parsed.username or parsed.password:
        raise SSRFError("credentials-in-URL are not allowed")
    return url
