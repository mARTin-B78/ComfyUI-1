"""Unit tests for the credential store and the per-hop credential resolver.

Covers the critical rule: a secret is only ever attached when the current
hop's host matches a stored credential, and never over a non-https hop.
"""

from __future__ import annotations

import asyncio

import pytest

from app.model_downloader.credentials import resolver
from app.model_downloader.credentials.store import (
    CREDENTIAL_STORE,
    CredentialValidationError,
    normalize_host,
)
from app.model_downloader.database.models import HostCredential


# ----- pure host normalization + matching -----


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Civitai.com", "civitai.com"),
        ("HuggingFace.co:443", "huggingface.co"),
        ("  Example.COM  ", "example.com"),
    ],
)
def test_normalize_host(raw, expected):
    assert normalize_host(raw) == expected


def _cred(**kw) -> HostCredential:
    base = dict(
        id="x", host="civitai.com", match_subdomains=False, auth_scheme="bearer",
        secret="SECRET", enabled=True,
    )
    base.update(kw)
    return HostCredential(**base)


def test_matches_exact_only_by_default():
    c = _cred(host="civitai.com")
    assert resolver._matches(c, "civitai.com") is True
    assert resolver._matches(c, "api.civitai.com") is False
    assert resolver._matches(c, "evil-civitai.com") is False


def test_matches_subdomain_label_boundary():
    c = _cred(host="example.com", match_subdomains=True)
    assert resolver._matches(c, "api.example.com") is True
    assert resolver._matches(c, "example.com") is True
    # not a label boundary -> no match
    assert resolver._matches(c, "evil-example.com") is False


def test_build_auth_shapes():
    assert resolver._build_auth(_cred(auth_scheme="bearer")).headers == {
        "Authorization": "Bearer SECRET"
    }
    assert resolver._build_auth(
        _cred(auth_scheme="header", header_name="X-Api-Key")
    ).headers == {"X-Api-Key": "SECRET"}
    q = resolver._build_auth(_cred(auth_scheme="query", query_param="token"))
    assert q.query == {"token": "SECRET"}
    assert q.apply_to_url("https://civitai.com/x") == "https://civitai.com/x?token=SECRET"


# ----- DB-backed store + resolver -----


def test_store_upsert_is_write_only_and_masked():
    async def _run():
        view = await CREDENTIAL_STORE.upsert("civitai.com", "abcd1234", label="my key")
        # The view never carries the secret, only the last 4.
        assert not hasattr(view, "secret")
        assert view.secret_last4 == "1234"
        assert view.host == "civitai.com"
        listed = await CREDENTIAL_STORE.list()
        assert any(v.host == "civitai.com" for v in listed)
        await CREDENTIAL_STORE.delete(view.id)
    asyncio.run(_run())


def test_query_scheme_requires_param():
    async def _run():
        with pytest.raises(CredentialValidationError):
            await CREDENTIAL_STORE.upsert("civitai.com", "k", auth_scheme="query")
    asyncio.run(_run())


def test_resolver_never_crosses_host_boundary():
    async def _run():
        view = await CREDENTIAL_STORE.upsert("huggingface.co", "hf_secret_key")
        try:
            # matching host over https -> attached
            auth = await resolver.resolve_auth_for_hop("huggingface.co", "https")
            assert auth is not None
            assert auth.headers["Authorization"] == "Bearer hf_secret_key"
            # CDN redirect host -> dropped
            assert await resolver.resolve_auth_for_hop("cdn-lfs.huggingface.co", "https") is None
            # non-https hop -> never attached
            assert await resolver.resolve_auth_for_hop("huggingface.co", "http") is None
        finally:
            await CREDENTIAL_STORE.delete(view.id)
    asyncio.run(_run())
