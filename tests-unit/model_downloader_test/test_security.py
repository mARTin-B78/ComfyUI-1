"""Unit tests for the security layer: allowlist, SSRF checks, path safety."""

from __future__ import annotations

import pytest

from app.model_downloader.security import allowlist, paths
from app.model_downloader.security.ssrf import (
    SSRFError,
    check_redirect_hop,
    is_blocked_ip,
)


# ----- allowlist -----


@pytest.mark.parametrize(
    "url,allowed",
    [
        ("https://huggingface.co/org/repo/resolve/main/model.safetensors", True),
        ("https://civitai.com/api/download/x/model.safetensors", True),
        ("http://localhost/model.safetensors", True),
        # off-list host
        ("https://evil.example.com/model.safetensors", False),
        # http to a non-loopback allowlisted host is not permitted (https only)
        ("http://huggingface.co/org/repo/resolve/main/model.safetensors", False),
        # bad extension on an allowed host
        ("https://huggingface.co/org/repo/resolve/main/config.json", False),
        # userinfo trick: real host is the metadata IP, not 127.0.0.1
        ("http://127.0.0.1@169.254.169.254/x.safetensors", False),
    ],
)
def test_is_url_allowed(url, allowed):
    assert allowlist.is_url_allowed(url) is allowed


def test_allow_any_extension_relaxes_extension_only():
    url = "https://huggingface.co/org/repo/resolve/main/weights.bin"
    assert allowlist.is_url_allowed(url) is True  # .bin is in the known set
    odd = "https://huggingface.co/org/repo/resolve/main/weights.zip"
    assert allowlist.is_url_allowed(odd) is False
    assert allowlist.is_url_allowed(odd, allow_any_extension=True) is True


# ----- SSRF: blocked IPs -----


@pytest.mark.parametrize(
    "ip,blocked",
    [
        ("169.254.169.254", True),  # cloud metadata / link-local
        ("127.0.0.1", True),
        ("10.0.0.5", True),
        ("192.168.1.1", True),
        ("172.16.0.1", True),
        ("::1", True),
        ("0.0.0.0", True),
        # IPv4-mapped IPv6: must see through the mapping even on CPython
        # versions predating the gh-113171 is_* property fix.
        ("::ffff:169.254.169.254", True),  # mapped cloud metadata
        ("::ffff:127.0.0.1", True),  # mapped loopback
        ("::ffff:10.0.0.1", True),  # mapped RFC1918
        ("::ffff:8.8.8.8", False),  # mapped public address stays allowed
        ("8.8.8.8", False),
        ("1.1.1.1", False),
        ("not-an-ip", True),  # unparseable -> refuse
    ],
)
def test_is_blocked_ip(ip, blocked):
    assert is_blocked_ip(ip) is blocked


# ----- SSRF: redirect hop validation -----


def test_check_redirect_hop_rejects_bad_scheme_and_userinfo():
    with pytest.raises(SSRFError):
        check_redirect_hop("ftp://huggingface.co/x.safetensors")
    with pytest.raises(SSRFError):
        check_redirect_hop("https://user:pass@cdn.example.com/x")
    # A CDN host that is NOT on the allowlist is allowed as a redirect target
    # (private-IP protection is the resolver's job; credential leak is prevented
    # by exact host matching).
    assert check_redirect_hop("https://cdn-lfs.huggingface.co/abc") is not None


def test_check_redirect_hop_http_only_for_loopback():
    # Plain http to an external host is rejected (no plaintext downgrade).
    with pytest.raises(SSRFError):
        check_redirect_hop("http://cdn-lfs.huggingface.co/abc")
    # http is still honored for loopback/approved dev hosts.
    assert check_redirect_hop("http://localhost/x.safetensors") is not None
    assert check_redirect_hop("http://127.0.0.1/x.safetensors") is not None


# ----- path safety -----


def test_parse_model_id_valid(model_root):
    directory, filename = paths.parse_model_id("loras/my_lora.safetensors")
    assert directory == "loras"
    assert filename == "my_lora.safetensors"


@pytest.mark.parametrize(
    "model_id",
    [
        "loras/../etc/passwd.safetensors",  # traversal
        "loras/sub/dir.safetensors",  # nested
        "unknownfolder/x.safetensors",  # unknown folder
        "loras/model.txt",  # bad extension
        "noslash.safetensors",  # missing directory
        "loras/",  # empty filename
    ],
)
def test_parse_model_id_rejects(model_root, model_id):
    with pytest.raises(paths.InvalidModelId):
        paths.parse_model_id(model_id)


def test_resolve_destination_stays_in_root(model_root):
    final_path, temp_path = paths.resolve_destination("loras/x.safetensors")
    assert final_path.startswith(model_root)
    assert temp_path.startswith(model_root)
    assert temp_path != final_path
