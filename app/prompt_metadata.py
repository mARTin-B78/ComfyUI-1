"""Per-prompt metadata envelope shared between submission and outbound events.

The metadata envelope is a small flat ``dict[str, str]`` (e.g.
``{"workflow_id": ...}``) attached to a prompt at submission and injected
by the server into every outbound execution event that carries a
``prompt_id``. It lets consumers scope state by tags they care about
(workflow, trace, tenant) without the execution layer ever needing to
know those tags exist.

This module is intentionally pure — no imports from ``server`` or
``execution`` — so ``PromptServer`` can own a ``PromptMetadataStore``
instance and the helpers can be unit-tested without the rest of the app.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional


# Bounds. The envelope is forwarded to every WebSocket client connected to
# the server on every execution event for the prompt — bounding key count,
# key length, value length, and refusing nested structures keeps a
# malicious or buggy client from inflating the broadcast volume.
MAX_ENVELOPE_KEYS = 16
MAX_ENVELOPE_KEY_LEN = 64
MAX_ENVELOPE_VALUE_LEN = 256

# Cap on concurrently registered prompt envelopes. Acts as a backstop if
# the cleanup hook is ever bypassed; FIFO eviction so the oldest stale
# entry goes first.
DEFAULT_STORE_CAPACITY = 4096


def _sanitize_envelope(envelope: Any) -> Optional[dict]:
    """Validate and copy a candidate envelope.

    Enforces the ``dict[str, str]`` contract that downstream consumers
    (cloud projections, frontend zod schemas, OpenAPI docs) rely on:

    - must be a non-empty ``dict``
    - at most ``MAX_ENVELOPE_KEYS`` entries
    - every key and value must be a ``str``
    - keys at most ``MAX_ENVELOPE_KEY_LEN`` chars
    - values at most ``MAX_ENVELOPE_VALUE_LEN`` chars

    Returns a defensive shallow copy on success, ``None`` on any
    violation. Logs a warning on violation so abuse is visible.
    """
    if not isinstance(envelope, dict) or not envelope:
        return None
    if len(envelope) > MAX_ENVELOPE_KEYS:
        logging.warning(
            "prompt metadata envelope rejected: %d keys exceeds limit %d",
            len(envelope), MAX_ENVELOPE_KEYS,
        )
        return None
    sanitized: dict[str, str] = {}
    for key, value in envelope.items():
        if not isinstance(key, str) or not isinstance(value, str):
            logging.warning(
                "prompt metadata envelope rejected: non-string key/value (%s=%s)",
                type(key).__name__, type(value).__name__,
            )
            return None
        if len(key) > MAX_ENVELOPE_KEY_LEN or len(value) > MAX_ENVELOPE_VALUE_LEN:
            logging.warning(
                "prompt metadata envelope rejected: key or value exceeds length limit",
            )
            return None
        sanitized[key] = value
    return sanitized


def extract_envelope_from_extra_data(extra_data: Any) -> Optional[dict]:
    """Pull the per-prompt metadata envelope out of a submitted prompt's
    ``extra_data``.

    Two sources, in order:

    1. Explicit ``extra_data["metadata"]`` — sanitized via
       ``_sanitize_envelope``. Oversized or wrong-typed envelopes are
       rejected (a warning is logged) rather than truncated, so the
       contract stays strict at the boundary.
    2. ``extra_data["extra_pnginfo"]["workflow"]["id"]`` — backward-
       compatibility fallback. Frontends that already stamp the workflow
       id into ``extra_pnginfo`` keep working; the synthesized envelope
       is ``{"workflow_id": <id>}``. A debug log fires so the legacy path
       remains observable.

    Returns ``None`` when neither source yields a usable envelope.
    """
    if not isinstance(extra_data, dict):
        return None

    if "metadata" in extra_data:
        sanitized = _sanitize_envelope(extra_data["metadata"])
        if sanitized is not None:
            return sanitized
        # Explicit metadata was supplied but rejected — do not fall
        # through to the legacy path; the caller asked for something
        # specific and got it wrong.
        if isinstance(extra_data["metadata"], dict) and extra_data["metadata"]:
            return None

    extra_pnginfo = extra_data.get("extra_pnginfo")
    if isinstance(extra_pnginfo, dict):
        workflow = extra_pnginfo.get("workflow")
        if isinstance(workflow, dict):
            workflow_id = workflow.get("id")
            if (
                isinstance(workflow_id, str)
                and workflow_id
                and len(workflow_id) <= MAX_ENVELOPE_VALUE_LEN
            ):
                logging.debug(
                    "prompt metadata envelope synthesized from extra_pnginfo.workflow.id"
                )
                return {"workflow_id": workflow_id}

    return None


def inject_envelope(
    data: Any,
    envelope_lookup: Callable[[str], Optional[dict]],
) -> Any:
    """Return ``data`` with the per-prompt envelope's keys spread onto it.

    ``envelope_lookup`` is called with the payload's ``prompt_id`` and is
    expected to return the registered envelope or ``None``. This keeps
    the function pure and avoids depending on any specific storage.

    The envelope's keys are merged onto the payload at the top level so
    consumers can read them directly (e.g. ``event.workflow_id``) —
    matching the wire shape of the prior workflow-id-on-events work and
    avoiding an extra nesting hop for clients. Server-emitted fields on
    the payload always win on collision (``{**envelope, **d}``); a
    misbehaving client cannot shadow ``prompt_id``, ``node``, etc.

    Two payload shapes are handled:

    - **dict** carrying ``prompt_id``. A shallow copy is returned with
      the envelope's keys merged onto it.
    - **(preview_image, metadata_dict) tuple** — the format used by
      ``PREVIEW_IMAGE_WITH_METADATA``. Only the inner dict is augmented;
      the binary preview is passed through by reference.

    No-op for payloads without a ``prompt_id``, prompts with no
    registered envelope, or any other payload shape.
    """
    def inject(d: dict) -> dict:
        if not isinstance(d, dict):
            return d
        prompt_id = d.get("prompt_id")
        if not prompt_id:
            return d
        envelope = envelope_lookup(prompt_id)
        if envelope is None:
            return d
        return {**envelope, **d}

    if isinstance(data, dict):
        return inject(data)
    if isinstance(data, tuple) and len(data) == 2 and isinstance(data[1], dict):
        injected = inject(data[1])
        if injected is data[1]:
            return data
        return (data[0], injected)
    return data


class PromptMetadataStore:
    """Bounded ``prompt_id -> envelope`` map.

    Owned by ``PromptServer``. Populated at submission, drained when the
    prompt finishes, wiped on queue cancel/delete. The FIFO cap is a
    backstop: if any cleanup hook is ever skipped, the store sheds the
    oldest entry instead of growing without bound.

    Access is serialized through a ``threading.Lock``. ``register`` runs
    on the aiohttp event-loop thread, ``unregister`` runs on the
    ``prompt_worker`` thread, and ``inject`` runs on whichever thread
    fires ``send_sync`` (event loop, worker, asset seeder). Individual
    ``dict`` ops are GIL-atomic, but ``register``'s
    ``len() -> pop -> __setitem__`` and ``inject``'s ``get -> {**a, **b}``
    are multi-step compounds whose interleaving without a lock is
    racy. The lock is uncontended in steady state (sub-microsecond
    critical sections) so the cost is negligible.
    """

    def __init__(self, capacity: int = DEFAULT_STORE_CAPACITY):
        self._envelopes: dict[str, dict] = {}
        self._capacity = capacity
        self._lock = threading.Lock()

    def register(self, prompt_id: str, extra_data: Any) -> None:
        envelope = extract_envelope_from_extra_data(extra_data)
        if envelope is None:
            return
        with self._lock:
            if len(self._envelopes) >= self._capacity:
                self._envelopes.pop(next(iter(self._envelopes)))
            self._envelopes[prompt_id] = envelope

    def unregister(self, prompt_id: str) -> None:
        with self._lock:
            self._envelopes.pop(prompt_id, None)

    def inject(self, data: Any) -> Any:
        # Snapshot the envelope under the lock so the spread in
        # ``inject_envelope`` runs against a consistent view even if a
        # concurrent ``register``/``unregister`` is mutating the map.
        def locked_lookup(prompt_id: str) -> Optional[dict]:
            with self._lock:
                return self._envelopes.get(prompt_id)
        return inject_envelope(data, locked_lookup)

    def __len__(self) -> int:
        with self._lock:
            return len(self._envelopes)

    def __contains__(self, prompt_id: str) -> bool:
        with self._lock:
            return prompt_id in self._envelopes
