"""Unit tests for the metadata-envelope module in ``app.prompt_metadata``.

Covers the two pure helpers (``extract_envelope_from_extra_data`` and
``inject_envelope``) and the ``PromptMetadataStore`` integration class
that ``PromptServer`` owns.
"""

from __future__ import annotations

import pytest

from app.prompt_metadata import (
    MAX_ENVELOPE_KEYS,
    MAX_ENVELOPE_KEY_LEN,
    MAX_ENVELOPE_VALUE_LEN,
    PromptMetadataStore,
    extract_envelope_from_extra_data,
    inject_envelope,
)


class TestExtractEnvelopeFromExtraData:
    def test_explicit_metadata_dict_is_used_as_is(self):
        extra_data = {"metadata": {"workflow_id": "wf-1", "trace_id": "t-9"}}
        assert extract_envelope_from_extra_data(extra_data) == {
            "workflow_id": "wf-1",
            "trace_id": "t-9",
        }

    def test_explicit_metadata_takes_precedence_over_extra_pnginfo(self):
        extra_data = {
            "metadata": {"workflow_id": "explicit"},
            "extra_pnginfo": {"workflow": {"id": "fallback"}},
        }
        assert extract_envelope_from_extra_data(extra_data) == {
            "workflow_id": "explicit"
        }

    def test_falls_back_to_extra_pnginfo_workflow_id(self):
        extra_data = {"extra_pnginfo": {"workflow": {"id": "wf-legacy"}}}
        assert extract_envelope_from_extra_data(extra_data) == {
            "workflow_id": "wf-legacy"
        }

    def test_returns_none_when_no_metadata_and_no_workflow_id(self):
        assert extract_envelope_from_extra_data({}) is None
        assert (
            extract_envelope_from_extra_data({"extra_pnginfo": {"workflow": {}}})
            is None
        )

    @pytest.mark.parametrize("bad", ["", 123, None, [], {}])
    def test_rejects_non_string_or_empty_workflow_id(self, bad):
        extra_data = {"extra_pnginfo": {"workflow": {"id": bad}}}
        assert extract_envelope_from_extra_data(extra_data) is None

    def test_rejects_non_dict_inputs_at_each_level(self):
        assert extract_envelope_from_extra_data(None) is None
        assert extract_envelope_from_extra_data("not-a-dict") is None
        assert (
            extract_envelope_from_extra_data({"extra_pnginfo": "not-a-dict"})
            is None
        )
        assert (
            extract_envelope_from_extra_data(
                {"extra_pnginfo": {"workflow": "not-a-dict"}}
            )
            is None
        )

    def test_empty_explicit_metadata_falls_through_to_workflow_id(self):
        extra_data = {
            "metadata": {},
            "extra_pnginfo": {"workflow": {"id": "wf-legacy"}},
        }
        assert extract_envelope_from_extra_data(extra_data) == {
            "workflow_id": "wf-legacy"
        }

    def test_returned_envelope_is_copy_not_reference(self):
        original = {"workflow_id": "wf-1"}
        result = extract_envelope_from_extra_data({"metadata": original})
        assert result is not None
        result["new_key"] = "x"
        assert "new_key" not in original

    def test_non_dict_explicit_metadata_falls_through_to_workflow_id(self):
        extra_data = {
            "metadata": "not-a-dict",
            "extra_pnginfo": {"workflow": {"id": "wf-legacy"}},
        }
        assert extract_envelope_from_extra_data(extra_data) == {
            "workflow_id": "wf-legacy"
        }


class TestEnvelopeSanitization:
    """The wire contract is ``dict[str, str]`` with bounded size. A bad
    envelope is dropped (and a warning is logged) rather than truncated,
    so the boundary stays strict."""

    def test_rejects_too_many_keys(self, caplog):
        envelope = {f"k{i}": "v" for i in range(MAX_ENVELOPE_KEYS + 1)}
        with caplog.at_level("WARNING"):
            assert extract_envelope_from_extra_data({"metadata": envelope}) is None
        assert any("exceeds limit" in r.message for r in caplog.records)

    def test_accepts_max_keys_exactly(self):
        envelope = {f"k{i}": "v" for i in range(MAX_ENVELOPE_KEYS)}
        assert extract_envelope_from_extra_data({"metadata": envelope}) == envelope

    def test_rejects_non_string_keys(self, caplog):
        with caplog.at_level("WARNING"):
            assert (
                extract_envelope_from_extra_data({"metadata": {42: "v"}})
                is None
            )
        assert any("non-string" in r.message for r in caplog.records)

    def test_rejects_non_string_values(self, caplog):
        for bad_value in [42, None, ["x"], {"nested": "dict"}, b"bytes"]:
            with caplog.at_level("WARNING"):
                assert (
                    extract_envelope_from_extra_data(
                        {"metadata": {"k": bad_value}}
                    )
                    is None
                )

    def test_rejects_oversized_key(self):
        envelope = {"x" * (MAX_ENVELOPE_KEY_LEN + 1): "v"}
        assert extract_envelope_from_extra_data({"metadata": envelope}) is None

    def test_rejects_oversized_value(self):
        envelope = {"k": "x" * (MAX_ENVELOPE_VALUE_LEN + 1)}
        assert extract_envelope_from_extra_data({"metadata": envelope}) is None

    def test_accepts_max_lengths_exactly(self):
        envelope = {
            "x" * MAX_ENVELOPE_KEY_LEN: "y" * MAX_ENVELOPE_VALUE_LEN
        }
        assert extract_envelope_from_extra_data({"metadata": envelope}) == envelope

    def test_oversized_workflow_id_in_pnginfo_rejected(self):
        """The legacy synthesized path also respects the value bound."""
        extra_data = {
            "extra_pnginfo": {
                "workflow": {"id": "x" * (MAX_ENVELOPE_VALUE_LEN + 1)}
            }
        }
        assert extract_envelope_from_extra_data(extra_data) is None

    def test_invalid_explicit_metadata_does_not_fall_through(self):
        """An explicit but invalid metadata dict means the caller asked
        for something specific and got it wrong; the synthesized
        fallback must not silently substitute."""
        extra_data = {
            "metadata": {"k": 42},  # non-string value
            "extra_pnginfo": {"workflow": {"id": "wf-legacy"}},
        }
        assert extract_envelope_from_extra_data(extra_data) is None


class TestInjectEnvelope:
    @staticmethod
    def _lookup(table):
        return table.get

    def test_spreads_envelope_keys_onto_payload(self):
        """Envelope keys are merged at the top level so consumers can
        read them directly (e.g. ``event.workflow_id``)."""
        lookup = self._lookup({"p1": {"workflow_id": "wf-1", "trace_id": "t-9"}})
        assert inject_envelope({"node": "5", "prompt_id": "p1"}, lookup) == {
            "node": "5",
            "prompt_id": "p1",
            "workflow_id": "wf-1",
            "trace_id": "t-9",
        }

    def test_passthrough_when_prompt_id_not_registered(self):
        lookup = self._lookup({})
        data = {"node": "5", "prompt_id": "unknown"}
        assert inject_envelope(data, lookup) == data

    def test_passthrough_when_payload_lacks_prompt_id(self):
        lookup = self._lookup({"p1": {"workflow_id": "wf-1"}})
        data = {"status": "ok"}
        assert inject_envelope(data, lookup) == data

    def test_server_keys_win_on_collision_with_envelope(self):
        """A misbehaving client cannot shadow server-emitted fields by
        stamping the same key in their submission envelope."""
        lookup = self._lookup({
            "p1": {"prompt_id": "client-claimed", "node": "spoofed", "workflow_id": "wf-1"}
        })
        result = inject_envelope({"prompt_id": "p1", "node": "5"}, lookup)
        assert result["prompt_id"] == "p1"
        assert result["node"] == "5"
        assert result["workflow_id"] == "wf-1"

    def test_does_not_mutate_input_dict(self):
        lookup = self._lookup({"p1": {"workflow_id": "wf-1"}})
        original = {"node": "5", "prompt_id": "p1"}
        inject_envelope(original, lookup)
        assert "workflow_id" not in original

    def test_does_not_mutate_envelope_dict(self):
        envelope = {"workflow_id": "wf-1"}
        lookup = self._lookup({"p1": envelope})
        inject_envelope({"prompt_id": "p1", "node": "5"}, lookup)
        assert envelope == {"workflow_id": "wf-1"}

    def test_injects_into_inner_dict_of_preview_metadata_tuple(self):
        """``PREVIEW_IMAGE_WITH_METADATA`` payloads arrive as
        ``(preview_image, metadata_dict)``; the inner dict is the only
        place the envelope can attach."""
        lookup = self._lookup({"p1": {"workflow_id": "wf-1"}})
        preview_image = ("PNG", object(), 256)
        inner = {"node_id": "5", "prompt_id": "p1"}
        result = inject_envelope((preview_image, inner), lookup)
        assert isinstance(result, tuple)
        assert result[0] is preview_image
        assert result[1] == {
            "node_id": "5",
            "prompt_id": "p1",
            "workflow_id": "wf-1",
        }
        assert "workflow_id" not in inner

    def test_preview_tuple_passthrough_when_no_envelope_registered(self):
        lookup = self._lookup({})
        preview_image = ("PNG", object(), 256)
        inner = {"node_id": "5", "prompt_id": "unknown"}
        result = inject_envelope((preview_image, inner), lookup)
        assert result == (preview_image, inner)

    @pytest.mark.parametrize("payload", [b"raw-bytes", None, 42])
    def test_non_dict_non_tuple_payloads_passthrough(self, payload):
        lookup = self._lookup({"p1": {"workflow_id": "wf-1"}})
        assert inject_envelope(payload, lookup) == payload

    def test_tuple_of_wrong_arity_passthrough(self):
        """Only the 2-tuple ``(preview, metadata_dict)`` shape is
        special-cased. Other tuples must not be touched."""
        lookup = self._lookup({"p1": {"workflow_id": "wf-1"}})
        triple = (1, {"prompt_id": "p1"}, 3)
        assert inject_envelope(triple, lookup) is triple

    def test_envelope_lookup_called_per_invocation(self):
        """The lookup runs each time the function is called, so changes
        to the backing store are immediately visible."""
        store = {"p1": {"workflow_id": "wf-1"}}
        first = inject_envelope({"prompt_id": "p1"}, store.get)
        store["p1"] = {"workflow_id": "wf-2"}
        second = inject_envelope({"prompt_id": "p1"}, store.get)
        del store["p1"]
        third = inject_envelope({"prompt_id": "p1"}, store.get)
        assert first["workflow_id"] == "wf-1"
        assert second["workflow_id"] == "wf-2"
        assert "workflow_id" not in third


class TestPromptMetadataStore:
    """End-to-end wiring tests that exercise the full register/inject/
    unregister cycle the way ``PromptServer`` does."""

    def test_register_inject_unregister_cycle(self):
        store = PromptMetadataStore()
        store.register(
            "p1", {"extra_pnginfo": {"workflow": {"id": "wf-1"}}}
        )
        injected = store.inject({"node": "5", "prompt_id": "p1"})
        assert injected == {
            "node": "5",
            "prompt_id": "p1",
            "workflow_id": "wf-1",
        }
        store.unregister("p1")
        passthrough = store.inject({"node": "5", "prompt_id": "p1"})
        assert "workflow_id" not in passthrough

    def test_register_with_no_derivable_envelope_is_noop(self):
        store = PromptMetadataStore()
        store.register("p1", {})
        assert "p1" not in store
        data = {"prompt_id": "p1"}
        assert store.inject(data) == data

    def test_register_with_oversized_envelope_is_noop(self):
        """Sanitization rejection means nothing is registered — the
        store stays empty and inject is a passthrough."""
        store = PromptMetadataStore()
        store.register(
            "p1",
            {"metadata": {f"k{i}": "v" for i in range(MAX_ENVELOPE_KEYS + 1)}},
        )
        assert "p1" not in store

    def test_unregister_unknown_prompt_is_silent(self):
        store = PromptMetadataStore()
        store.unregister("does-not-exist")

    def test_fifo_eviction_when_capacity_exceeded(self):
        """If cleanup hooks are ever bypassed, the store must shed the
        oldest entry rather than grow without bound."""
        store = PromptMetadataStore(capacity=3)
        store.register("p1", {"metadata": {"workflow_id": "wf-1"}})
        store.register("p2", {"metadata": {"workflow_id": "wf-2"}})
        store.register("p3", {"metadata": {"workflow_id": "wf-3"}})
        assert len(store) == 3

        store.register("p4", {"metadata": {"workflow_id": "wf-4"}})
        assert len(store) == 3
        assert "p1" not in store
        assert "p4" in store

        # The newer entries are still injectable.
        assert store.inject({"prompt_id": "p4"})["workflow_id"] == "wf-4"
        # The evicted one is gone.
        assert "workflow_id" not in store.inject({"prompt_id": "p1"})

    def test_register_after_unregister_does_not_count_against_capacity(self):
        """Normal lifecycle: register, unregister, register many — the
        store should not silently evict valid entries because of stale
        accounting."""
        store = PromptMetadataStore(capacity=2)
        for i in range(10):
            store.register(f"p{i}", {"metadata": {"workflow_id": f"wf-{i}"}})
            store.unregister(f"p{i}")
            assert len(store) == 0

    def test_re_register_overwrites(self):
        store = PromptMetadataStore()
        store.register("p1", {"metadata": {"workflow_id": "wf-1"}})
        store.register("p1", {"metadata": {"workflow_id": "wf-2"}})
        assert store.inject({"prompt_id": "p1"})["workflow_id"] == "wf-2"

    def test_inject_with_no_registrations_is_passthrough(self):
        store = PromptMetadataStore()
        data = {"prompt_id": "p1", "node": "5"}
        assert store.inject(data) == data

    def test_inject_into_preview_tuple(self):
        store = PromptMetadataStore()
        store.register("p1", {"metadata": {"workflow_id": "wf-1"}})
        result = store.inject((b"image-bytes", {"prompt_id": "p1"}))
        assert result == (b"image-bytes", {
            "prompt_id": "p1",
            "workflow_id": "wf-1",
        })

    def test_concurrent_access_does_not_corrupt_or_raise(self):
        """Smoke test for the store's lock. ``register`` is called from
        the aiohttp event-loop thread, ``unregister`` from the worker
        thread, and ``inject`` fires on every ``send_sync`` from
        whichever thread emits the event. Run all three concurrently
        and assert no exception escapes and the store stays internally
        consistent (the FIFO cap is never exceeded)."""
        import threading

        store = PromptMetadataStore(capacity=64)
        stop = threading.Event()
        errors: list[BaseException] = []

        def registrar():
            i = 0
            try:
                while not stop.is_set():
                    store.register(
                        f"p{i % 100}",
                        {"metadata": {"workflow_id": f"wf-{i}"}},
                    )
                    i += 1
            except BaseException as e:
                errors.append(e)

        def canceller():
            i = 0
            try:
                while not stop.is_set():
                    store.unregister(f"p{i % 100}")
                    i += 1
            except BaseException as e:
                errors.append(e)

        def injector():
            i = 0
            try:
                while not stop.is_set():
                    store.inject({"prompt_id": f"p{i % 100}", "node": "5"})
                    i += 1
            except BaseException as e:
                errors.append(e)

        threads = [
            threading.Thread(target=registrar),
            threading.Thread(target=registrar),
            threading.Thread(target=canceller),
            threading.Thread(target=injector),
            threading.Thread(target=injector),
        ]
        for t in threads:
            t.start()
        # Brief burst — long enough to interleave many ops, short enough
        # not to slow CI.
        threading.Event().wait(0.1)
        stop.set()
        for t in threads:
            t.join(timeout=2.0)

        assert errors == [], f"concurrent access raised: {errors[:3]}"
        assert len(store) <= 64, "FIFO cap was breached under contention"
