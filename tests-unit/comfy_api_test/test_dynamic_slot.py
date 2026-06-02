"""Unit tests for the redesigned ``DynamicSlot`` with type-keyed options."""

import pytest

from comfy_api.latest import _io as io


def _opt(when, ids=None):
    """Build an Option whose inputs are placeholder String widgets named after ids."""
    ids = ids or []
    inputs = [io.String.Input(name) for name in ids]
    return io.DynamicSlot.Option(when=when, inputs=inputs)


# ---------------------------------------------------------------------------
# Option.when normalization
# ---------------------------------------------------------------------------

def test_option_when_none():
    o = _opt(None, ["a"])
    assert o._when_types is None
    assert o.as_dict()["when"] is None


def test_option_when_single_type():
    o = _opt(io.Image)
    assert o._when_types == frozenset({"IMAGE"})
    assert o.as_dict()["when"] == ["IMAGE"]


def test_option_when_anytype():
    o = _opt(io.AnyType)
    assert o._when_types == frozenset({"*"})
    assert o.as_dict()["when"] == ["*"]


def test_option_when_list():
    o = _opt([io.Image, io.Mask])
    assert o._when_types == frozenset({"IMAGE", "MASK"})
    # list form sorted for stable serialization
    assert o.as_dict()["when"] == ["IMAGE", "MASK"]


def test_option_when_multitype_input():
    mt = io.MultiType.Input("x", types=[io.Image, io.Latent])
    o = _opt(mt)
    assert o._when_types == frozenset({"IMAGE", "LATENT"})


def test_option_when_empty_list_rejected():
    with pytest.raises(ValueError, match="when=\\[\\]"):
        io.DynamicSlot.Option(when=[], inputs=[])


def test_option_when_garbage_rejected():
    with pytest.raises(ValueError, match="when must be"):
        io.DynamicSlot.Option(when="IMAGE", inputs=[])


def test_option_when_list_with_non_comfytype_rejected():
    with pytest.raises(ValueError, match="list entries"):
        io.DynamicSlot.Option(when=[io.Image, "MASK"], inputs=[])


# ---------------------------------------------------------------------------
# DynamicSlot.Input construction and serialization
# ---------------------------------------------------------------------------

def test_input_requires_at_least_one_option():
    with pytest.raises(ValueError, match="at least one Option"):
        io.DynamicSlot.Input("x", options=[])


def test_input_requires_non_none_option():
    with pytest.raises(ValueError, match="non-None `when`"):
        io.DynamicSlot.Input("x", options=[_opt(None, ["a"])])


def test_input_auto_derives_slot_type():
    inp = io.DynamicSlot.Input("x", options=[
        _opt(io.Image, ["a"]),
        _opt(io.Mask, ["b"]),
        _opt(None, ["c"]),
    ])
    # Declared order preserved across non-None options; None contributes nothing.
    # Note: get_io_type() intentionally still returns the dynamic class io_type
    # (COMFY_DYNAMICSLOT_V3) so parse_class_inputs dispatches into the expander.
    # The auto-derived slot type is exposed via the `slotType` field of as_dict()
    # and via the private `_slot_io_type` attribute (used by the type resolver).
    assert inp._slot_io_type == "IMAGE,MASK"
    d = inp.as_dict()
    assert d["slotType"] == "IMAGE,MASK"
    assert len(d["options"]) == 3


def test_input_includes_anytype_in_slot_type():
    inp = io.DynamicSlot.Input("x", options=[
        _opt(io.Image, ["a"]),
        _opt(io.AnyType, ["b"]),
    ])
    assert inp._slot_io_type == "IMAGE,*"


def test_input_get_all_dedups_inputs_by_id():
    inp = io.DynamicSlot.Input("x", options=[
        _opt(io.Image, ["shared", "image_only"]),
        _opt(io.Mask, ["shared", "mask_only"]),
    ])
    ids = [i.id for i in inp.get_all()]
    assert ids == ["shared", "image_only", "mask_only"]


# ---------------------------------------------------------------------------
# Option selection
# ---------------------------------------------------------------------------

def _select(options, live_input_types, has_link, finalized_id="x"):
    """Convenience wrapper that runs the dispatch through the dict form (post-as_dict)."""
    serialized = [o.as_dict() for o in options]
    return io.DynamicSlot._select_option(
        serialized, live_input_types, finalized_id, has_link
    )


def test_select_unconnected_picks_none_option():
    options = [_opt(io.Image, ["img_widgets"]), _opt(None, ["empty_widgets"])]
    sel = _select(options, {}, has_link=False)
    assert sel is not None
    assert sel["when"] is None


def test_select_unconnected_with_no_none_option_returns_none():
    options = [_opt(io.Image, ["x"])]
    assert _select(options, {}, has_link=False) is None


def test_select_concrete_type_match():
    options = [
        _opt(io.Image, ["a"]),
        _opt(io.Mask, ["b"]),
        _opt(io.AnyType, ["c"]),
    ]
    sel = _select(options, {"x": "MASK"}, has_link=True)
    assert sel["when"] == ["MASK"]


def test_select_anytype_matches_wildcard_resolved():
    options = [_opt(io.Image, ["a"]), _opt(io.AnyType, ["c"])]
    sel = _select(options, {"x": "*"}, has_link=True)
    assert sel["when"] == ["*"]


def test_select_anytype_does_not_match_concrete():
    options = [_opt(io.AnyType, ["c"])]
    # MASK isn't in any option's set; AnyType only matches "*". No expansion.
    assert _select(options, {"x": "MASK"}, has_link=True) is None


def test_select_first_match_wins():
    options = [
        _opt([io.Image, io.Mask], ["both"]),
        _opt(io.Image, ["image_only"]),
    ]
    # Resolved IMAGE matches both; first option wins.
    sel = _select(options, {"x": "IMAGE"}, has_link=True)
    assert sel["inputs"]
    # The "both" option's first input is named "both"
    first_input_id = next(iter(sel["inputs"]["required"].keys()))
    assert first_input_id == "both"


def test_select_multitype_upstream_intersects_option_set():
    """When upstream declares MultiType like 'IMAGE,MASK', any option that
    intersects with that set matches (first wins)."""
    options = [
        _opt(io.Latent, ["latent_only"]),
        _opt(io.Mask, ["mask_only"]),
    ]
    sel = _select(options, {"x": "IMAGE,MASK"}, has_link=True)
    assert sel["when"] == ["MASK"]


def test_select_missing_resolved_falls_through_to_anytype():
    """If live_input_types lacks an entry for this slot but a link exists,
    we treat it as '*' (resolver default for unresolvable links)."""
    options = [_opt(io.Image, ["a"]), _opt(io.AnyType, ["c"])]
    sel = _select(options, {}, has_link=True)
    assert sel["when"] == ["*"]


# ---------------------------------------------------------------------------
# End-to-end expansion via _expand_schema_for_dynamic
# ---------------------------------------------------------------------------

def test_expand_unconnected_path():
    """An unconnected slot with a `when=None` option expands that option's children."""
    inp = io.DynamicSlot.Input("x", options=[
        _opt(io.Image, ["image_widget"]),
        _opt(None, ["empty_widget"]),
    ])
    d = inp.as_dict()
    value = (io.DynamicSlot.io_type, d)
    out_dict = {
        "required": {}, "optional": {}, "hidden": {},
        "dynamic_paths": {}, "dynamic_paths_default_value": {},
    }
    io.DynamicSlot._expand_schema_for_dynamic(
        out_dict=out_dict,
        live_inputs={},  # no entry for "x" → unconnected
        value=value,
        input_type="optional",
        curr_prefix=["x"],
        live_input_types=None,
    )
    # The slot itself is always advertised in the caller's bucket.
    assert "x" in out_dict["optional"]
    # Children land in their own buckets (required by default) with
    # parent-prefixed ids.
    assert "x.empty_widget" in out_dict["required"]
    assert "x.image_widget" not in out_dict["required"]


def test_expand_typed_path():
    """A connected slot expands the matching type's children."""
    inp = io.DynamicSlot.Input("x", options=[
        _opt(io.Image, ["image_widget"]),
        _opt(io.Mask, ["mask_widget"]),
    ])
    d = inp.as_dict()
    value = (io.DynamicSlot.io_type, d)
    out_dict = {
        "required": {}, "optional": {}, "hidden": {},
        "dynamic_paths": {}, "dynamic_paths_default_value": {},
    }
    io.DynamicSlot._expand_schema_for_dynamic(
        out_dict=out_dict,
        live_inputs={"x": ["src_node", 0]},  # link present
        value=value,
        input_type="optional",
        curr_prefix=["x"],
        live_input_types={"x": "MASK"},
    )
    assert "x" in out_dict["optional"]
    assert "x.mask_widget" in out_dict["required"]
    assert "x.image_widget" not in out_dict["required"]


def test_expand_unmatched_concrete_still_advertises_slot():
    """Resolved type not in any option → no children, but the slot itself stays."""
    inp = io.DynamicSlot.Input("x", options=[_opt(io.Image, ["image_widget"])])
    d = inp.as_dict()
    value = (io.DynamicSlot.io_type, d)
    out_dict = {
        "required": {}, "optional": {}, "hidden": {},
        "dynamic_paths": {}, "dynamic_paths_default_value": {},
    }
    io.DynamicSlot._expand_schema_for_dynamic(
        out_dict=out_dict,
        live_inputs={"x": ["src_node", 0]},
        value=value,
        input_type="optional",
        curr_prefix=["x"],
        live_input_types={"x": "LATENT"},
    )
    assert "x" in out_dict["optional"]
    assert "x.image_widget" not in out_dict["required"]
