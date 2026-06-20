"""Tests for the BE-1641 edit-type MOVE/re-register behavior.

A flag-on ``model_type:`` edit on a filesystem-backed model asset must move the
file to the folder that matches the new ``model_type:<folder_name>`` so the file
location stays coherent with the label (not a label-only relabel). The move runs
off the ``POST /tags`` add path (``apply_tags``).
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy.orm import Session

from app.assets.database.models import Asset, AssetReference
from app.assets.database.queries import (
    add_tags_to_reference,
    ensure_tags_exist,
    get_reference_tags,
)
from app.assets.helpers import get_utc_now
from app.assets.services import ModelMoveError, apply_tags
from app.assets.services.ingest import relocate_model_asset_for_model_type_tags


@pytest.fixture
def model_dirs():
    """Temp model/input/output/temp dirs with a shared on-disk model folder.

    ``diffusion_models`` and ``unet_gguf`` both register the same ``models/unet``
    dir (spec-drift §1 plural membership), so a file there belongs to both.
    """
    with tempfile.TemporaryDirectory() as root:
        root_path = Path(root)
        checkpoints = root_path / "models" / "checkpoints"
        loras = root_path / "models" / "loras"
        unet = root_path / "models" / "unet"
        input_dir = root_path / "input"
        output_dir = root_path / "output"
        temp_dir = root_path / "temp"
        for d in (checkpoints, loras, unet, input_dir, output_dir, temp_dir):
            d.mkdir(parents=True)

        folders = [
            ("checkpoints", [str(checkpoints)]),
            ("loras", [str(loras)]),
            ("diffusion_models", [str(unet)]),
            ("unet_gguf", [str(unet)]),
        ]

        with patch("app.assets.services.path_utils.folder_paths") as mock_fp, patch(
            "app.assets.services.path_utils.get_comfy_models_folders",
            return_value=folders,
        ):
            mock_fp.get_input_directory.return_value = str(input_dir)
            mock_fp.get_output_directory.return_value = str(output_dir)
            mock_fp.get_temp_directory.return_value = str(temp_dir)
            yield {
                "checkpoints": checkpoints,
                "loras": loras,
                "unet": unet,
                "input": input_dir,
                "output": output_dir,
                "temp": temp_dir,
            }


def _make_fs_ref(
    session: Session,
    file_path: Path,
    tags: list[str],
    *,
    contents: bytes = b"weights",
    name: str | None = None,
    owner_id: str = "",
) -> AssetReference:
    """Create an on-disk file plus a filesystem-backed reference carrying ``tags``."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(contents)

    asset = Asset(hash=f"blake3:{file_path.name}", size_bytes=len(contents))
    session.add(asset)
    session.flush()

    now = get_utc_now()
    ref = AssetReference(
        owner_id=owner_id,
        name=name or file_path.name,
        asset_id=asset.id,
        file_path=str(file_path),
        mtime_ns=os.stat(file_path).st_mtime_ns,
        created_at=now,
        updated_at=now,
        last_access_time=now,
    )
    session.add(ref)
    session.flush()

    if tags:
        ensure_tags_exist(session, tags)
        add_tags_to_reference(session, reference_id=ref.id, tags=tags)
    session.commit()
    return ref


def _tags_after(session: Session, reference_id: str) -> set[str]:
    session.expire_all()
    return set(get_reference_tags(session, reference_id))


class TestMoveHappyPath:
    def test_checkpoint_to_lora_moves_file_and_reconciles_tags(
        self, mock_create_session, session: Session, model_dirs
    ):
        src = model_dirs["checkpoints"] / "model.safetensors"
        ref = _make_fs_ref(session, src, ["models", "model_type:checkpoints"])

        apply_tags(reference_id=ref.id, tags=["model_type:loras"])

        dst = model_dirs["loras"] / "model.safetensors"
        assert not src.exists()
        assert dst.exists()
        assert dst.read_bytes() == b"weights"

        session.expire_all()
        moved = session.get(AssetReference, ref.id)
        assert moved.file_path == str(dst)

        tags = _tags_after(session, ref.id)
        assert "model_type:loras" in tags
        assert "model_type:checkpoints" not in tags
        assert "models" in tags

    def test_preserves_subfolder_structure(
        self, mock_create_session, session: Session, model_dirs
    ):
        src = model_dirs["checkpoints"] / "sub" / "nested" / "m.safetensors"
        ref = _make_fs_ref(session, src, ["models", "model_type:checkpoints"])

        apply_tags(reference_id=ref.id, tags=["model_type:loras"])

        dst = model_dirs["loras"] / "sub" / "nested" / "m.safetensors"
        assert not src.exists()
        assert dst.exists()
        session.expire_all()
        assert session.get(AssetReference, ref.id).file_path == str(dst)

    def test_preserves_user_labels(
        self, mock_create_session, session: Session, model_dirs
    ):
        src = model_dirs["checkpoints"] / "labelled.safetensors"
        ref = _make_fs_ref(
            session, src, ["models", "model_type:checkpoints", "favorite", "sdxl"]
        )

        apply_tags(reference_id=ref.id, tags=["model_type:loras"])

        tags = _tags_after(session, ref.id)
        assert {"favorite", "sdxl"} <= tags

    def test_refreshes_filename_metadata(
        self, mock_create_session, session: Session, model_dirs
    ):
        src = model_dirs["checkpoints"] / "deep" / "m.safetensors"
        ref = _make_fs_ref(session, src, ["models", "model_type:checkpoints"])

        apply_tags(reference_id=ref.id, tags=["model_type:loras"])

        session.expire_all()
        moved = session.get(AssetReference, ref.id)
        # filename is relative to the category root, so the subfolder survives.
        assert moved.user_metadata["filename"] == "deep/m.safetensors"


class TestNoMoveCases:
    def test_same_type_is_noop(
        self, mock_create_session, session: Session, model_dirs
    ):
        src = model_dirs["checkpoints"] / "m.safetensors"
        ref = _make_fs_ref(session, src, ["models", "model_type:checkpoints"])

        moved = relocate_model_asset_for_model_type_tags(
            session, ref, ["model_type:checkpoints"]
        )
        assert moved is False
        assert src.exists()

    def test_shared_dir_does_not_move(
        self, mock_create_session, session: Session, model_dirs
    ):
        # File in the shared unet dir belongs to BOTH diffusion_models and
        # unet_gguf; editing to the sibling that shares the dir must not move.
        src = model_dirs["unet"] / "g.gguf"
        ref = _make_fs_ref(
            session,
            src,
            ["models", "model_type:diffusion_models", "model_type:unet_gguf"],
        )

        moved = relocate_model_asset_for_model_type_tags(
            session, ref, ["model_type:unet_gguf"]
        )
        assert moved is False
        assert src.exists()

    def test_hash_only_reference_is_label_only(
        self, mock_create_session, session: Session, model_dirs
    ):
        asset = Asset(hash="blake3:hashonly", size_bytes=10)
        session.add(asset)
        session.flush()
        now = get_utc_now()
        ref = AssetReference(
            name="hashonly",
            asset_id=asset.id,
            file_path=None,
            created_at=now,
            updated_at=now,
            last_access_time=now,
        )
        session.add(ref)
        session.commit()

        moved = relocate_model_asset_for_model_type_tags(
            session, ref, ["model_type:loras"]
        )
        assert moved is False

    def test_non_model_filesystem_asset_is_label_only(
        self, mock_create_session, session: Session, model_dirs
    ):
        src = model_dirs["input"] / "photo.png"
        ref = _make_fs_ref(session, src, ["input"])

        moved = relocate_model_asset_for_model_type_tags(
            session, ref, ["model_type:loras"]
        )
        assert moved is False
        assert src.exists()

    def test_no_model_type_tag_is_noop(
        self, mock_create_session, session: Session, model_dirs
    ):
        src = model_dirs["checkpoints"] / "m.safetensors"
        ref = _make_fs_ref(session, src, ["models", "model_type:checkpoints"])

        moved = relocate_model_asset_for_model_type_tags(
            session, ref, ["favorite"]
        )
        assert moved is False
        assert src.exists()


class TestUnknownFolderIsLabelOnly:
    def test_unknown_folder_is_stored_as_label_not_rejected(
        self, mock_create_session, session: Session, model_dirs
    ):
        # Core stays permissive about model_type: LABELS (spec-drift §3): an
        # unregistered folder_name can't map to a real location, so it's a plain
        # label, not a move and not a reject.
        src = model_dirs["checkpoints"] / "m.safetensors"
        ref = _make_fs_ref(session, src, ["models", "model_type:checkpoints"])

        result = apply_tags(reference_id=ref.id, tags=["model_type:bogus"])

        assert src.exists()  # not moved
        assert "model_type:bogus" in result.total_tags
        # The real path-derived type is untouched.
        assert "model_type:checkpoints" in result.total_tags


class TestRejects:
    def test_collision_rejected_409_and_not_clobbered(
        self, mock_create_session, session: Session, model_dirs
    ):
        src = model_dirs["checkpoints"] / "m.safetensors"
        ref = _make_fs_ref(session, src, ["models", "model_type:checkpoints"])

        # Pre-existing file at the destination must not be overwritten.
        dst = model_dirs["loras"] / "m.safetensors"
        dst.write_bytes(b"existing-lora")

        with pytest.raises(ModelMoveError) as ei:
            apply_tags(reference_id=ref.id, tags=["model_type:loras"])
        assert ei.value.status == 409
        assert ei.value.code == "DESTINATION_EXISTS"
        assert src.exists()
        assert dst.read_bytes() == b"existing-lora"

    def test_collision_with_registered_reference_rejected(
        self, mock_create_session, session: Session, model_dirs
    ):
        src = model_dirs["checkpoints"] / "m.safetensors"
        ref = _make_fs_ref(session, src, ["models", "model_type:checkpoints"])

        # Another reference already owns the destination path (no on-disk file).
        dst = model_dirs["loras"] / "m.safetensors"
        other = Asset(hash="blake3:other", size_bytes=5)
        session.add(other)
        session.flush()
        now = get_utc_now()
        session.add(
            AssetReference(
                name="m.safetensors",
                asset_id=other.id,
                file_path=str(dst),
                created_at=now,
                updated_at=now,
                last_access_time=now,
            )
        )
        session.commit()

        with pytest.raises(ModelMoveError) as ei:
            apply_tags(reference_id=ref.id, tags=["model_type:loras"])
        assert ei.value.code == "DESTINATION_EXISTS"
        assert src.exists()


class TestRollback:
    def test_file_rolled_back_when_db_update_fails(
        self, mock_create_session, session: Session, model_dirs
    ):
        src = model_dirs["checkpoints"] / "m.safetensors"
        ref = _make_fs_ref(session, src, ["models", "model_type:checkpoints"])
        dst = model_dirs["loras"] / "m.safetensors"

        with patch(
            "app.assets.services.ingest.get_size_and_mtime_ns",
            side_effect=OSError("boom"),
        ):
            with pytest.raises(OSError, match="boom"):
                apply_tags(reference_id=ref.id, tags=["model_type:loras"])

        # The half-move must be undone: source restored, destination clean.
        assert src.exists()
        assert not dst.exists()
