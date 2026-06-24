from __future__ import annotations

import sys

import pytest


def test_parser_requires_data_root() -> None:
    """--data-root is required; --val-root/--val-data-local plumbing is public."""
    from un0.train_imagenet import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])  # missing required --data-root
    args = parser.parse_args(["--data-root", "/tmp/x"])  # noqa: S108
    assert args.data_root == "/tmp/x"  # noqa: S108
    assert args.fid_every_epochs == 0


def test_fid_args_present_and_streaming_absent() -> None:
    """The public entry point exposes --val-root and must not import streaming
    nor reference the dropped --val-data-local arg."""
    import un0.train_imagenet as train_imagenet

    parser = train_imagenet.build_parser()
    dests = {a.dest for a in parser._actions}
    assert "val_root" in dests
    assert "val_data_local" not in dests
    assert "data_local" not in dests
    assert "streaming" not in sys.modules


def test_read_val_labels_is_balanced_prefix(tmp_path) -> None:
    """_read_val_labels reads the val ImageFolder tree (no MDS) and returns a
    deterministic, ~class-balanced label prefix for 1-to-1 FID conditioning."""
    from PIL import Image

    from un0.train_imagenet import _read_val_labels

    for c in (0, 1, 2, 3):
        d = tmp_path / f"{c:05d}"
        d.mkdir()
        for i in range(5):
            Image.new("RGB", (64, 64), (c, c, c)).save(d / f"{i:08d}.png")

    labels = _read_val_labels(str(tmp_path), num_labels=8, seed=0)
    assert labels.shape == (8,)
    assert int(labels.min()) >= 0
    assert int(labels.max()) <= 3
    # Deterministic under fixed seed.
    again = _read_val_labels(str(tmp_path), num_labels=8, seed=0)
    assert (labels == again).all()
