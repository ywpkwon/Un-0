from __future__ import annotations

from pathlib import Path

from PIL import Image
import torch

from imagenet_data import (
    NUM_CLASSES,
    build_imagenet64_dataloader,
    collate_image_batch,
    decode_image,
)


def _write_tree(root: Path, class_ids: list[int], per_class: int = 2) -> None:
    for c in class_ids:
        d = root / f"{c:05d}"
        d.mkdir(parents=True)
        for i in range(per_class):
            color = (c % 256, (c * 7) % 256, (c * 13) % 256)
            Image.new("RGB", (64, 64), color).save(d / f"{i:08d}.png")


def test_num_classes_is_1000() -> None:
    assert NUM_CLASSES == 1000


def test_decode_image_yields_flat_unit_range_tensor() -> None:
    """A 64x64 RGB image decodes to a flat (12288,) tensor in [-1, 1]."""
    img = Image.new("RGB", (64, 64), (255, 0, 0))
    data = decode_image(img)
    assert data.shape == (3 * 64 * 64,)
    assert float(data.min()) >= -1.0
    assert float(data.max()) <= 1.0
    chw = data.reshape(3, 64, 64)
    assert torch.allclose(chw[0], torch.ones_like(chw[0]))
    assert torch.allclose(chw[1], -torch.ones_like(chw[1]))


def test_collate_image_batch_stacks_into_data_and_class_id() -> None:
    """Collate stacks (tensor, label) pairs into the model's batch contract."""
    batch = [
        (torch.zeros(3 * 64 * 64), 1),
        (torch.ones(3 * 64 * 64), 2),
    ]
    out = collate_image_batch(batch)
    assert set(out) == {"data", "class_id"}
    assert out["data"].shape == (2, 3 * 64 * 64)
    assert out["class_id"].tolist() == [1, 2]
    assert out["class_id"].dtype == torch.long


def test_dataloader_reads_imagefolder_tree(tmp_path: Path) -> None:
    """build_imagenet64_dataloader reads an ImageFolder tree and yields the
    model batch contract with class_id matching the integer folder names."""
    root = tmp_path / "train"
    _write_tree(root, class_ids=[0, 5, 12], per_class=2)
    loader = build_imagenet64_dataloader(
        root=str(root),
        batch_size=2,
        num_workers=0,
        shuffle=False,
        drop_last=False,
    )
    seen_classes: set[int] = set()
    total = 0
    for batch in loader:
        assert batch["data"].shape[1] == 3 * 64 * 64
        assert float(batch["data"].min()) >= -1.0
        assert float(batch["data"].max()) <= 1.0
        seen_classes.update(batch["class_id"].tolist())
        total += batch["data"].shape[0]
    assert total == 6
    # ImageFolder maps sorted dir names -> 0,1,2; we name dirs by class id so the
    # loader must surface those original ids, not the ImageFolder index.
    assert seen_classes == {0, 5, 12}
