from __future__ import annotations

import torch

from un0.data import collate_image_batch


def test_collate_image_batch_flattens_normalizes_and_keeps_labels() -> None:
    """Collate should produce flattened [-1,1] images and int64 labels."""
    batch = [
        {"image": torch.zeros(3, 4, 4), "label": 3},
        {"image": torch.ones(3, 4, 4), "label": 7},
    ]
    collated = collate_image_batch(batch)

    assert collated["data"].shape == (2, 3 * 4 * 4)
    assert collated["data"][0].min() == -1.0
    assert collated["data"][1].max() == 1.0
    assert collated["class_id"].dtype == torch.long
    assert collated["class_id"].tolist() == [3, 7]


def test_collate_includes_sample_id_when_present() -> None:
    """Collate passes through ``sample_id`` when each row has it."""
    batch = [
        {"image": torch.zeros(3, 4, 4), "label": 1, "sample_id": 10},
        {"image": torch.ones(3, 4, 4), "label": 2, "sample_id": 20},
    ]
    collated = collate_image_batch(batch)
    assert collated["sample_id"].dtype == torch.long
    assert collated["sample_id"].tolist() == [10, 20]
