from __future__ import annotations

import pytest
import torch

from un0.losses import PerClassQueue


def _ready_ids(mask: torch.Tensor) -> list[int]:
    return mask.nonzero(as_tuple=False).flatten().tolist()


def test_init_exposes_shape_and_starts_empty() -> None:
    """Fresh queues report their configured shape and no class is ready."""
    q = PerClassQueue(num_classes=4, queue_size=100, data_dim=8)
    assert q.num_classes == 4
    assert q.queue_size == 100
    assert q.data_dim == 8
    assert _ready_ids(q.ready_mask(1)) == []


def test_init_rejects_invalid_queue_size() -> None:
    """queue_size must be positive."""
    with pytest.raises(ValueError, match="queue_size"):
        PerClassQueue(num_classes=4, queue_size=0, data_dim=8)


def test_init_rejects_invalid_data_dim() -> None:
    """data_dim must be positive."""
    with pytest.raises(ValueError, match="data_dim"):
        PerClassQueue(num_classes=4, queue_size=10, data_dim=0)


def test_init_rejects_invalid_num_classes() -> None:
    """num_classes must be positive."""
    with pytest.raises(ValueError, match="num_classes"):
        PerClassQueue(num_classes=0, queue_size=10, data_dim=4)


def test_push_updates_ready_mask_per_class() -> None:
    """ready_mask reports only classes with enough samples for num_pos."""
    q = PerClassQueue(num_classes=3, queue_size=100, data_dim=4)
    q.push(torch.randn(5, 4), torch.tensor([0, 0, 1, 1, 1]))
    assert _ready_ids(q.ready_mask(1)) == [0, 1]
    assert _ready_ids(q.ready_mask(2)) == [0, 1]
    assert _ready_ids(q.ready_mask(3)) == [1]
    assert _ready_ids(q.ready_mask(4)) == []


def test_push_wraps_and_overwrites_oldest_entries() -> None:
    """After capacity is exceeded, draws only see the most recent rows."""
    q = PerClassQueue(num_classes=2, queue_size=3, data_dim=2)
    cls = torch.zeros(3, dtype=torch.long)
    q.push(torch.zeros(3, 2), cls)
    q.push(torch.ones(3, 2), cls)
    samples, labels, _ = q.draw(torch.tensor([0]), num_pos=3)
    assert torch.equal(labels, torch.zeros(3, dtype=torch.long))
    assert torch.allclose(samples, torch.ones(3, 2))


def test_push_handles_multiple_same_class_samples_per_batch() -> None:
    """Same-class samples in one push must land in distinct slots."""
    q = PerClassQueue(num_classes=2, queue_size=10, data_dim=2)
    x = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]])
    ids = torch.tensor([0, 0, 1, 0, 1])
    q.push(x, ids)
    samples0, _, _ = q.draw(torch.tensor([0]), num_pos=3)
    assert {tuple(r.tolist()) for r in samples0} == {
        (1.0, 1.0),
        (2.0, 2.0),
        (4.0, 4.0),
    }
    samples1, _, _ = q.draw(torch.tensor([1]), num_pos=2)
    assert {tuple(r.tolist()) for r in samples1} == {
        (3.0, 3.0),
        (5.0, 5.0),
    }


def test_push_empty_batch_is_noop() -> None:
    """Pushing zero samples does not change state."""
    q = PerClassQueue(num_classes=2, queue_size=5, data_dim=3)
    q.push(torch.zeros(0, 3), torch.zeros(0, dtype=torch.long))
    assert _ready_ids(q.ready_mask(1)) == []


def test_push_rejects_wrong_rank_input() -> None:
    """Samples must be 2-D."""
    q = PerClassQueue(num_classes=2, queue_size=5, data_dim=3)
    with pytest.raises(ValueError, match="shape"):
        q.push(torch.zeros(4), torch.zeros(4, dtype=torch.long))


def test_push_rejects_mismatched_class_ids_length() -> None:
    """class_ids length must equal batch size."""
    q = PerClassQueue(num_classes=2, queue_size=5, data_dim=3)
    with pytest.raises(ValueError, match="class_ids"):
        q.push(torch.zeros(4, 3), torch.zeros(3, dtype=torch.long))


def test_draw_samples_all_entries_without_replacement() -> None:
    """Drawing queue_size items returns every distinct pushed row."""
    q = PerClassQueue(num_classes=2, queue_size=10, data_dim=1)
    x = torch.arange(10, dtype=torch.float32).unsqueeze(1)
    q.push(x, torch.zeros(10, dtype=torch.long))
    samples, _, _ = q.draw(torch.tensor([0]), num_pos=10)
    assert set(samples.flatten().tolist()) == set(range(10))


def test_draw_returns_labels_aligned_with_class_ids() -> None:
    """Returned labels repeat each requested class id num_pos times."""
    q = PerClassQueue(num_classes=3, queue_size=5, data_dim=2)
    q.push(torch.randn(6, 2), torch.tensor([0, 0, 1, 1, 2, 2]))
    samples, labels, _ = q.draw(torch.tensor([0, 2]), num_pos=2)
    assert samples.shape == (4, 2)
    assert labels.tolist() == [0, 0, 2, 2]


def test_draw_returns_none_sample_ids_when_not_tracking() -> None:
    """Without tracking, draw returns None for sample_ids on both paths."""
    q = PerClassQueue(num_classes=2, queue_size=5, data_dim=3)
    q.push(torch.randn(4, 3), torch.tensor([0, 0, 1, 1]))
    _, _, sids = q.draw(torch.tensor([0]), num_pos=2)
    assert sids is None
    _, _, empty_sids = q.draw(torch.zeros(0, dtype=torch.long), num_pos=2)
    assert empty_sids is None


def test_draw_empty_class_ids_returns_empty_tensors() -> None:
    """Drawing zero classes yields zero-row samples and labels."""
    q = PerClassQueue(num_classes=2, queue_size=5, data_dim=3)
    q.push(torch.randn(4, 3), torch.tensor([0, 0, 1, 1]))
    samples, labels, _ = q.draw(torch.zeros(0, dtype=torch.long), num_pos=2)
    assert samples.shape == (0, 3)
    assert labels.shape == (0,)


def test_tracking_push_requires_sample_ids() -> None:
    """A tracking queue rejects a push that omits sample_ids."""
    q = PerClassQueue(
        num_classes=2,
        queue_size=5,
        data_dim=3,
        track_sample_ids=True,
    )
    with pytest.raises(ValueError, match="sample_ids"):
        q.push(torch.zeros(4, 3), torch.zeros(4, dtype=torch.long))


def test_tracking_push_rejects_mismatched_sample_ids_length() -> None:
    """sample_ids length must equal batch size when tracking."""
    q = PerClassQueue(
        num_classes=2,
        queue_size=5,
        data_dim=3,
        track_sample_ids=True,
    )
    with pytest.raises(ValueError, match="sample_ids"):
        q.push(
            torch.zeros(4, 3),
            torch.zeros(4, dtype=torch.long),
            torch.zeros(3, dtype=torch.long),
        )


def test_tracking_draw_returns_sample_ids_aligned_with_samples() -> None:
    """Drawn sample_ids correspond to the same rows as drawn samples."""
    q = PerClassQueue(
        num_classes=2,
        queue_size=10,
        data_dim=1,
        track_sample_ids=True,
    )
    x = torch.arange(10, dtype=torch.float32).unsqueeze(1)
    # Encode the row index in both the value (float i) and the id (100 + i)
    # so each draw can be checked for self-consistency.
    q.push(x, torch.zeros(10, dtype=torch.long), torch.arange(100, 110))
    samples, _, drawn_sids = q.draw(torch.tensor([0]), num_pos=10)
    for value, sid in zip(samples.flatten().tolist(), drawn_sids.tolist()):
        assert int(value) + 100 == int(sid)


def test_tracking_draw_overwrites_oldest_sample_ids() -> None:
    """Wrapped draws return the sample_ids of the most recent rows."""
    q = PerClassQueue(
        num_classes=2,
        queue_size=3,
        data_dim=2,
        track_sample_ids=True,
    )
    cls = torch.zeros(3, dtype=torch.long)
    q.push(torch.zeros(3, 2), cls, torch.tensor([0, 1, 2]))
    q.push(torch.ones(3, 2), cls, torch.tensor([10, 11, 12]))
    _, _, sids = q.draw(torch.tensor([0]), num_pos=3)
    assert set(sids.tolist()) == {10, 11, 12}


def test_tracking_draw_empty_class_ids_returns_empty_sample_ids() -> None:
    """An empty tracked draw still returns a zero-row sample_ids tensor."""
    q = PerClassQueue(
        num_classes=2,
        queue_size=5,
        data_dim=3,
        track_sample_ids=True,
    )
    q.push(
        torch.randn(4, 3),
        torch.tensor([0, 0, 1, 1]),
        torch.arange(4, dtype=torch.long),
    )
    _, _, sids = q.draw(torch.zeros(0, dtype=torch.long), num_pos=2)
    assert sids is not None
    assert sids.shape == (0,)


def test_queue_on_cuda_matches_input_device() -> None:
    """CUDA queue: push/draw stays on device and returns device tensors."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    q = PerClassQueue(
        num_classes=3,
        queue_size=10,
        data_dim=4,
        device="cuda",
        track_sample_ids=True,
    )
    q.push(
        torch.randn(6, 4, device="cuda"),
        torch.tensor([0, 1, 2, 0, 1, 2], device="cuda"),
        torch.arange(6, dtype=torch.long, device="cuda"),
    )
    mask = q.ready_mask(2)
    assert mask.device.type == "cuda"
    samples, labels, sids = q.draw(
        torch.tensor([0, 1, 2], device="cuda"),
        num_pos=2,
    )
    assert samples.device.type == "cuda"
    assert labels.device.type == "cuda"
    assert sids.device.type == "cuda"
