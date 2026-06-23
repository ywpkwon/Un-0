from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from metrics import _class_balanced_ids, _shard_for_rank, compute_fid


def test_class_balanced_ids_divisible_case() -> None:
    """With num_samples % num_classes == 0, every class has equal count."""
    ids = _class_balanced_ids(num_samples=100, num_classes=10, device=torch.device("cpu"))
    assert ids.shape == (100,)
    assert set(ids.tolist()) == set(range(10))
    counts = torch.bincount(ids, minlength=10)
    assert counts.tolist() == [10] * 10


def test_class_balanced_ids_non_divisible_pads_with_random_remainder() -> None:
    """Non-divisible counts still return exactly num_samples ids in [0, num_classes)."""
    ids = _class_balanced_ids(num_samples=103, num_classes=10, device=torch.device("cpu"))
    assert ids.shape == (103,)
    assert ids.min().item() >= 0
    assert ids.max().item() < 10


def test_compute_fid_cifar_path_uses_named_stats_and_balanced_ids() -> None:
    """Default (CIFAR) path: balanced ids under the current seed, scored against
    the named cifar10/train clean-fid stats at 32x32. This is the no-regression
    guard for the validated FID-11.14 path: same labels in, same clean-fid call."""
    model = MagicMock()
    fake_fid = MagicMock()
    fake_fid.compute_fid.return_value = 11.14

    torch.manual_seed(0)
    expected_ids = _class_balanced_ids(100, 10, torch.device("cpu"))

    with (
        patch.dict("sys.modules", {"cleanfid": MagicMock(fid=fake_fid)}),
        patch("metrics._dump_samples") as dump,
    ):
        torch.manual_seed(0)
        value = compute_fid(
            model,
            num_samples=100,
            num_classes=10,
            batch_size=64,
            device=torch.device("cpu"),
        )

    assert value == 11.14
    assert torch.equal(dump.call_args.kwargs["class_ids"], expected_ids)
    kw = fake_fid.compute_fid.call_args.kwargs
    assert kw["dataset_name"] == "cifar10"
    assert kw["dataset_split"] == "train"
    assert kw["dataset_res"] == 32
    assert kw["mode"] == "clean"
    fake_fid.make_custom_stats.assert_not_called()


def test_compute_fid_imagenet_path_builds_custom_stats_and_uses_given_ids() -> None:
    """ImageNet path: when real_image_dir is set, build custom stats from it and
    score with dataset_split='custom'; gen is conditioned on the given ids 1-to-1
    (not synthetic balanced ids), at image_size=64."""
    model = MagicMock()
    real_dir = "/tmp/real_val"  # noqa: S108  (only a path string; never touched — fs is mocked)
    gen_ids = torch.arange(10000) % 1000

    fake_fid = MagicMock()
    fake_fid.test_stats_exists.return_value = False
    fake_fid.compute_fid.return_value = 8.5

    with (
        patch.dict("sys.modules", {"cleanfid": MagicMock(fid=fake_fid)}),
        patch("metrics._dump_samples") as dump,
    ):
        value = compute_fid(
            model,
            num_samples=10000,
            num_classes=1000,
            batch_size=512,
            device=torch.device("cpu"),
            image_size=64,
            real_image_dir=real_dir,
            num_real_samples=50000,
            gen_class_ids=gen_ids,
        )

    assert value == 8.5
    fake_fid.make_custom_stats.assert_called_once()
    assert real_dir in fake_fid.make_custom_stats.call_args.args
    assert fake_fid.compute_fid.call_args.kwargs["dataset_split"] == "custom"
    assert torch.equal(dump.call_args.kwargs["class_ids"], gen_ids)
    assert dump.call_args.kwargs["image_size"] == 64


def test_compute_fid_imagenet_reuses_existing_custom_stats() -> None:
    """If the named custom stats already exist, do not rebuild them."""
    model = MagicMock()
    fake_fid = MagicMock()
    fake_fid.test_stats_exists.return_value = True
    fake_fid.compute_fid.return_value = 7.0

    with (
        patch.dict("sys.modules", {"cleanfid": MagicMock(fid=fake_fid)}),
        patch("metrics._dump_samples"),
    ):
        compute_fid(
            model,
            num_samples=4,
            num_classes=2,
            batch_size=2,
            device=torch.device("cpu"),
            image_size=64,
            real_image_dir="/tmp/real",
            num_real_samples=10,  # noqa: S108
            gen_class_ids=torch.tensor([0, 1, 0, 1]),
        )

    fake_fid.make_custom_stats.assert_not_called()


def test_dump_samples_uses_explicit_class_ids_in_order(tmp_path) -> None:
    """`_dump_samples` conditions sampling on the provided class_ids, in order."""
    from metrics import _dump_samples

    model = MagicMock()
    model.sample.return_value = torch.zeros(2, 3 * 64 * 64)
    class_ids = torch.tensor([7, 7, 42, 42])

    with patch("metrics.save_image"):
        _dump_samples(
            model,
            class_ids=class_ids,
            batch_size=2,
            device=torch.device("cpu"),
            image_dir=tmp_path,
            image_size=64,
        )

    sampled = torch.cat([c.args[0] for c in model.sample.call_args_list])
    assert torch.equal(sampled, class_ids)


def test_shard_for_rank_partitions_without_gaps_or_overlap() -> None:
    """Each rank's shard is disjoint and their concatenation in rank order is the
    full label set, so the sharded scored set equals a single-process dump."""
    labels = torch.arange(100, 110)  # 10 labels, world_size 3 -> 4, 3, 3
    shards = [_shard_for_rank(labels, rank=r, world_size=3) for r in range(3)]

    assert [s.numel() for s in shards] == [4, 3, 3]
    assert torch.equal(torch.cat(shards), labels)


def test_shard_for_rank_handles_fewer_labels_than_ranks() -> None:
    """Ranks beyond the label count get empty shards (no crash, union intact)."""
    labels = torch.arange(2)
    shards = [_shard_for_rank(labels, rank=r, world_size=4) for r in range(4)]

    assert [s.numel() for s in shards] == [1, 1, 0, 0]
    assert torch.equal(torch.cat(shards), labels)


def test_compute_fid_distributed_rank0_scores_others_nan(tmp_path) -> None:
    """Sharded mode: every rank generates its label shard into the shared dir
    with a rank-unique prefix and hits the barrier; rank 0 scores the combined
    dir, other ranks return nan without scoring."""
    import math

    gen_ids = torch.arange(12)
    fake_fid = MagicMock()
    fake_fid.test_stats_exists.return_value = False
    fake_fid.compute_fid.return_value = 9.0

    def run_rank(rank: int) -> tuple[float, MagicMock, MagicMock]:
        with (
            patch.dict("sys.modules", {"cleanfid": MagicMock(fid=fake_fid)}),
            patch("metrics._dump_samples") as dump,
            patch("metrics.dist.barrier") as barrier,
        ):
            value = compute_fid(
                MagicMock(),
                num_samples=12,
                num_classes=4,
                batch_size=4,
                device=torch.device("cpu"),
                image_size=64,
                real_image_dir=str(tmp_path / "real"),
                num_real_samples=10,
                gen_class_ids=gen_ids,
                image_dir=tmp_path / "shared",
                rank=rank,
                world_size=3,
            )
        return value, dump, barrier

    v0, dump0, barrier0 = run_rank(0)
    v1, dump1, _ = run_rank(1)

    # rank 0 scores; rank 1 returns nan and never calls clean-fid for itself
    assert v0 == 9.0
    assert math.isnan(v1)
    # each rank generated only its shard, with a rank-unique filename prefix
    assert dump0.call_args.kwargs["prefix"] == "gen_r0_"
    assert dump1.call_args.kwargs["prefix"] == "gen_r1_"
    # rank 0 and rank 1 generated their own contiguous shards of gen_ids
    assert torch.equal(
        dump0.call_args.kwargs["class_ids"],
        _shard_for_rank(gen_ids, rank=0, world_size=3),
    )
    assert torch.equal(
        dump1.call_args.kwargs["class_ids"],
        _shard_for_rank(gen_ids, rank=1, world_size=3),
    )
    # barrier is collective: called on every rank (pre-mkdir sync + post-gen)
    assert barrier0.call_count == 2


def test_compute_fid_distributed_requires_image_dir() -> None:
    """world_size > 1 without a shared image_dir is a usage error."""
    import pytest

    with patch.dict("sys.modules", {"cleanfid": MagicMock(fid=MagicMock())}):
        with pytest.raises(ValueError, match="image_dir"):
            compute_fid(
                MagicMock(),
                num_samples=4,
                num_classes=2,
                batch_size=2,
                device=torch.device("cpu"),
                gen_class_ids=torch.arange(4),
                rank=0,
                world_size=2,
            )
