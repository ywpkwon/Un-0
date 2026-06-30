from __future__ import annotations

import torch
import pytest

import un0.model as model_mod
from un0.model import (
    ConditionalImplicitKuramotoGenerator,
    build_cifar10_model,
    build_imagenet64_model,
)


def _save_ckpt(tmp_path, fname, state_dict, config):
    path = tmp_path / fname
    torch.save({"model": state_dict, "config": config}, path)
    return path


@pytest.fixture
def patch_download(tmp_path, monkeypatch):
    """Route hf_hub_download to a dict of {filename: local path}."""
    registry: dict[str, str] = {}

    def fake_download(repo_id, filename, **kwargs):
        if filename not in registry:
            raise AssertionError(f"unexpected download: {filename}")
        return registry[filename]

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    return registry


def test_from_pretrained_cifar10_round_trip(tmp_path, patch_download):
    # Tiny cifar10 with a NON-default relativization, to prove arch comes from config.
    ref = build_cifar10_model(
        n_oscillators=16, n_conditional_oscillators=8, relativization="mean_relative", num_steps=2
    )
    cfg = {
        "n_oscillators": 16,
        "n_conditional_oscillators": 8,
        "class_dropout_prob": 0.1,
        "num_steps": 2,
        "decoder_in_channels": None,
        "parameterization": "standard",
        "relativization": "mean_relative",
        "encoding": "sin_cos",
        "solver": "rk4",
    }
    path = _save_ckpt(tmp_path, "cifar10_n16.pt", ref.state_dict(), cfg)
    patch_download["cifar10_n1024.pt"] = str(path)

    model = ConditionalImplicitKuramotoGenerator.from_pretrained("cifar10/n1024", device="cpu")
    # arch-from-config took effect (not the builder default "ref_oscillator")
    assert model.readout.relativization == "mean_relative"
    out = model.sample(torch.tensor([0, 1, 2]))
    assert out.shape == (3, 3 * 32 * 32)
    # weights actually loaded (equal to the saved reference)
    ref_sd, got_sd = ref.state_dict(), model.state_dict()
    assert torch.equal(
        ref_sd["decoder._orig_mod.to_output.bias"], got_sd["decoder._orig_mod.to_output.bias"]
    )


def test_from_pretrained_imagenet64_round_trip(tmp_path, patch_download):
    ref = build_imagenet64_model(n_oscillators=16, n_conditional_oscillators=1, num_steps=2)
    cfg = {
        "n_oscillators": 16,
        "n_conditional_oscillators": 1,
        "class_dropout_prob": 0.1,
        "num_steps": 2,
        "decoder_in_channels": None,
        "parameterization": "mup",
        "relativization": "ref_oscillator",
    }
    path = _save_ckpt(tmp_path, "in64.pt", ref.state_dict(), cfg)
    patch_download["imagenet64_n16384.pt"] = str(path)

    model = ConditionalImplicitKuramotoGenerator.from_pretrained("imagenet64/n16384", device="cpu")
    out = model.sample(torch.tensor([0, 1]))
    assert out.shape == (2, 3 * 64 * 64)


def test_build_from_config_override_and_fallback():
    # Present arch keys override the builder default; absent keys fall back to it;
    # non-builder training keys (integration_method, batch_size) are ignored, not errors.
    from un0.model import (
        ConditionalFixedAnchorLoheDynamics,
        build_cifar10_model,
        build_from_config,
        build_imagenet64_model,
    )

    # cifar10: a present non-default relativization wins; absent encoding/solver fall back.
    cifar = build_from_config(
        build_cifar10_model,
        {
            "n_oscillators": 16,
            "relativization": "mean_relative",
            "num_steps": 2,
            "batch_size": 2048,
        },  # batch_size is a junk key
    )
    assert cifar.readout.relativization == "mean_relative"  # from config
    assert cifar.dynamics.n == 16  # from config

    lohe = build_from_config(
        build_cifar10_model,
        {
            "dynamics": "lohe_fixed",
            "n_oscillators": 16,
            "n_conditional_oscillators": 4,
            "lohe_dim": 2,
            "num_steps": 1,
        },
    )
    assert isinstance(lohe.dynamics, ConditionalFixedAnchorLoheDynamics)
    assert lohe.decoder.feature_dim == 32

    # imagenet64 with a near-arch-less config (the n16384 pre-patch schema):
    # relativization/parameterization fall back to builder defaults; the junk
    # integration_method key is dropped without error. n_oscillators is set small
    # only to keep the test cheap (the fallback under test is relativization).
    in64 = build_from_config(
        build_imagenet64_model,
        {"n_oscillators": 16, "num_steps": 2, "integration_method": "euler"},
    )
    assert in64.readout.relativization == "ref_oscillator"  # builder default (absent)


def test_from_pretrained_unknown_name():
    with pytest.raises(ValueError, match="Unknown pretrained name"):
        ConditionalImplicitKuramotoGenerator.from_pretrained("cifar10/nope")


def test_from_pretrained_uses_weights_only(tmp_path, patch_download, monkeypatch):
    ref = build_cifar10_model(n_oscillators=16, num_steps=2)
    cfg = {
        "n_oscillators": 16,
        "n_conditional_oscillators": 8,
        "class_dropout_prob": 0.1,
        "num_steps": 2,
        "decoder_in_channels": None,
        "parameterization": "standard",
        "relativization": "ref_oscillator",
        "encoding": "sin_cos",
        "solver": "rk4",
    }
    path = _save_ckpt(tmp_path, "c.pt", ref.state_dict(), cfg)
    patch_download["cifar10_n1024.pt"] = str(path)

    seen = {}
    real_load = torch.load

    def spy_load(p, *a, **k):
        seen["weights_only"] = k.get("weights_only")
        return real_load(p, *a, **k)

    monkeypatch.setattr(torch, "load", spy_load)
    ConditionalImplicitKuramotoGenerator.from_pretrained("cifar10/n1024", device="cpu")
    assert seen["weights_only"] is True


def test_inference_parser_requires_one_source():
    import un0.inference as inference

    parser = inference.build_parser()
    # neither source -> error
    with pytest.raises(SystemExit):
        parser.parse_args(["--output", "x.png"])


def test_inference_parser_pretrained_choices():
    import un0.inference as inference
    from un0.model import PRETRAINED_NAMES

    parser = inference.build_parser()
    args = parser.parse_args(["--pretrained", "imagenet64/n16384"])
    assert args.pretrained == "imagenet64/n16384"
    # invalid name rejected by argparse choices
    with pytest.raises(SystemExit):
        parser.parse_args(["--pretrained", "bogus/n1"])
    assert "imagenet64/n16384" in PRETRAINED_NAMES
