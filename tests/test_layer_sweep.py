import csv

import pytest
import torch
from transformers import AutoModelForCausalLM, LlamaConfig

from rmu import layer_sweep
from rmu.utils import get_params


def row(k, cos_u, loss=1.0, frozen=1.0, retain_k=1.0, retain_cos_u=0.0):
    return {
        "unlearn_loss": loss,
        "unlearn_loss_frozen": frozen,
        "forget_norm_ratio": k,
        "forget_cos_u": cos_u,
        "retain_norm_ratio": retain_k,
        "retain_cos_u": retain_cos_u,
    }


def test_pure_shrinkage_has_no_rotation_term():
    terms = layer_sweep.decompose([row(0.8, 0.0)], "forget")

    assert terms["forget_shrinkage"] == pytest.approx(0.18)
    assert terms["forget_rotation"] == pytest.approx(0.0)


def test_rotation_term_is_k_times_cosine():
    terms = layer_sweep.decompose([row(1.0, 0.3), row(0.5, 0.2)], "forget")

    assert terms["forget_k"] == pytest.approx(0.75)
    assert terms["forget_shrinkage"] == pytest.approx((0 + 0.375) / 2)
    assert terms["forget_rotation"] == pytest.approx((0.3 + 0.1) / 2)


def test_summary_uses_last_steps_and_measures_drop():
    rows = [row(0.0, 0.0, loss=100.0)] * 5 + [row(0.8, 0.1, loss=0.75, frozen=1.0)] * 3

    summary = layer_sweep.summarize_layer(12, rows, last=3)

    assert summary["layer"] == 12
    assert summary["steps"] == 3
    assert summary["forget_drop"] == pytest.approx(0.25)
    assert summary["forget_shrinkage"] == pytest.approx(0.18)
    assert summary["forget_rotation"] == pytest.approx(0.08)
    assert summary["retain_shrinkage"] == pytest.approx(0.0)


def test_append_row_writes_header_once(tmp_path):
    path = tmp_path / "sweep.csv"

    layer_sweep.append_row(path, {"layer": 0, "drop": 0.1})
    layer_sweep.append_row(path, {"layer": 4, "drop": 0.2})

    assert list(csv.DictReader(path.open())) == [
        {"layer": "0", "drop": "0.1"},
        {"layer": "4", "drop": "0.2"},
    ]


def test_trained_layers_are_clamped_at_zero():
    assert layer_sweep.trained_layers(0) == [0]
    assert layer_sweep.trained_layers(1) == [0, 1]
    assert layer_sweep.trained_layers(32) == [30, 31, 32]


def tiny_llama():
    torch.manual_seed(0)
    config = LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=64,
    )
    return AutoModelForCausalLM.from_config(config)


def test_restore_params_copies_only_the_given_weights_back():
    frozen = tiny_llama()
    updated = tiny_llama()
    names = ["mlp.down_proj.weight"]
    with torch.no_grad():
        for param in get_params(updated, [0, 1, 2], names):
            param.add_(1.0)

    layer_sweep.restore_params(updated, frozen, [1, 2], names)

    restored = get_params(updated, [0, 1, 2], names)
    original = get_params(frozen, [0, 1, 2], names)
    assert not torch.equal(restored[0], original[0])
    assert torch.equal(restored[1], original[1])
    assert torch.equal(restored[2], original[2])


def test_sweep_defaults_are_the_agreed_hyperparameters():
    args = layer_sweep.get_sweep_args(["--layers", "0,47"])

    assert args.layers == [0, 47]
    assert args.split and args.verbose
    assert (args.loss_positions, args.beta, args.lr, args.alpha, args.max_num_batches) == (
        "words",
        1.0,
        5e-4,
        [5.0],
        257,
    )
