import csv
from argparse import Namespace

import pytest
import torch

from rmu import unlearn


def test_loss_log_writes_header_and_rows_immediately(tmp_path):
    path = tmp_path / "losses.csv"
    log = unlearn.LossLog(path)

    log.write({"step": 0, "loss": 2.5})
    # Readable before close(): the job can be watched while it runs.
    assert path.read_text().splitlines() == ["step,loss", "0,2.5"]

    log.write({"step": 1, "loss": 1.5})
    log.close()
    assert list(csv.DictReader(path.open())) == [
        {"step": "0", "loss": "2.5"},
        {"step": "1", "loss": "1.5"},
    ]


def args(**overrides):
    defaults = dict(
        output_dir=None,
        model_name_or_path="lapa-12b-instruct",
        alpha=[100.0],
        layer_id=7,
        loss_positions="words",
        beta=None,
    )
    return Namespace(**{**defaults, **overrides})


def test_output_path_has_no_brackets_in_alpha(monkeypatch, tmp_path):
    monkeypatch.setattr(unlearn, "run_dir", lambda: tmp_path)

    assert (
        unlearn.output_path(args(), 257).name
        == "lapa-12b-instruct_alpha-100_batches-257_layer-7_positions-words"
    )
    assert unlearn.output_path(args(alpha=[100.0, 1.5]), 3).name.startswith(
        "lapa-12b-instruct_alpha-100,1.5_"
    )
    assert unlearn.output_path(args(beta=1.0), 257).name.endswith("_layer-7_beta-1_positions-words")


def test_output_dir_overrides_default(tmp_path):
    assert unlearn.output_path(args(output_dir=str(tmp_path / "x")), 1) == tmp_path / "x"


def test_change_stats_separates_shrinking_from_rotation():
    frozen = torch.tensor([[[3.0, 4.0], [1.0, 0.0]]])
    direction = torch.tensor([[[0.0, 1.0]]])
    words = torch.tensor([[True, False]])

    # Halving every activation: shorter, same direction.
    stats = unlearn.change_stats("forget", 0.5 * frozen, frozen, direction, words)

    assert stats["forget_norm_ratio"] == pytest.approx(0.5)
    assert stats["forget_cos_u"] == pytest.approx(0.8)


def left_padded_batch():
    # [pad, BOS, w, w] and [BOS, w, w, w], as the tokenizer returns them.
    return {
        "input_ids": torch.tensor([[0, 2, 10, 11], [2, 12, 13, 14]]),
        "attention_mask": torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]]),
        "special_tokens_mask": torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]]),
    }


def test_words_mode_keeps_only_real_tokens():
    inputs = left_padded_batch()

    keep, words = unlearn.loss_positions(inputs, "words")

    assert keep.tolist() == [[False, False, True, True], [False, True, True, True]]
    assert torch.equal(keep, words)
    assert "special_tokens_mask" not in inputs  # the model must not receive it


def test_all_mode_keeps_every_position_but_still_reports_words():
    keep, words = unlearn.loss_positions(left_padded_batch(), "all")

    assert keep.all()
    assert words.tolist() == [[False, False, True, True], [False, True, True, True]]


def test_masked_mse_with_everything_kept_equals_mse_loss():
    activations = torch.randn(2, 4, 8)
    target = torch.randn(1, 1, 8)
    keep = torch.ones(2, 4, dtype=torch.bool)

    expected = torch.nn.functional.mse_loss(activations, target.expand_as(activations))
    assert unlearn.masked_mse(activations, target, keep) == pytest.approx(expected.item())


def test_masked_mse_ignores_a_huge_masked_position():
    # The BOS problem in miniature: one enormous position must not affect the loss.
    activations = torch.ones(1, 3, 4)
    activations[0, 0] = 80_000.0
    target = torch.zeros(1, 1, 4)
    keep = torch.tensor([[False, True, True]])

    assert unlearn.masked_mse(activations, target, keep).item() == pytest.approx(1.0)


def test_masked_mse_is_float32_for_bf16_inputs():
    activations = torch.ones(1, 2, 4, dtype=torch.bfloat16)
    keep = torch.ones(1, 2, dtype=torch.bool)

    assert unlearn.masked_mse(activations, activations, keep).dtype == torch.float32


def test_loss_plot_is_saved_next_to_the_csv(tmp_path):
    csv_path = tmp_path / "losses.csv"
    log = unlearn.LossLog(csv_path)
    for step in range(3):
        log.write(
            {"step": step, "loss": 3.0 - step, "unlearn_loss": 3.0 - step, "retain_loss": 0.1}
        )
    log.close()

    png_path = unlearn.save_loss_plot(csv_path, title="test")

    assert png_path == tmp_path / "losses.png"
    assert png_path.stat().st_size > 0


def test_loss_plot_failure_does_not_raise(tmp_path):
    assert unlearn.save_loss_plot(tmp_path / "missing.csv", title="test") is None
