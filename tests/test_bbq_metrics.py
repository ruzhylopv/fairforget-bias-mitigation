import pytest

from evaluation.bbq.metrics import (
    compute_accuracy,
    compute_bbq_metrics,
    compute_condition_metrics,
)


def test_compute_accuracy():
    assert compute_accuracy([], []) == 0.0
    assert compute_accuracy([0, 1, 2], [0, 1, 2]) == 1.0
    assert compute_accuracy([0, 1, 0], [0, 1, 2]) == pytest.approx(2 / 3)
    assert compute_accuracy([1, 2], [0, 0]) == 0.0


def test_condition_metrics_empty():
    res = compute_condition_metrics([])
    assert res["total"] == 0
    assert res["accuracy"] == 0.0
    assert res["s_dis"] == 0.0


def test_condition_metrics_disambiguated():
    records = [
        {"pred_label": 0, "label": 0, "target_loc": 0, "unknown_loc": 1},
        {"pred_label": 0, "label": 2, "target_loc": 0, "unknown_loc": 1},
        {"pred_label": 2, "label": 2, "target_loc": 0, "unknown_loc": 1},
        {"pred_label": 1, "label": 0, "target_loc": 0, "unknown_loc": 1},
    ]
    res = compute_condition_metrics(records)
    assert res["total"] == 4
    assert res["correct"] == 2
    assert res["accuracy"] == 0.5
    assert res["n_unknown_outputs"] == 1
    assert res["n_non_unknown_outputs"] == 3
    assert res["n_biased_ans"] == 2
    assert res["s_dis"] == pytest.approx(1 / 3)


def test_bbq_metrics_full():
    ambig = [
        {
            "context_condition": "ambig",
            "pred_label": 1,
            "label": 1,
            "target_loc": 0,
            "unknown_loc": 1,
            "category": "Gender",
        },
        {
            "context_condition": "ambig",
            "pred_label": 1,
            "label": 1,
            "target_loc": 0,
            "unknown_loc": 1,
            "category": "Gender",
        },
        {
            "context_condition": "ambig",
            "pred_label": 0,
            "label": 1,
            "target_loc": 0,
            "unknown_loc": 1,
            "category": "Gender",
        },
        {
            "context_condition": "ambig",
            "pred_label": 2,
            "label": 1,
            "target_loc": 0,
            "unknown_loc": 1,
            "category": "Gender",
        },
    ]

    disambig = [
        {
            "context_condition": "disambig",
            "pred_label": 0,
            "label": 0,
            "target_loc": 0,
            "unknown_loc": 1,
            "category": "Gender",
        },
        {
            "context_condition": "disambig",
            "pred_label": 0,
            "label": 2,
            "target_loc": 0,
            "unknown_loc": 1,
            "category": "Gender",
        },
    ]

    metrics = compute_bbq_metrics(ambig + disambig)

    assert metrics["total_examples"] == 6
    assert metrics["overall_accuracy"] == pytest.approx(3 / 6)
    assert metrics["ambig_accuracy"] == 0.5
    assert metrics["disambig_accuracy"] == 0.5
    assert metrics["s_dis"] == 1.0
    assert metrics["s_amb"] == 0.0
    assert metrics["s_dis_pct"] == 100.0
    assert "Gender" in metrics["categories"]


def test_bbq_metrics_perfect_unbiased():
    records = [
        {
            "context_condition": "ambig",
            "pred_label": 1,
            "label": 1,
            "target_loc": 0,
            "unknown_loc": 1,
            "category": "test",
        },
        {
            "context_condition": "disambig",
            "pred_label": 0,
            "label": 0,
            "target_loc": 0,
            "unknown_loc": 1,
            "category": "test",
        },
    ]
    metrics = compute_bbq_metrics(records)
    assert metrics["ambig_accuracy"] == 1.0
    assert metrics["s_amb"] == 0.0
