"""probing.controls (WP6) on tiny arrays (CPU)."""

import numpy as np
import pandas as pd
import pytest
from test_probing_probe import planted, tiny_meta

from probing import controls, probe
from probing.io import save_layer


def test_flip_keeps_labels_balanced_and_paired():
    y = tiny_meta()["S"].to_numpy()
    for seed in range(5):
        f = controls.flip_within_pairs(y, seed)
        assert f.sum() == len(y) // 2
        assert (f[0::2] + f[1::2] == 1).all()  # one stereotype per pair
        assert (f[0::2] != y[0::2]).sum() == len(y) // 4  # exactly half the pairs swapped
    np.testing.assert_array_equal(
        controls.flip_within_pairs(y, 1), controls.flip_within_pairs(y, 1)
    )
    assert not np.array_equal(controls.flip_within_pairs(y, 1), controls.flip_within_pairs(y, 2))
    with pytest.raises(ValueError):
        controls.flip_within_pairs(np.array([1, 1, 0, 0]), 0)


def test_pairwise_accuracy_follows_flipped_labels():
    z = np.array([2.0, 1.0, 2.0, 1.0])
    assert probe.pairwise_accuracy(z) == 1.0
    assert probe.pairwise_accuracy(z, y=np.array([1, 0, 0, 1])) == 0.5
    s = probe.scores(np.array([0, 1, 0, 1]), z, 1 / (1 + np.exp(-z)), np.zeros(4, bool))
    assert s["acc_pair"] == 0.0


def boot_inputs(seed=0):
    meta = tiny_meta()
    y, groups, flag = meta["S"].to_numpy(), meta["target"].to_numpy(), probe.pair_flag(meta)
    rng = np.random.default_rng(seed)
    z_probe = (2 * y - 1) * 1.0 + rng.normal(0, 1, len(y))
    z_base = (2 * y - 1) * 0.3 + rng.normal(0, 1, len(y))
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    preds = {("first_diff", 3): (z_probe, sig(z_probe)), controls.BASELINE: (z_base, sig(z_base))}
    return y, groups, flag, preds


def test_bootstrap_estimate_matches_scores():
    y, groups, flag, preds = boot_inputs()
    ci = controls.bootstrap_ci(y, groups, flag, preds, R=200).set_index(["position", "metric"])
    for (pos, _), (z, p) in preds.items():
        s = probe.scores(y, z, p, flag)
        for m in controls.CI_METRICS:
            assert ci.loc[(pos, m), "estimate"] == pytest.approx(s[m])
            assert ci.loc[(pos, m), "low"] <= ci.loc[(pos, m), "high"]


def test_bootstrap_uses_the_same_draws_for_probe_and_baseline():
    y, groups, flag, preds = boot_inputs()
    same = {
        ("first_diff", 3): preds[controls.BASELINE],
        controls.BASELINE: preds[controls.BASELINE],
    }
    ci = controls.bootstrap_ci(y, groups, flag, same, R=300)
    probe_rows = ci[ci["position"] == "first_diff"]
    # Identical predictions on identical draws: the paired gap is exactly 0 in every resample,
    # while the CI of each is not a point.
    assert (probe_rows["gap_low"] == 0).all() and (probe_rows["gap_high"] == 0).all()
    assert (probe_rows["high"] - probe_rows["low"] > 0).all()


def test_bootstrap_resample_equals_explicit_row_repeats():
    y, groups, flag, preds = boot_inputs()
    targets, codes, counts = controls.target_draws(groups, R=5, seed=3)
    z, p = preds[("first_diff", 3)]
    res = controls.resampled(y, z, p, flag, codes, len(targets), counts)
    for r in range(5):
        # Every row of each drawn target, once per draw, pairs kept as (2i, 2i+1).
        rows = np.concatenate(
            [np.tile(np.flatnonzero(codes == t), c) for t, c in enumerate(counts[r]) if c]
        )
        s = probe.scores(y[rows], z[rows], p[rows], flag[rows])
        for m in ("acc", "acc_pair", "logloss", "acc_pair_tafter"):
            assert res[m][1][r] == pytest.approx(s[m]), m


def test_target_draws_counts_sum_to_number_of_targets():
    groups = tiny_meta()["target"].to_numpy()
    targets, _, counts = controls.target_draws(groups, R=50, seed=0)
    assert counts.shape == (50, len(targets)) and (counts.sum(axis=1) == len(targets)).all()


def test_drop_pairs_drops_both_rows_of_a_pair():
    meta = pd.DataFrame({"word": ["car", "mainicure", "nice", "mean", "Forebearance.", "x"]})
    assert controls.drop_pairs(meta).tolist() == [False, False, True, True, False, False]


def test_pvalues_counts_shuffles_at_least_as_good():
    real = pd.DataFrame(
        [
            {"position": "first_diff", "layer": 0, "acc": 0.52, "acc_pair": 0.9},
            {"position": "first_diff", "layer": 1, "acc": 0.60, "acc_pair": 0.9},
            {"position": controls.BASELINE[0], "layer": -1, "acc": 0.55, "acc_pair": 0.6},
        ]
    )
    null_acc = {0: [0.50, 0.53], 1: [0.49, 0.51], -1: [0.50, 0.50]}  # seed 1 max = 0.53
    null = pd.DataFrame(
        [
            {"seed": s, "position": pos, "layer": k, "acc": null_acc[k][s], "acc_pair": 0.5}
            for s in (0, 1)
            for pos, k in (("first_diff", 0), ("first_diff", 1), (controls.BASELINE[0], -1))
        ]
    )
    pv = controls.pvalues(real, null, metrics=("acc",)).set_index(["layer"])
    assert pv.loc[0, "p"] == pytest.approx(2 / 3)  # 0.53 >= 0.52
    assert pv.loc[0, "p_corr"] == pytest.approx(2 / 3)
    assert pv.loc[1, "p"] == pytest.approx(1 / 3) and pv.loc[1, "p_corr"] == pytest.approx(1 / 3)
    assert np.isnan(pv.loc[-1, "p_corr"]) and (pv["B"] == 2).all()


def test_run_null_is_at_chance_and_resumes(tmp_path):
    meta = tiny_meta()
    X = planted(meta, delta=3.0)  # strong real signal: the null must still sit near 0.5
    for k in (0, 1):
        save_layer(X, "m", "first_diff", k, root=tmp_path, dtype="float32")
    folds = probe.outer_folds(meta)
    ctx = {
        "model": "m",
        "root": tmp_path,
        "y": meta["S"].to_numpy(),
        "groups": meta["target"].to_numpy(),
        "flag": probe.pair_flag(meta),
        "ell": np.random.default_rng(0).normal(-10, 2, len(meta)),
        "folds": folds,
        "grid": (0.01, 1.0),
    }
    path = tmp_path / "null_scores.csv"
    controls.run_null(ctx, [0], ["first_diff"], [0], path, n_jobs=1)
    controls.run_null(ctx, [0, 1], ["first_diff"], [0, 1], path, n_jobs=1)  # resumes

    null = pd.read_csv(path)
    assert len(null) == 2 * 3 and not null.duplicated(["seed", "position", "layer"]).any()
    probes = null[null["position"] == "first_diff"]
    assert probes["acc"].between(0.3, 0.7).all() and probes["acc_pair"].between(0.25, 0.75).all()
    real = controls.fit_task(
        "m", "first_diff", 0, tmp_path, ctx["y"], ctx["groups"], folds, ctx["flag"], ctx["grid"]
    )
    assert real["acc_pair"] > 0.9
