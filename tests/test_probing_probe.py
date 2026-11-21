"""probing.probe on tiny arrays (CPU)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from probing import probe

N_TARGETS, PAIRS_PER_TARGET = 10, 6


def tiny_meta(seed=0):
    """Pairs in order (S=1 then S=0); each pair has one target; some pairs flagged."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(N_TARGETS):
        for i in range(PAIRS_PER_TARGET):
            after = bool(rng.random() < 0.3)
            for s in (1, 0):
                rows.append(
                    {"target": f"t{t}", "S": s, "target_after_word": after and s == 1, "ell": -5.0}
                )
    return pd.DataFrame(rows)


def planted(meta, dim=20, delta=3.0, seed=0):
    """Gaussian X with a signal along one direction, raw scales very different per dimension."""
    rng = np.random.default_rng(seed)
    y = meta["S"].to_numpy()
    X = rng.standard_normal((len(meta), dim))
    X[:, 0] += delta * (2 * y - 1)
    return X * np.logspace(-2, 5, dim)  # like lapa's 1e5 outlier dims


def test_outer_folds_never_share_a_target_and_keep_pairs_together():
    meta = tiny_meta()
    folds = probe.outer_folds(meta)

    assert set(folds) == set(range(probe.N_OUTER))
    per_target = meta.assign(fold=folds).groupby("target")["fold"].nunique()
    assert (per_target == 1).all()
    assert (folds[0::2] == folds[1::2]).all()


def test_pairwise_accuracy_counts_ties_as_half():
    z = [1.0, 0.0, 0.0, 1.0, 2.0, 2.0]  # right, wrong, tie
    assert probe.pairwise_accuracy(z) == pytest.approx(0.5)
    assert probe.pairwise_accuracy(z, mask=[1, 1, 0, 0, 1, 1]) == pytest.approx(0.75)
    assert np.isnan(probe.pairwise_accuracy(z, mask=[0] * 6))


def test_pair_flag_is_true_on_both_rows_if_either_row_is():
    meta = pd.DataFrame({"target_after_word": [True, False, False, False, False, True]})
    assert probe.pair_flag(meta).tolist() == [True, True, False, False, True, True]


def test_unscale_reproduces_decision_function_on_raw_x():
    meta = tiny_meta()
    X, y = planted(meta), meta["S"].to_numpy()
    pipe = probe.make_pipe(C=0.1).fit(X, y)

    w, b = probe.unscale(pipe)

    np.testing.assert_allclose(X @ w + b, pipe.decision_function(X), rtol=1e-8, atol=1e-8)
    # The scaled coef_ is not the raw direction.
    assert not np.allclose(X @ pipe[-1].coef_[0] + pipe[-1].intercept_[0], X @ w + b)


class RecordingGroupKFold(GroupKFold):
    """GroupKFold that records the groups it is given."""

    seen = []

    def split(self, X, y=None, groups=None):
        RecordingGroupKFold.seen.append(None if groups is None else np.asarray(groups))
        return super().split(X, y, groups)


def test_fit_layer_passes_training_groups_to_the_inner_cv():
    meta = tiny_meta()
    X, y, groups = planted(meta), meta["S"].to_numpy(), meta["target"].to_numpy()
    folds = probe.outer_folds(meta)
    RecordingGroupKFold.seen = []

    res = probe.fit_layer(X, y, groups, folds, grid=(0.01, 1.0), inner_cv=RecordingGroupKFold(4))

    assert len(RecordingGroupKFold.seen) == probe.N_OUTER
    for f, seen in enumerate(RecordingGroupKFold.seen):
        assert seen is not None
        assert set(seen) == set(groups[folds != f])  # training targets only
    assert not np.isnan(res["z"]).any() and set(res["C"]) == set(range(probe.N_OUTER))
    assert set(res["C"].values()) <= {0.01, 1.0}
    # Planted signal, graded on unseen targets: clearly above chance.
    s = probe.scores(y, res["z"], res["p"], probe.pair_flag(meta))
    assert s["acc"] > 0.8 and s["acc_pair"] > 0.9


def test_fit_layer_is_at_chance_without_signal():
    meta = tiny_meta()
    X = planted(meta, delta=0.0, seed=1)
    y, groups = meta["S"].to_numpy(), meta["target"].to_numpy()
    res = probe.fit_layer(X, y, groups, probe.outer_folds(meta), grid=(0.01, 1.0))
    assert 0.3 < probe.scores(y, res["z"], res["p"], probe.pair_flag(meta))["acc"] < 0.7


def test_baseline_pairwise_equals_share_of_pairs_with_higher_ell():
    meta = tiny_meta()
    rng = np.random.default_rng(3)
    ell = rng.normal(-10, 2, len(meta))
    ell[0::2] += 1.0  # stereotype more expected on average
    ell[5] = ell[4]  # one tie
    y, folds = meta["S"].to_numpy(), probe.outer_folds(meta)

    base = probe.fit_baseline(ell, y, folds)

    assert all(g > 0 for g in base["gamma"].values())
    assert probe.pairwise_accuracy(base["z"]) == pytest.approx(probe.pairwise_accuracy(ell))


def test_most_frequent_prefers_smaller_c_on_ties():
    assert probe.most_frequent([0.1, 0.1, 1.0]) == 0.1
    assert probe.most_frequent([1.0, 0.01, 1.0, 0.01]) == 0.01


def test_scores_split_by_subset():
    y = np.array([1, 0, 1, 0])
    z = np.array([1.0, -1.0, -1.0, 1.0])  # pair 0 right, pair 1 wrong
    p = 1 / (1 + np.exp(-z))
    flag = np.array([True, True, False, False])
    s = probe.scores(y, z, p, flag)
    assert s["acc"] == 0.5 and s["acc_tafter"] == 1.0 and s["acc_tfirst"] == 0.0
    assert s["acc_pair_tafter"] == 1.0 and s["acc_pair_tfirst"] == 0.0
    assert s["n_pairs_tafter"] == 1 and s["n_pairs_tfirst"] == 1


def test_parse_layers():
    assert probe.parse_layers("0-3,40") == [0, 1, 2, 3, 40]
    assert probe.parse_layers("20,21") == [20, 21]


def test_c_edge_names_which_end():
    grid = (1e-4, 1e-2, 1.0)
    assert probe.c_edge([1e-2, 1e-2], grid) == ""
    assert probe.c_edge([1e-4, 1e-2], grid) == "low"
    assert probe.c_edge([1.0], grid) == "high"
    assert probe.c_edge([1e-4, 1.0], grid) == "low+high"
