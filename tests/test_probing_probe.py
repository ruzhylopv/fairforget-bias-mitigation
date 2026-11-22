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


def test_workers_are_capped_by_memory_not_only_cpus(monkeypatch):
    """lapa/race on a JUPITER node: 288 CPUs, 490 null fits of 7,874 x 3,840 float64 layers.
    One worker per CPU ran out of memory; the cap keeps the total within 70% of what's free."""
    from probing import probe

    monkeypatch.setattr(probe, "usable_cpus", lambda: 288)
    shape, memory = (7874, 3840), 450e9
    n, why = probe.n_workers(490, -1, shape, memory=memory)
    per_worker = probe.FIT_COPIES * shape[0] * shape[1] * 8 + probe.FIT_OVERHEAD
    assert 40 < n < 288 and n * per_worker <= probe.MEMORY_SHARE * memory
    assert why.startswith("limit: memory") and "450 GB free" in why
    # a small layer gets one worker per CPU, and few tasks need few workers
    n, why = probe.n_workers(490, -1, (500, 3840), memory=memory)
    assert n == 288 and why.startswith("limit: CPUs")
    assert probe.n_workers(14, -1, shape, memory=memory)[0] == 14
    # --n_jobs wins; at least one worker even with almost no memory
    assert probe.n_workers(490, 8, shape, memory=memory) == (8, "--n_jobs")
    assert probe.n_workers(490, -1, shape, memory=1e6)[0] == 1


def _fake_node(root, cgroup):
    """A GH200-like node: 2 CPU NUMA nodes (100 GB free each) and 2 CPU-less HBM nodes
    (90 GB free each); /proc/meminfo counts all four."""
    gb = 2**30 // 1024  # kB per GiB
    for i, cpus in enumerate(["0-71", "72-143", "", ""]):
        n = root / f"sys/devices/system/node/node{i}"
        n.mkdir(parents=True)
        (n / "cpulist").write_text(cpus + "\n")
        free = 80 if cpus else 90
        (n / "meminfo").write_text(
            f"Node {i} MemTotal: {120 * gb} kB\nNode {i} MemFree: {free * gb} kB\n"
            f"Node {i} FilePages: {20 * gb if cpus else 0} kB\n"
        )
    (root / "proc/self").mkdir(parents=True)
    (root / "proc/meminfo").write_text(f"MemAvailable: {380 * gb} kB\n")
    (root / "proc/self/cgroup").write_text(cgroup)


def test_memory_counts_only_numa_nodes_with_cpus_and_slurm_limits(tmp_path, monkeypatch):
    from probing import probe

    monkeypatch.delenv("SLURM_MEM_PER_NODE", raising=False)
    gib = 2**30
    root = tmp_path / "v2"
    _fake_node(root, "0::/slurm/job1\n")
    cg = root / "sys/fs/cgroup/slurm/job1"
    cg.mkdir(parents=True)
    (cg / "memory.max").write_text("max\n")
    (cg / "memory.current").write_text("0\n")
    src = probe.memory_sources(root)
    assert src == {"cpu_numa": 200 * gib}  # not the 380 GiB of MemAvailable; no cgroup limit
    assert probe.available_memory(root) == (200 * gib, "cpu_numa")

    (cg / "memory.max").write_text(str(150 * gib))
    (cg / "memory.current").write_text(str(10 * gib))
    assert probe.available_memory(root) == (140 * gib, "cgroup")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", str(100 * 1024))  # MiB
    assert probe.available_memory(root) == (100 * gib, "slurm")

    monkeypatch.delenv("SLURM_MEM_PER_NODE")
    root = tmp_path / "v1"
    _fake_node(root, "4:cpu,cpuacct:/slurm/job1\n9:memory:/slurm/job1\n")
    cg = root / "sys/fs/cgroup/memory/slurm/job1"
    cg.mkdir(parents=True)
    (cg / "memory.limit_in_bytes").write_text(str(120 * gib))
    (cg / "memory.usage_in_bytes").write_text(str(20 * gib))
    assert probe.available_memory(root) == (100 * gib, "cgroup")

    # cgroup v2 as Slurm sets it: the limit is on the job's cgroup, the task's own says "max"
    root = tmp_path / "v2-nested"
    _fake_node(root, "0::/system.slice/slurmstepd.scope/job_7/step_0/user/task_0\n")
    job = root / "sys/fs/cgroup/system.slice/slurmstepd.scope/job_7"
    task = job / "step_0/user/task_0"
    task.mkdir(parents=True)
    for d, lim, cur in ((task, "max", 5), (job, str(90 * gib), 30 * gib)):
        (d / "memory.max").write_text(lim + "\n")
        (d / "memory.current").write_text(f"{cur}\n")
    assert probe.available_memory(root) == (60 * gib, "cgroup")

    # no NUMA information: MemAvailable
    assert probe.memory_sources(tmp_path / "nothing") == {}
