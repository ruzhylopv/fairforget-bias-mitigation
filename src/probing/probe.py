"""RQ2 probes: per layer and position, logistic regression P(S | activation), graded on unseen
targets, next to the predictability baseline P(S | ell). CPU only.

Cross-validation (docs/rq2_probing.md):

* outer: ``GroupKFold(5)`` by ``target``, computed once and saved as ``folds.csv``; the probe,
  the baseline and later the WP6 shuffles all use these folds;
* inner: ``GridSearchCV`` over C with ``GroupKFold(4)`` on the training targets only (written
  as an explicit loop, so the groups reach the inner split);
* the baseline has one feature and no inner search (C = 1e4).

Scores come from the stored held-out predictions: per-sentence accuracy (z > 0), pairwise
accuracy (rows 2i, 2i+1: correct if z[2i] > z[2i+1], ties count 1/2) and log-loss, overall and
split by ``target_after_word`` at pair level (true on either row).

Outputs (``probing.io.run_paths(model, bias)``)::

    {probes}/folds.csv                        row, fold, target
    {probes}/baseline_ell.csv                 row, fold, z, p, C
    {probes}/{position}/preds_layer_kk.csv    row, fold, z, p, C
    {probes}/{position}/probe_layer_kk.npz    w_raw, b_raw, mu, scale, C
    {probes}/run.json                         settings (the grid is reused by probing.controls)
    {results}/summary.csv                     one row per position x layer + baseline, with
                                              rel_depth = layer / n_layers (NaN on the baseline)
    {results}/run.json                        the same settings, as step "probe"

with ``{probes}`` = ``{rq2_root}/{bias}/{model}/probes`` and ``{results}`` =
``results/rq2/{bias}/{model}`` (in the repo, committed).

Usage:
    python -m probing.probe --model lapa-12b-instruct --bias gender \\
        --positions first_diff,last_subtok [--layers 0-48] [--root PATH] [--n_jobs -1]

``--layers`` defaults to every hidden state: 0 .. n_layers, read from ``{acts}/run.json``.
"""

import argparse
import datetime
import json
import os
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from probing.io import (
    POSITIONS,
    git_commit,
    load_layer,
    load_meta,
    n_decoder_layers,
    record_run,
    rel_depth,
    run_paths,
)

GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
BASELINE_C = 1e4
N_OUTER, N_INNER = 5, 4
# Inner selection by held-out log-loss: smoother than accuracy on small inner folds.
INNER_SCORING = "neg_log_loss"
SUBSETS = {"": None, "_tfirst": False, "_tafter": True}  # target_after_word at pair level
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "FLEXIBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


# --------------------------------------------------------------------------- building blocks


def make_pipe(C=1.0):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def outer_folds(meta: pd.DataFrame, n_splits: int = N_OUTER) -> np.ndarray:
    """Fold (0..n_splits-1) per row, grouped by target. Pairs share a target, so never split."""
    folds = np.full(len(meta), -1)
    groups = meta["target"].to_numpy()
    for f, (_, test) in enumerate(GroupKFold(n_splits).split(np.zeros(len(meta)), groups=groups)):
        folds[test] = f
    return folds


def pair_flag(meta: pd.DataFrame) -> np.ndarray:
    """Per row: target_after_word is true on either row of its pair."""
    taw = meta["target_after_word"].to_numpy().astype(bool)
    pair = taw[0::2] | taw[1::2]
    return np.repeat(pair, 2)


def pair_correct(z, y=None) -> np.ndarray:
    """Per pair (rows 2i, 2i+1): 1 if the row labelled 1 has the higher z, 0 if lower, 1/2 tie.

    Without ``y`` row 2i is the labelled-1 row (the real labels). With the WP6 null labels
    (swapped within some pairs) ``y`` says which row of each pair counts as the stereotype.
    """
    z = np.asarray(z, dtype=float)
    d = z[0::2] - z[1::2]
    if y is not None:
        y = np.asarray(y)
        if not (y[0::2] + y[1::2] == 1).all():
            raise ValueError("each pair needs one row with y=1 and one with y=0")
        d = np.where(y[0::2] == 1, d, -d)
    return (d > 0) + 0.5 * (d == 0)


def pairwise_accuracy(z, mask=None, y=None) -> float:
    """Share of pairs (rows 2i = stereotype, 2i+1 = anti) with z[2i] > z[2i+1]; ties count 1/2.

    ``mask`` (per row, constant within a pair) selects pairs; ``y`` see ``pair_correct``.
    """
    c = pair_correct(z, y)
    keep = np.ones(len(c), bool) if mask is None else np.asarray(mask, bool)[0::2]
    if not keep.any():
        return float("nan")
    return float(c[keep].mean())


def scores(y, z, p, flag) -> dict:
    """Per-sentence accuracy, pairwise accuracy and log-loss, overall and by subset."""
    y, z, p = np.asarray(y), np.asarray(z), np.asarray(p)
    out = {}
    for suffix, value in SUBSETS.items():
        m = np.ones(len(y), bool) if value is None else (flag == value)
        if not m.any():
            out |= {f"acc{suffix}": np.nan, f"acc_pair{suffix}": np.nan, f"logloss{suffix}": np.nan}
            continue
        out[f"acc{suffix}"] = float(((z[m] > 0).astype(int) == y[m]).mean())
        out[f"acc_pair{suffix}"] = pairwise_accuracy(z, m, y)
        out[f"logloss{suffix}"] = float(log_loss(y[m], p[m], labels=[0, 1]))
        out[f"n_pairs{suffix}"] = int(m.sum() // 2)
    return out


def fit_layer(X, y, groups, folds, grid=GRID, inner_cv=None, scoring=INNER_SCORING):
    """Nested grouped CV. Returns held-out ``z`` and ``p`` per row, the C chosen in each outer
    fold, and the number of convergence warnings.

    The outer loop is explicit so that ``groups`` reaches the inner ``GroupKFold``
    (``cross_val_score`` around ``GridSearchCV`` would drop them).
    """
    X = np.asarray(X, dtype=np.float64)
    y, groups, folds = np.asarray(y), np.asarray(groups), np.asarray(folds)
    z, p = np.full(len(y), np.nan), np.full(len(y), np.nan)
    chosen, n_warn = {}, 0
    for f in np.unique(folds):
        train, test = folds != f, folds == f
        search = GridSearchCV(
            make_pipe(),
            {"logisticregression__C": list(grid)},
            cv=inner_cv or GroupKFold(N_INNER),
            scoring=scoring,
            n_jobs=1,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            search.fit(X[train], y[train], groups=groups[train])
        n_warn += sum(issubclass(w.category, ConvergenceWarning) for w in caught)
        z[test] = search.decision_function(X[test])
        p[test] = search.predict_proba(X[test])[:, 1]
        chosen[int(f)] = float(search.best_params_["logisticregression__C"])
    return {"z": z, "p": p, "C": chosen, "n_conv_warnings": n_warn}


def fit_baseline(ell, y, folds, C=BASELINE_C):
    """The same pipeline on the single column ``ell``, per outer fold (no inner search)."""
    X = np.asarray(ell, dtype=np.float64).reshape(-1, 1)
    z, p, gamma = np.full(len(y), np.nan), np.full(len(y), np.nan), {}
    for f in np.unique(folds):
        train, test = folds != f, folds == f
        pipe = make_pipe(C).fit(X[train], y[train])
        z[test] = pipe.decision_function(X[test])
        p[test] = pipe.predict_proba(X[test])[:, 1]
        gamma[int(f)] = float(unscale(pipe)[0][0])
    return {"z": z, "p": p, "C": {int(f): C for f in np.unique(folds)}, "gamma": gamma}


class _RecordingGroupKFold(GroupKFold):
    """GroupKFold that records the groups each ``split`` call receives."""

    def __init__(self, n_splits):
        super().__init__(n_splits)
        self.seen = []

    def split(self, X, y=None, groups=None):
        self.seen.append(None if groups is None else np.unique(groups))
        return super().split(X, y, groups)


def inner_groups_check(n_targets=12, pairs_per_target=6, seed=0) -> tuple:
    """(ok, detail): run ``fit_layer`` on tiny synthetic pairs with a recording inner CV and
    check that every outer fold's inner split received exactly the training targets (the
    groups must reach the inner CV, or targets leak between inner folds). Takes < 1 s."""
    rng = np.random.default_rng(seed)
    targets = np.repeat([f"t{i}" for i in range(n_targets)], 2 * pairs_per_target)
    y = np.tile([1, 0], n_targets * pairs_per_target)
    X = rng.standard_normal((len(y), 4)) + y[:, None]
    meta = pd.DataFrame({"target": targets})
    folds = outer_folds(meta)
    inner = _RecordingGroupKFold(N_INNER)
    fit_layer(X, y, targets, folds, grid=(1.0,), inner_cv=inner)
    expected = [np.unique(targets[folds != f]) for f in np.unique(folds)]
    ok = len(inner.seen) == len(expected) and all(
        s is not None and np.array_equal(s, e) for s, e in zip(inner.seen, expected)
    )
    detail = (
        f"fit_layer on synthetic data: {len(inner.seen)} inner splits for {len(expected)} outer "
        f"folds, each {'with' if ok else 'WITHOUT'} exactly the training targets as groups"
    )
    return ok, detail


def unscale(pipe):
    """(w_raw, b_raw) with ``X @ w_raw + b_raw == pipe.decision_function(X)`` on raw X.

    w_raw_j = w'_j / s_j and b_raw = b' - sum_j w'_j mu_j / s_j. The scaled ``coef_`` is not a
    direction in activation space.
    """
    scaler, lr = pipe.named_steps["standardscaler"], pipe.named_steps["logisticregression"]
    w = lr.coef_[0] / scaler.scale_
    b = float(lr.intercept_[0] - np.sum(lr.coef_[0] * scaler.mean_ / scaler.scale_))
    return w, b


def c_edge(chosen, grid) -> str:
    """'low' / 'high' / 'low+high' if any fold chose the smallest / largest C, else ''."""
    chosen = set(chosen)
    return "+".join(e for e, c in (("low", min(grid)), ("high", max(grid))) if c in chosen)


def most_frequent(values):
    """Most frequent C; ties go to the smaller (more regularised) one."""
    counts = Counter(values)
    best = max(counts.values())
    return min(c for c, n in counts.items() if n == best)


# --------------------------------------------------------------------------- one layer


def run_layer(paths, position, k, meta_cols, folds, grid):
    """Fit, score and save one (position, layer) of the run ``paths``. Runs in a joblib worker
    (one BLAS thread, set through the environment in ``main``)."""
    t0 = time.time()
    y, groups, flag = meta_cols["S"], meta_cols["target"], meta_cols["flag"]
    X = np.asarray(load_layer(paths, position, k), dtype=np.float64)
    if X.shape[0] != len(y):
        raise ValueError(f"{position} layer {k}: {X.shape[0]} rows, meta has {len(y)}")
    res = fit_layer(X, y, groups, folds, grid)
    C_all = most_frequent(list(res["C"].values()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        full = make_pipe(C_all).fit(X, y)
    w_raw, b_raw = unscale(full)
    scaler = full.named_steps["standardscaler"]

    folder = paths.probes / position
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "row": np.arange(len(y)),
            "fold": folds,
            "z": res["z"],
            "p": res["p"],
            "C": [res["C"][f] for f in folds],
        }
    ).to_csv(folder / f"preds_layer_{k:02d}.csv", index=False)
    np.savez(
        folder / f"probe_layer_{k:02d}.npz",
        w_raw=w_raw,
        b_raw=b_raw,
        mu=scaler.mean_,
        scale=scaler.scale_,
        C=C_all,
    )
    row = {"position": position, "layer": k, **scores(y, res["z"], res["p"], flag)}
    row |= {f"C_f{f}": c for f, c in sorted(res["C"].items())}
    row |= {
        "C_full": C_all,
        "C_edge": c_edge(res["C"].values(), grid),
        "n_conv_warnings": res["n_conv_warnings"],
        "seconds": round(time.time() - t0, 1),
    }
    print(
        f"  {position:11s} layer {k:2d}: acc {row['acc']:.3f}  pair {row['acc_pair']:.3f}  "
        f"C {sorted(res['C'].values())}  ({row['seconds']:.0f}s)",
        flush=True,
    )
    return row


# --------------------------------------------------------------------------- CLI


def parse_layers(text):
    """'0-48' / '18-24' / '0,20,21' / '0-3,40' -> sorted list of ints."""
    layers = set()
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            layers.update(range(int(a), int(b) + 1))
        elif part:
            layers.add(int(part))
    return sorted(layers)


def usable_cpus() -> int:
    """Cores this process may run on (Slurm can allow fewer than the node has)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # macOS
        return os.cpu_count() or 1


def save_folds(meta, folds, out_dir) -> Path:
    """Write folds.csv; if one exists, it must match (WP6 reuses the same folds)."""
    path = out_dir / "folds.csv"
    df = pd.DataFrame({"row": np.arange(len(meta)), "fold": folds, "target": meta["target"]})
    if path.exists():
        old = pd.read_csv(path, keep_default_na=False)
        if len(old) != len(df) or not (old["fold"].to_numpy() == folds).all():
            raise ValueError(f"{path} exists with different folds; move it away to refit")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--bias", default="gender", help="StereoSet bias type (default: gender)")
    ap.add_argument("--positions", default=",".join(POSITIONS))
    ap.add_argument(
        "--layers", default=None, help="e.g. 0-48 or 18-24 (default: every hidden state)"
    )
    ap.add_argument(
        "--root",
        default=None,
        help="data root (default: paths.rq2_root); results then go to {root}/results",
    )
    ap.add_argument("--grid", default=",".join(f"{c:g}" for c in GRID))
    ap.add_argument("--n_jobs", type=int, default=-1, help="parallel (position, layer) fits")
    args = ap.parse_args(argv)

    from joblib import Parallel, delayed, parallel_config

    t0 = time.time()
    paths = run_paths(args.model, args.bias, root=args.root)
    out_dir, summary_path = paths.probes, paths.summary
    grid = tuple(float(c) for c in args.grid.split(","))
    positions = [p for p in args.positions.split(",") if p]
    unknown = set(positions) - set(POSITIONS)
    if unknown:
        raise ValueError(f"unknown positions {sorted(unknown)}; expected {POSITIONS}")
    n_layers = n_decoder_layers(paths)
    layers = parse_layers(args.layers) if args.layers else list(range(n_layers + 1))
    if max(layers) > n_layers:
        raise ValueError(f"--layers goes to {max(layers)}; {args.model} has {n_layers} layers")

    meta = load_meta(paths)
    y = meta["S"].to_numpy()
    folds = outer_folds(meta)
    save_folds(meta, folds, out_dir)
    flag = pair_flag(meta)
    cols = {"S": y, "target": meta["target"].to_numpy(), "flag": flag}
    print(
        f"{args.model}: {len(meta)} rows, {meta['target'].nunique()} targets, "
        f"{N_OUTER} outer folds (sizes {np.bincount(folds).tolist()}); "
        f"{int(flag.sum() // 2)} pairs with target after the word; grid {list(grid)}; "
        f"{usable_cpus()} usable CPUs",
        flush=True,
    )

    # Baseline: ell alone.
    if meta["ell"].isna().any():
        raise ValueError("ell has NaN; run probing.extract first")
    base = fit_baseline(meta["ell"].to_numpy(), y, folds)
    pd.DataFrame(
        {"row": np.arange(len(y)), "fold": folds, "z": base["z"], "p": base["p"], "C": BASELINE_C}
    ).to_csv(out_dir / "baseline_ell.csv", index=False)
    ell = meta["ell"].to_numpy()
    base_row = {"position": "baseline_ell", "layer": -1, **scores(y, base["z"], base["p"], flag)}
    base_row |= {f"C_f{f}": c for f, c in sorted(base["C"].items())}
    base_row |= {"C_full": BASELINE_C, "C_edge": "", "n_conv_warnings": 0}
    ss = pairwise_accuracy(ell)
    print(
        f"  baseline ell: acc {base_row['acc']:.3f}  pair {base_row['acc_pair']:.3f} "
        f"(share ell_stereo > ell_anti, ties 1/2: {ss:.3f}); gamma per fold "
        f"{[round(g, 3) for g in base['gamma'].values()]}",
        flush=True,
    )

    # Probes, (position, layer) in parallel with one BLAS thread each.
    tasks = [(p, k) for p in positions for k in layers]
    n_jobs = min(len(tasks), usable_cpus()) if args.n_jobs == -1 else args.n_jobs
    # One BLAS thread per worker, via the environment the workers start with (calling into the
    # BLAS library at runtime, as threadpoolctl does, makes JUPITER's FlexiBLAS print "BLAS").
    for var in THREAD_VARS:
        os.environ[var] = "1"
    with parallel_config(backend="loky", inner_max_num_threads=1):
        rows = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(run_layer)(paths, p, k, cols, folds, grid) for p, k in tasks
        )

    summary = pd.DataFrame([base_row, *rows])
    summary.insert(2, "rel_depth", rel_depth(summary["layer"], n_layers))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    high = summary[summary["C_edge"].str.contains("high", na=False)]
    low = summary[summary["C_edge"].str.contains("low", na=False)]
    info = {
        "model": args.model,
        "bias": args.bias,
        "n_rows": len(meta),
        "positions": positions,
        "layers": layers,
        "n_layers": n_layers,
        "grid": list(grid),
        "inner_scoring": INNER_SCORING,
        "baseline_C": BASELINE_C,
        "n_jobs": n_jobs,
        "seconds_total": round(time.time() - t0, 1),
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "summary": str(summary_path),
    }
    (out_dir / "run.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    record_run(paths, "probe", info, stamp=False)

    print(f"\ndone in {info['seconds_total']:.0f}s with {n_jobs} workers")
    if len(high):
        print(
            f"C at the TOP of the grid in {len(high)} of {len(rows)} probes (wants less "
            "regularisation; widen --grid upwards): "
            + ", ".join(f"{r.position}/{r.layer}" for r in high.itertuples())
        )
    if len(low):
        print(
            f"C at the bottom of the grid in {len(low)} of {len(rows)} probes (flattest probe "
            "fits best: expected where there is no signal): "
            + ", ".join(f"{r.position}/{r.layer}" for r in low.itertuples())
        )
    print(f"summary: {summary_path}\npredictions and weights: {out_dir}")


if __name__ == "__main__":
    main()
