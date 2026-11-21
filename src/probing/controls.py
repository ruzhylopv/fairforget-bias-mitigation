"""RQ2 WP6 controls on the WP5 probes (gate 2): null, p-values, bootstrap CIs, misspellings,
the C grid below 1e-4, and the bookkeeping checks. CPU only.

Everything reuses WP5 (``probing.probe``): the same ``folds.csv``, the same ``fit_layer`` with
the same grid (read from ``run.json``), the same ``scores``.

* **null**: labels swapped within a random half of the pairs (``flip_within_pairs``), then the
  whole nested CV is rerun for every probe. Only scores are kept, one row per (seed, probe),
  appended as they finish, so an interrupted run resumes where it stopped.
* **stats**: from the stored WP5 predictions and the null scores: p-values (per probe and
  max-over-probes corrected), bootstrap CIs over targets (the same draws for every probe and the
  baseline, so the gap probe - baseline is paired), the summary without the misspelled pairs,
  and the gate 2 checklist. The CIs treat the fitted probes as fixed (no refitting variance).
* **cgrid**: the real labels with another grid (default 1e-6 .. 1e-3) for some layers; only
  scores and the chosen C are written, the WP5 predictions are not touched.

Outputs (small, in ``run_paths(model, bias).results`` = ``results/rq2/{bias}/{model}``)::

    null_scores.csv                   seed, position, layer, scores (as summary.csv), C per fold
    pvalues.csv                       position, layer, metric, real, p, p_corr, B
    ci.csv                            position, layer, rel_depth, metric, estimate, low, high,
                                      gap (vs ell), gap_layer0 (vs layer 0), their CIs
    robustness_no_misspellings.csv    position, layer, metric, all_pairs, no_misspellings, change
    c_grid_low.csv                    cgrid: scores with the new grid next to summary.csv
    gate2.txt                         the checklist, as printed by stats
    run.json                          steps "controls null" / "controls cgrid" / "controls stats"

The misspelled words come from ``config/rq2/{bias}.yaml``.

Usage (on a CPU node; ``probing.probe`` must have run, see docs/rq2_probing.md)::

    python -m probing.controls null --model lapa-12b-instruct --bias gender --seeds 0-4
    python -m probing.controls null --model lapa-12b-instruct --bias gender --seeds 0-99  # resumes
    python -m probing.controls cgrid --model lapa-12b-instruct --bias gender   # C at low edge
    python -m probing.controls stats --model lapa-12b-instruct --bias gender
    uv run python scripts/plot_controls.py --model lapa-12b-instruct --bias gender   # laptop
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from probing.gates import write_gate
from probing.io import (
    POSITIONS,
    bias_config,
    chosen_on,
    fixed_layer,
    load_layer,
    load_meta,
    n_decoder_layers,
    record_run,
    rel_depth,
    report_config,
    run_paths,
)
from probing.probe import (
    GRID,
    N_INNER,
    SUBSETS,
    THREAD_VARS,
    c_edge,
    fit_baseline,
    fit_layer,
    inner_groups_check,
    outer_folds,
    pair_correct,
    pair_flag,
    parse_layers,
    scores,
    usable_cpus,
)

BASELINE = ("baseline_ell", -1)
LEXICAL_LAYER = 0  # lexical baseline: the probe on the token embedding (hidden_states[0])
ACC_METRICS = tuple(f"{m}{s}" for s in SUBSETS for m in ("acc", "acc_pair"))
CI_METRICS = tuple(f"{m}{s}" for s in SUBSETS for m in ("acc", "acc_pair", "logloss"))
NULL_BAND = 0.03  # gate 2: mean shuffled-label accuracy within 0.50 +- 3 points
N_BOOT = 2000


# --------------------------------------------------------------------------- null labels


def flip_within_pairs(y, seed) -> np.ndarray:
    """Null labels: swap y[2i] and y[2i+1] in a random half (n_pairs // 2) of the pairs.

    Every pair keeps one 1 and one 0, so labels stay 50/50 overall and per fold. The flips
    depend only on ``seed`` and the number of pairs, so every probe of a seed gets the same.
    """
    y = np.asarray(y)
    if len(y) % 2 or not (y[0::2] + y[1::2] == 1).all():
        raise ValueError("y must come in pairs (rows 2i, 2i+1) with one 1 and one 0")
    n_pairs = len(y) // 2
    swap = np.zeros(n_pairs, bool)
    swap[np.random.default_rng(seed).choice(n_pairs, n_pairs // 2, replace=False)] = True
    return np.where(np.repeat(swap, 2), 1 - y, y)


def fit_task(paths, position, k, y, groups, folds, flag, grid, seed=None) -> dict:
    """One probe of the run ``paths``: nested CV (``fit_layer``) with the real labels (seed
    None) or the null labels of ``seed``; returns the scores and the chosen C, not the
    predictions."""
    t0 = time.time()
    X = np.asarray(load_layer(paths, position, k), dtype=np.float64)
    if X.shape[0] != len(y):
        raise ValueError(f"{position} layer {k}: {X.shape[0]} rows, meta has {len(y)}")
    y_fit = y if seed is None else flip_within_pairs(y, seed)
    res = fit_layer(X, y_fit, groups, folds, grid)
    row = {"seed": seed, "position": position, "layer": k}
    row |= scores(y_fit, res["z"], res["p"], flag)
    row |= {f"C_f{f}": c for f, c in sorted(res["C"].items())}
    row |= {
        "C_edge": c_edge(res["C"].values(), grid),
        "n_conv_warnings": res["n_conv_warnings"],
        "seconds": round(time.time() - t0, 1),
    }
    return row


def baseline_task(ell, y, folds, flag, seed=None) -> dict:
    y_fit = y if seed is None else flip_within_pairs(y, seed)
    res = fit_baseline(ell, y_fit, folds)
    return {"seed": seed, "position": BASELINE[0], "layer": BASELINE[1]} | scores(
        y_fit, res["z"], res["p"], flag
    )


def score_columns(n_folds) -> list:
    cols = ["seed", "position", "layer"]
    for s in SUBSETS:
        cols += [f"acc{s}", f"acc_pair{s}", f"logloss{s}", f"n_pairs{s}"]
    return cols + [f"C_f{f}" for f in range(n_folds)] + ["C_edge", "n_conv_warnings", "seconds"]


def run_parallel(tasks, n_jobs, on_result):
    """Run ``fit_task`` over ``tasks`` (kwargs dicts), one BLAS thread per worker, calling
    ``on_result(row)`` as each finishes (any order)."""
    from joblib import Parallel, delayed, parallel_config

    # As in probing.probe: one BLAS thread per worker, set through the environment.
    for var in THREAD_VARS:
        os.environ[var] = "1"
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
            delayed(fit_task)(**t) for t in tasks
        )
        for row in results:
            on_result(row)


def run_null(ctx, seeds, positions, layers, path, n_jobs=-1):
    """Null scores for ``seeds`` x every probe (+ the baseline), appended to ``path`` (CSV).

    (seed, position, layer) rows already in ``path`` are skipped, so a run cut off by the time
    limit resumes with the same command.
    """
    path = Path(path)
    cols = score_columns(len(np.unique(ctx["folds"])))
    done = set()
    if path.exists():
        old = pd.read_csv(path)
        done = set(zip(old["seed"].astype(int), old["position"], old["layer"].astype(int)))
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="raise")
        if new_file:
            writer.writeheader()
            fh.flush()

        def write(row):
            writer.writerow({c: row.get(c, "") for c in cols})
            fh.flush()

        for seed in seeds:  # the baseline has one feature: seconds for all seeds
            if (seed, *BASELINE) not in done:
                write(baseline_task(ctx["ell"], ctx["y"], ctx["folds"], ctx["flag"], seed))
        tasks = [
            {
                "paths": ctx["paths"],
                "position": p,
                "k": k,
                "y": ctx["y"],
                "groups": ctx["groups"],
                "folds": ctx["folds"],
                "flag": ctx["flag"],
                "grid": ctx["grid"],
                "seed": seed,
            }
            for seed in seeds
            for p in positions
            for k in layers
            if (seed, p, k) not in done
        ]
        n_jobs = min(len(tasks), usable_cpus()) if n_jobs == -1 else n_jobs
        print(
            f"null: {len(seeds)} seeds x {len(positions) * len(layers)} probes, "
            f"{len(tasks)} fits to do ({len(done)} rows already in {path}), {n_jobs} workers",
            flush=True,
        )
        if not tasks:
            return
        left_per_seed = {s: 0 for s in seeds}
        for t in tasks:
            left_per_seed[t["seed"]] += 1
        bar = Progress(len(tasks), len(seeds), sum(v == 0 for v in left_per_seed.values()))
        print(
            "progress: a line per 1% of the fits (and at least every 2 min while fits finish); "
            f"nothing until the first round of {n_jobs} fits, one per worker, is done",
            flush=True,
        )

        def report(row):
            write(row)
            left_per_seed[row["seed"]] -= 1
            bar.update(row, seed_done=left_per_seed[row["seed"]] == 0)

        run_parallel(tasks, n_jobs, report)


def _hm(seconds) -> str:
    m = int(seconds // 60)
    return f"{m // 60}h{m % 60:02d}m" if m >= 60 else f"{m}m"


class Progress:
    """Text progress bar for logs: prints a new line (no carriage returns, so ``tail -f`` and
    the saved log both read cleanly) at every whole percent of the fits, or after ``every``
    seconds without a line, and at the end."""

    def __init__(self, total, n_seeds, seeds_done=0, width=30, every=120.0, out=None):
        self.total, self.n_seeds, self.seeds_done = total, n_seeds, seeds_done
        self.width, self.every, self.out = width, every, out
        self.n, self.t0, self.last_pct, self.last_t = 0, time.time(), 0, time.time()

    def line(self, row=None) -> str:
        frac = self.n / self.total if self.total else 1.0
        fill = int(round(frac * self.width))
        el = time.time() - self.t0
        eta = _hm(el / self.n * (self.total - self.n)) if self.n else "?"
        text = (
            f"[{'#' * fill}{'-' * (self.width - fill)}] {100 * frac:3.0f}%  "
            f"{self.n}/{self.total} fits | seeds done {self.seeds_done}/{self.n_seeds} | "
            f"{_hm(el)} elapsed, ~{eta} left"
        )
        if row is not None:
            text += (
                f" | last: seed {row['seed']} {row['position']} {row['layer']}: "
                f"acc {row['acc']:.3f} pair {row['acc_pair']:.3f}"
            )
        return text

    def update(self, row=None, seed_done=False):
        self.n += 1
        self.seeds_done += bool(seed_done)
        pct = int(100 * self.n / self.total) if self.total else 100
        now = time.time()
        if pct > self.last_pct or now - self.last_t >= self.every or self.n == self.total:
            self.last_pct, self.last_t = pct, now
            print(self.line(row), file=self.out, flush=True)


# --------------------------------------------------------------------------- p-values


def complete_seeds(null, probes) -> list:
    """Seeds with a null score for every (position, layer) in ``probes``."""
    want = set(probes)
    have = {}
    for s, pos, k in zip(null["seed"], null["position"], null["layer"]):
        have.setdefault(int(s), set()).add((pos, int(k)))
    return sorted(s for s, got in have.items() if want <= got)


def pvalues(real, null, metrics=ACC_METRICS) -> pd.DataFrame:
    """Permutation p-values per probe and metric (higher = better), B = complete null seeds.

    p      = (1 + #{b: acc_b >= acc_real}) / (B + 1), the same probe in shuffle b;
    p_corr = (1 + #{b: max over probes of acc_b >= acc_real}) / (B + 1), the max over all
             probes of shuffle b (not the baseline), which corrects for testing every layer.

    The baseline gets its own p (from its own null) and no p_corr.
    """
    is_base = real["position"] == BASELINE[0]
    probes = list(zip(real.loc[~is_base, "position"], real.loc[~is_base, "layer"]))
    seeds = complete_seeds(null, probes)
    if not seeds:
        raise ValueError("no null seed covers every probe in the summary")
    null = null[null["seed"].isin(seeds)]
    probe_rows = null[null["position"] != BASELINE[0]]
    probe_rows = probe_rows[
        pd.MultiIndex.from_frame(probe_rows[["position", "layer"]]).isin(probes)
    ]
    B = len(seeds)
    rows = []
    for m in metrics:
        best = probe_rows.groupby("seed")[m].max().reindex(seeds).to_numpy()
        per = {key: d.set_index("seed")[m] for key, d in null.groupby(["position", "layer"])}
        for r in real.itertuples(index=False):
            key = (r.position, int(r.layer))
            if key not in per:
                continue
            value = getattr(r, m)
            acc_b = per[key].reindex(seeds).to_numpy()
            p = (1 + np.sum(acc_b >= value)) / (B + 1)
            p_corr = np.nan if key == BASELINE else (1 + np.sum(best >= value)) / (B + 1)
            rows.append(
                {
                    "position": key[0],
                    "layer": key[1],
                    "metric": m,
                    "real": value,
                    "null_mean": float(np.mean(acc_b)),
                    "null_max": float(np.max(acc_b)),
                    "p": float(p),
                    "p_corr": float(p_corr),
                    "B": B,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- bootstrap


def target_draws(groups, R=N_BOOT, seed=0):
    """(targets, counts): counts[r, t] = how often target t was drawn in resample r, when the
    n_targets targets are drawn with replacement. One matrix is shared by every probe."""
    targets, codes = np.unique(np.asarray(groups), return_inverse=True)
    T = len(targets)
    idx = np.random.default_rng(seed).integers(0, T, size=(R, T))
    counts = np.zeros((R, T), dtype=np.int64)
    np.add.at(counts, (np.repeat(np.arange(R), T), idx.ravel()), 1)
    return targets, codes, counts


def _per_target(codes, T, values, mask):
    """Sum of ``values`` and count over rows in ``mask``, per target."""
    v = np.where(mask, values, 0.0)
    return np.bincount(codes, weights=v, minlength=T), np.bincount(
        codes, weights=mask.astype(float), minlength=T
    )


def resampled(y, z, p, flag, codes, T, counts) -> dict:
    """Per metric: (estimate on the full data, values under every resample in ``counts``).

    A resample takes every row of each drawn target, with repeats, so a ratio metric becomes
    (counts @ per-target sum) / (counts @ per-target n). Log-loss clips p like sklearn.
    """
    y, z, p, flag = (np.asarray(a) for a in (y, z, p, flag))
    correct = ((z > 0).astype(int) == y).astype(float)
    eps = np.finfo(np.float64).eps
    pc = np.clip(p.astype(np.float64), eps, 1 - eps)
    nll = -(y * np.log(pc) + (1 - y) * np.log(1 - pc))
    pair_c = np.repeat(pair_correct(z, y), 2)
    pair_rows = np.zeros(len(y), bool)
    pair_rows[0::2] = True  # one row per pair carries the pair's score
    ones = np.ones(T)
    out = {}
    for suffix, value in SUBSETS.items():
        sub = np.ones(len(y), bool) if value is None else (flag == value)
        for name, vals, mask in (
            ("acc", correct, sub),
            ("acc_pair", pair_c, sub & pair_rows),
            ("logloss", nll, sub),
        ):
            num, den = _per_target(codes, T, vals, mask)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"{name}{suffix}"] = (
                    num @ ones / (den @ ones),
                    (counts @ num) / (counts @ den),
                )
    return out


def bootstrap_ci(
    y,
    groups,
    flag,
    preds,
    baseline=BASELINE,
    lexical_layer=LEXICAL_LAYER,
    R=N_BOOT,
    seed=0,
    level=0.95,
):
    """Percentile CIs over targets for every entry of ``preds`` ({key: (z, p)}) and, paired on
    the same draws, for two gaps:

    * ``gap*``: entry - ``baseline`` (predictability, ``ell``);
    * ``gap_layer0*``: entry - the probe at ``lexical_layer`` (the lexical baseline: the token
      embedding, no context) of the same position. NaN for the baseline and when that layer
      is not in ``preds``.

    Every metric and subset of ``CI_METRICS`` (acc, acc_pair, logloss; all, _tfirst, _tafter).

    ``share_*_le0`` is the share of resamples where the gap is <= 0 (for log-loss, lower is
    better: read the gap's sign)."""
    _, codes, counts = target_draws(groups, R, seed)
    T = counts.shape[1]
    lo, hi = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    res = {key: resampled(y, z, p, flag, codes, T, counts) for key, (z, p) in preds.items()}
    base = res[baseline]

    def gap_cols(prefix, est, boot, ref):
        if ref is None:
            return dict.fromkeys(
                (prefix, f"{prefix}_low", f"{prefix}_high", f"share_{prefix}_le0"), np.nan
            )
        d = boot - ref[1]
        return {
            prefix: est - ref[0],
            f"{prefix}_low": np.nanpercentile(d, lo),
            f"{prefix}_high": np.nanpercentile(d, hi),
            f"share_{prefix}_le0": float(np.nanmean(d <= 0)),
        }

    rows = []
    for key, r in res.items():
        lex = None if key == baseline else res.get((key[0], lexical_layer))
        for m in CI_METRICS:
            est, boot = r[m]
            rows.append(
                {
                    "position": key[0],
                    "layer": key[1],
                    "metric": m,
                    "estimate": est,
                    "low": np.nanpercentile(boot, lo),
                    "high": np.nanpercentile(boot, hi),
                    **gap_cols("gap", est, boot, base[m]),
                    **gap_cols("gap_layer0", est, boot, None if lex is None else lex[m]),
                    "R": R,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- misspellings


def drop_pairs(meta, words) -> np.ndarray:
    """Row mask keeping the pairs where neither row's attribute word is in ``words``
    (case-insensitive, surrounding punctuation ignored)."""
    w = (
        meta["word"]
        .astype(str)
        .str.lower()
        .str.strip(" .,!?;:'\"")
        .isin([x.lower() for x in words])
    )
    w = w.to_numpy()
    bad_pair = w[0::2] | w[1::2]
    return ~np.repeat(bad_pair, 2)


def robustness(meta, preds, keep) -> pd.DataFrame:
    """Scores of every probe from the stored predictions, on all pairs and on ``keep`` only."""
    y, flag = meta["S"].to_numpy(), pair_flag(meta)
    rows = []
    for key, (z, p) in preds.items():
        a = scores(y, z, p, flag)
        b = scores(y[keep], z[keep], p[keep], flag[keep])
        for m in CI_METRICS:
            rows.append(
                {
                    "position": key[0],
                    "layer": key[1],
                    "metric": m,
                    "all_pairs": a[m],
                    "no_misspellings": b[m],
                    "change": b[m] - a[m],
                    "n_pairs_dropped": int((~keep).sum() // 2),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- loading


def load_context(paths) -> dict:
    """Meta columns, the saved folds (checked against a fresh ``outer_folds``) and the grid of
    the WP5 run (``{probes}/run.json``)."""
    meta = load_meta(paths)
    out_dir = paths.probes
    folds_df = pd.read_csv(out_dir / "folds.csv", keep_default_na=False)
    folds = folds_df["fold"].to_numpy()
    if len(folds) != len(meta) or not (folds == outer_folds(meta)).all():
        raise ValueError(f"{out_dir / 'folds.csv'} does not match outer_folds(meta)")
    if not (folds_df["target"].to_numpy() == meta["target"].to_numpy()).all():
        raise ValueError(f"{out_dir / 'folds.csv'}: targets differ from the meta")
    run = out_dir / "run.json"
    grid = tuple(json.loads(run.read_text())["grid"]) if run.exists() else GRID
    return {
        "paths": paths,
        "meta": meta,
        "y": meta["S"].to_numpy(),
        "groups": meta["target"].to_numpy(),
        "flag": pair_flag(meta),
        "ell": meta["ell"].to_numpy(),
        "folds": folds,
        "folds_df": folds_df,
        "grid": grid,
    }


def load_preds(out_dir, keys) -> dict:
    """{(position, layer): (z, p, fold)} from the WP5 prediction files, baseline included."""
    preds = {}
    for pos, k in keys:
        path = (
            out_dir / "baseline_ell.csv"
            if (pos, k) == BASELINE
            else out_dir / pos / f"preds_layer_{k:02d}.csv"
        )
        d = pd.read_csv(path)
        if not (d["row"].to_numpy() == np.arange(len(d))).all():
            raise ValueError(f"{path}: rows out of order")
        preds[(pos, int(k))] = (d["z"].to_numpy(), d["p"].to_numpy(), d["fold"].to_numpy())
    return preds


# --------------------------------------------------------------------------- gate 2


def null_check(null, n_rows, band=NULL_BAND) -> tuple:
    """Gate 2 item: the shuffled-label null sits at chance.

    Judged per probe on the **mean over seeds** of acc and acc_pair: a leak (labels reaching
    the test fold) shifts that mean, while single runs scatter around it. Single runs are
    not required to stay in the band: with many seeds and probes some always leave it, and
    with folds grouped by target their spread is wider than the binomial one (both are
    reported, as information).
    """
    name = f"null mean over seeds within 0.50 +- {band:g}, every probe"
    if null is None or null.empty:
        return (name, None, "no null_scores.csv yet")
    probes = null[null["position"] != BASELINE[0]]
    n_seeds = probes["seed"].nunique()
    chance_sd = {"acc": np.sqrt(0.25 / n_rows), "acc_pair": np.sqrt(0.25 / (n_rows // 2))}
    ok, parts = True, []
    for m in ("acc", "acc_pair"):
        means = probes.groupby(["position", "layer"])[m].mean()
        dev = (means - 0.5).abs()
        worst = dev.idxmax()
        ok &= bool((dev <= band).all())
        runs_out = int(((probes[m] - 0.5).abs() > band).sum())
        parts.append(
            f"{m}: largest |mean - 0.5| {dev.max():.3f} ({worst[0]} {worst[1]}); single runs "
            f"{probes[m].min():.3f}-{probes[m].max():.3f}, {runs_out}/{len(probes)} outside the "
            f"band, SD {probes[m].std():.3f} vs binomial {chance_sd[m]:.3f}"
        )
    return (f"{name} ({n_seeds} seeds)", ok, "; ".join(parts))


def gate_checks(ctx, summary, preds, null=None, cgrid=None) -> list:
    """[(item, ok or None, detail)] for gate 2. ok None = not run yet / judged by hand."""
    y, folds, groups = ctx["y"], ctx["folds"], ctx["groups"]
    items = []

    items.append(null_check(null, len(y)))

    # Folds: 50/50 by label, no target in two folds, saved folds = a fresh outer_folds.
    bal = [
        (int((y[folds == f] == 1).sum()), int((y[folds == f] == 0).sum())) for f in np.unique(folds)
    ]
    per_target = pd.Series(folds).groupby(groups).nunique()
    items.append(
        (
            "every outer fold 50/50; no target in train and test",
            all(a == b for a, b in bal) and bool((per_target == 1).all()),
            f"S=1/S=0 per fold {bal}; targets in >1 fold: {int((per_target > 1).sum())}; "
            "tests/test_probing_probe.py::test_outer_folds_never_share_a_target_and_keep_pairs"
            "_together",
        )
    )
    ok, detail = inner_groups_check()
    items.append(("groups reach the inner CV", ok, f"{detail} (inner GroupKFold({N_INNER}))"))

    # Probe and baseline on the same folds.csv, and the stored predictions give summary.csv.
    same = all((f == folds).all() for _, _, f in preds.values())
    flag = ctx["flag"]
    max_diff = 0.0
    for r in summary.itertuples(index=False):
        z, p, _ = preds[(r.position, int(r.layer))]
        s = scores(y, z, p, flag)
        max_diff = max(max_diff, abs(s["acc"] - r.acc), abs(s["acc_pair"] - r.acc_pair))
    items.append(
        (
            "probe and baseline scored on the same folds.csv",
            bool(same and max_diff < 1e-9),
            f"{len(preds)} prediction files, fold columns equal folds.csv: {same}; "
            f"max |score from preds - summary.csv| = {max_diff:.1e}",
        )
    )

    # C at the bottom of the grid: refit those probes with smaller C (cgrid); fine if the
    # scores barely move.
    name = "chosen C not at a grid edge, or explained"
    need = low_edge_probes(summary)
    have = set() if cgrid is None else set(zip(cgrid["position"], cgrid["layer"].astype(int)))
    if not need:
        items.append((name, True, "no probe chose the smallest C"))
    elif cgrid is None or cgrid.empty or not set(need) <= have:
        missing = sorted(set(need) - have)
        items.append((name, None, f"C at the bottom of the grid, no cgrid run yet for {missing}"))
    else:
        d_acc = (cgrid["acc"] - cgrid["acc_wp5"]).abs().max()
        d_pair = (cgrid["acc_pair"] - cgrid["acc_pair_wp5"]).abs().max()
        low = cgrid[cgrid["C_edge"].str.contains("low", na=False)]
        detail = (
            f"layers {sorted(int(k) for k in cgrid['layer'].unique())} "
            f"with grid {cgrid['grid'].iloc[0]}: "
            f"max change acc {d_acc:.3f}, pair {d_pair:.3f}; still at the bottom: "
            + (", ".join(f"{r.position}/{r.layer}" for r in low.itertuples()) or "none")
        )
        items.append(
            ("chosen C not at a grid edge, or explained", bool(max(d_acc, d_pair) < 0.01), detail)
        )
    return items


def low_edge_probes(summary) -> list:
    """[(position, layer)] of the probes where some outer fold chose the smallest C."""
    if summary is None:
        return []
    s = summary[(summary["position"] != BASELINE[0])]
    s = s[s["C_edge"].fillna("").astype(str).str.contains("low")]
    return sorted(zip(s["position"], s["layer"].astype(int)))


# --------------------------------------------------------------------------- CLI


def parse_seeds(text):
    return parse_layers(text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("null", "cgrid", "stats"):
        s = sub.add_parser(name)
        s.add_argument("--model", required=True)
        s.add_argument("--bias", default="gender", help="StereoSet bias type (default: gender)")
        s.add_argument(
            "--root",
            default=None,
            help="data root (default: paths.rq2_root); results then go to {root}/results",
        )
        s.add_argument("--positions", default=",".join(POSITIONS))
        s.add_argument(
            "--layers",
            default=None,
            help="default: every layer in summary.csv (cgrid: the probes with C at the low edge)",
        )
        s.add_argument("--n_jobs", type=int, default=-1)
    sub.choices["null"].add_argument("--seeds", default="0-4", help="e.g. 0-4 or 0-99")
    sub.choices["cgrid"].add_argument("--grid", default="1e-6,1e-5,1e-4,1e-3")
    sub.choices["stats"].add_argument("--n_boot", type=int, default=N_BOOT)
    sub.choices["stats"].add_argument("--boot_seed", type=int, default=0)
    args = ap.parse_args(argv)

    paths = run_paths(args.model, args.bias, root=args.root)
    out_dir, results = paths.probes, paths.results
    results.mkdir(parents=True, exist_ok=True)
    ctx = load_context(paths)
    summary_path = results / "summary.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else None
    positions = [p for p in args.positions.split(",") if p]
    if args.layers:
        layers = parse_layers(args.layers)
    elif summary is not None:
        layers = sorted(int(k) for k in summary["layer"].unique() if k >= 0)
    else:
        raise ValueError(f"no {summary_path}; pass --layers")
    print(
        f"{args.model} / {args.bias}: {len(ctx['y'])} rows, "
        f"{len(np.unique(ctx['groups']))} targets, grid {list(ctx['grid'])}, "
        f"{usable_cpus()} usable CPUs; WP5 outputs {out_dir}",
        flush=True,
    )

    settings = {"positions": positions, "layers": layers}
    if args.cmd == "null":
        seeds = parse_seeds(args.seeds)
        run_null(ctx, seeds, positions, layers, results / "null_scores.csv", args.n_jobs)
        record_run(paths, "controls null", settings | {"seeds": seeds})
        print(f"null scores: {results / 'null_scores.csv'}")

    elif args.cmd == "cgrid":
        grid = tuple(float(c) for c in args.grid.split(","))
        ctx_grid = ctx | {"grid": grid}
        if args.layers:
            keys = [(p, k) for p in positions for k in layers]
        else:  # the probes where some fold chose the smallest C of the WP5 grid
            keys = [(p, k) for p, k in low_edge_probes(summary) if p in positions]
        if not keys:
            print("cgrid: no probe chose the smallest C; nothing to refit")
            return
        tasks = [
            {k: ctx_grid[k] for k in ("paths", "y", "groups", "folds", "flag", "grid")}
            | {"position": p, "k": k}
            for p, k in keys
        ]
        rows = []
        n_jobs = min(len(tasks), usable_cpus()) if args.n_jobs == -1 else args.n_jobs
        print(f"cgrid: {len(tasks)} fits, grid {list(grid)}, {n_jobs} workers", flush=True)
        run_parallel(tasks, n_jobs, rows.append)
        df = pd.DataFrame(rows).drop(columns="seed").sort_values(["position", "layer"])
        df.insert(2, "grid", ",".join(f"{c:g}" for c in grid))
        if summary is not None:
            old = summary.set_index(["position", "layer"])
            for m in ("acc", "acc_pair"):
                df[f"{m}_wp5"] = [old.loc[(r.position, r.layer), m] for r in df.itertuples()]
            df["C_wp5"] = [
                ",".join(f"{old.loc[(r.position, r.layer), f'C_f{f}']:g}" for f in range(5))
                for r in df.itertuples()
            ]
        df.to_csv(results / "c_grid_low.csv", index=False)
        record_run(paths, "controls cgrid", {"probes": keys, "grid": list(grid)})
        for r in df.itertuples():
            print(
                f"  {r.position:11s} layer {r.layer}: acc {r.acc:.3f} (WP5 {r.acc_wp5:.3f})  pair "
                f"{r.acc_pair:.3f} (WP5 {r.acc_pair_wp5:.3f})  C "
                f"{[getattr(r, f'C_f{f}') for f in range(5)]}  {r.C_edge or 'inside'}"
            )
        print(f"wrote {results / 'c_grid_low.csv'}")

    elif args.cmd == "stats":
        if summary is None:
            raise ValueError(f"no {summary_path}")
        summary = summary[summary["position"].isin([*positions, BASELINE[0]])]
        summary = summary[(summary["layer"].isin(layers)) | (summary["position"] == BASELINE[0])]
        keys = list(zip(summary["position"], summary["layer"].astype(int)))
        preds = load_preds(out_dir, keys)
        meta, y, flag = ctx["meta"], ctx["y"], ctx["flag"]

        t0 = time.time()
        ci = bootstrap_ci(
            y,
            ctx["groups"],
            flag,
            {k: v[:2] for k, v in preds.items()},
            R=args.n_boot,
            seed=args.boot_seed,
        )
        n_layers = n_decoder_layers(paths)
        ci.insert(2, "rel_depth", rel_depth(ci["layer"], n_layers))
        ci.to_csv(results / "ci.csv", index=False)
        print(f"ci.csv: {len(ci)} rows, R = {args.n_boot} ({time.time() - t0:.0f}s)")

        misspelled = bias_config(args.bias)["misspelled"]
        keep = drop_pairs(meta, misspelled)
        dropped = meta.loc[~keep, ["word", "sentence"]]
        rob = robustness(meta, {k: v[:2] for k, v in preds.items()}, keep)
        rob.to_csv(results / "robustness_no_misspellings.csv", index=False)
        print(f"robustness_no_misspellings.csv: dropped {int((~keep).sum() // 2)} pairs:")
        for r in dropped.itertuples():
            print(f"    {r.word!r:16} {r.sentence}")

        null_path = results / "null_scores.csv"
        null = pd.read_csv(null_path) if null_path.exists() else None
        pv = None
        if null is not None and not null.empty:
            pv = pvalues(summary, null)
            pv.to_csv(results / "pvalues.csv", index=False)
            print(f"pvalues.csv: B = {pv['B'].iloc[0]} complete null seeds")
        cg_path = results / "c_grid_low.csv"
        cgrid = pd.read_csv(cg_path) if cg_path.exists() else None

        report = fixed_depth_report(ci, pv, paths, n_layers) + peak_report(ci, rob, pv, null)
        gate = gate_checks(ctx, summary, preds, null, cgrid)
        text = write_gate(results / "gate2.txt", gate, report)
        record_run(
            paths,
            "controls stats",
            settings
            | {
                "n_boot": args.n_boot,
                "boot_seed": args.boot_seed,
                "null_seeds_B": None if pv is None else int(pv["B"].iloc[0]),
                "misspelled": misspelled,
                "n_layers": n_layers,
                "fixed_depth": report_config()["fixed_depth"]
                | {"layer_this_run": fixed_layer(n_layers)},
            },
        )
        print("\nGate 2\n" + text)


def fixed_depth_report(ci, pv, paths, n_layers, cfg=None) -> list:
    """Lines for the fixed depth of ``config/rq2/report.yaml``: the run's nearest layer, its
    estimate and both gaps (no selection over layers, so p is the per-probe p-value)."""
    cfg = cfg or report_config()
    fd = cfg["fixed_depth"]
    k, pos = fixed_layer(n_layers, cfg), fd["position"]
    note = (
        " (CHOSEN ON THIS RUN: not an independent test)"
        if chosen_on(paths, cfg)
        else f" (fixed {fd['date']} on {fd['chosen_on']['model']}/{fd['chosen_on']['bias']})"
    )
    lines = [
        f"fixed depth {fd['rel_depth']:.3f} = layer {k} of {n_layers}, {pos}{note}",
    ]
    for m in ("acc_pair", "acc"):
        d = ci[(ci["position"] == pos) & (ci["layer"] == k) & (ci["metric"] == m)]
        if not len(d):
            lines.append(f"  {m}: layer {k} not in ci.csv")
            continue
        r = d.iloc[0]
        line = (
            f"  {m}: {r.estimate:.3f} [{r.low:.3f}, {r.high:.3f}], gap vs ell {r.gap:+.3f} "
            f"[{r.gap_low:+.3f}, {r.gap_high:+.3f}], gap vs layer {LEXICAL_LAYER} "
            f"{r.gap_layer0:+.3f} [{r.gap_layer0_low:+.3f}, {r.gap_layer0_high:+.3f}]"
        )
        if pv is not None:
            q = pv[(pv["position"] == pos) & (pv["layer"] == k) & (pv["metric"] == m)]
            if len(q):
                line += f", p {q.p.iloc[0]:.3f} (B {q.B.iloc[0]})"
        lines.append(line)
    return lines


def peak_report(ci, rob, pv, null) -> list:
    """Lines for the card: null range, CI and gap at the peak (the best layer: shown with p_corr,
    which corrects for picking it), misspelling change."""
    lines = []
    for m in ("acc_pair", "acc"):
        c = ci[(ci["metric"] == m) & (ci["position"] != BASELINE[0])]
        b = ci[(ci["metric"] == m) & (ci["position"] == BASELINE[0])].iloc[0]
        lines.append(
            f"{m}: baseline_ell {b.estimate:.3f} [{b.low:.3f}, {b.high:.3f}]"
            + _rob(rob, BASELINE, m)
        )
        for pos, d in c.groupby("position"):
            r = d.loc[d["estimate"].idxmax()]
            lex = d[d["layer"] == LEXICAL_LAYER]
            if len(lex):
                x = lex.iloc[0]
                lines.append(
                    f"  {pos} layer {LEXICAL_LAYER} (lexical): {x.estimate:.3f} "
                    f"[{x.low:.3f}, {x.high:.3f}]"
                )
            line = (
                f"  peak {pos} layer {int(r.layer)}: {r.estimate:.3f} [{r.low:.3f}, {r.high:.3f}],"
                f" gap vs ell {r.gap:+.3f} [{r.gap_low:+.3f}, {r.gap_high:+.3f}]"
            )
            if not np.isnan(r.gap_layer0):
                line += (
                    f", gap vs layer {LEXICAL_LAYER} {r.gap_layer0:+.3f} "
                    f"[{r.gap_layer0_low:+.3f}, {r.gap_layer0_high:+.3f}]"
                )
            if pv is not None:
                q = pv[(pv["position"] == pos) & (pv["layer"] == r.layer) & (pv["metric"] == m)]
                if len(q):
                    line += (
                        f", p {q.p.iloc[0]:.3f}, p_corr {q.p_corr.iloc[0]:.3f} (B {q.B.iloc[0]})"
                    )
            lines.append(line + _rob(rob, (pos, int(r.layer)), m))
            if null is not None and not null.empty:
                n = null[null["position"] == pos][m]
                lines.append(
                    f"    null range over all layers and seeds: {n.min():.3f}-{n.max():.3f}"
                )
    return lines


def _rob(rob, key, m) -> str:
    r = rob[(rob["position"] == key[0]) & (rob["layer"] == key[1]) & (rob["metric"] == m)]
    return f"; without misspellings {r.no_misspellings.iloc[0]:.3f} ({r.change.iloc[0]:+.4f})"


if __name__ == "__main__":
    main()
