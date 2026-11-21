"""RQ2 report over every finished run in ``results/rq2/{bias}/{model}/``.

A run is finished once ``probing.controls stats`` has written its ``ci.csv``. Writes, in
``results/rq2/`` (or ``--results_root``)::

    all_runs.csv             every (position, layer, metric) row of every run: ci.csv joined with
                             pvalues.csv and summary.csv, plus model, bias and run-level columns
                             (n_layers, n_pairs, n_targets, fixed layer, gate status). The rest
                             is made from this file only.
    report_table.csv / .tex  one row per (model, bias), see ``report_table``
    fig_gap_by_depth.png/pdf small multiples: bias type x model, gap probe - ell with its CI
                             against relative depth, the fixed depth as a vertical line
    fig_forest.png/pdf       the gaps vs ell and vs layer 0 at the fixed depth, one row per run

Gaps, not raw accuracies, are compared across runs: every run has its own baseline. All
numbers are for one metric (default: pairwise accuracy) at the fixed depth and position of
``config/rq2/report.yaml``; the best layer is reported too, marked post hoc, with p_corr.

Usage:
    uv run python scripts/rq2_report.py [--metric acc_pair] [--results_root results/rq2]
"""

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from plot_probes import GRID, INK, INK_2, MUTED, SERIES, SURFACE  # noqa: E402

from probing.gates import gate_passed, read_gate  # noqa: E402
from probing.io import BIAS_TYPES, RESULTS_ROOT, fixed_layer, report_config  # noqa: E402
from rmu.config import load_config, repo_path  # noqa: E402

BASELINE = "baseline_ell"
# Colors by comparison (validated pair, dataviz reference palette slots 1-2).
VS_ELL, VS_LAYER0 = SERIES["first_diff"], SERIES["last_subtok"]
METRIC_NAMES = {"acc_pair": "pairwise accuracy", "acc": "per-sentence accuracy"}
RUN_COLUMNS = [
    "model",
    "bias",
    "n_layers",
    "n_pairs",
    "n_targets",
    "fixed_layer",
    "fixed_position",
    "depth_chosen_here",
    "gate1",
    "gate2",
]


# --------------------------------------------------------------------------- collect


def find_runs(results_root) -> list:
    """[(bias, model, folder)] of the finished runs (with ci.csv), in bias-type order."""
    root = Path(results_root)
    runs = []
    for bias in BIAS_TYPES:
        for folder in sorted((root / bias).glob("*/")):
            if (folder / "ci.csv").exists():
                runs.append((bias, folder.name, folder))
    return runs


def gate_status(path) -> str:
    """PASS, FAIL (k of n), open (k of n) or missing."""
    path = Path(path)
    if not path.exists():
        return "missing"
    ok, _ = gate_passed(path)
    if ok:
        return "PASS"
    checks = read_gate(path)
    failed = sum(c is False for c, _ in checks)
    if failed:
        return f"FAIL ({failed} of {len(checks)})"
    return f"open ({sum(c is None for c, _ in checks)} of {len(checks)})"


def run_info(folder, ci, summary) -> dict:
    """Run-level numbers from run.json, with fallbacks for runs from before they were recorded:
    n_layers from the largest layer, n_targets from the committed label table (labels.csv)."""
    run = json.loads((folder / "run.json").read_text()) if (folder / "run.json").exists() else {}
    steps = run.get("steps", {})
    stats, probe = steps.get("controls stats", {}), steps.get("probe", {})
    n_layers = stats.get("n_layers") or probe.get("n_layers") or int(ci["layer"].max())
    n_targets = stats.get("n_targets")
    if n_targets is None and (folder / "labels.csv").exists():
        n_targets = pd.read_csv(folder / "labels.csv", usecols=["target"])["target"].nunique()
    n_pairs = stats.get("n_pairs")
    if n_pairs is None and summary is not None:
        n_pairs = int(summary.loc[summary["position"] == BASELINE, "n_pairs"].iloc[0])
    return {
        "n_layers": int(n_layers),
        "n_pairs": n_pairs,
        "n_targets": n_targets,
        "gate1": gate_status(folder / "gate1.txt"),
        "gate2": gate_status(folder / "gate2.txt"),
    }


def load_run(bias, model, folder, cfg) -> pd.DataFrame:
    """One run as long rows (position, layer, metric): ci.csv + pvalues.csv + summary.csv."""
    ci = pd.read_csv(folder / "ci.csv")
    keys = ["position", "layer", "metric"]
    pv_path = folder / "pvalues.csv"
    if pv_path.exists():
        pv = pd.read_csv(pv_path).drop(columns=["real"])  # = ci's estimate
        ci = ci.merge(pv, on=keys, how="left", validate="one_to_one")
    summary = pd.read_csv(folder / "summary.csv") if (folder / "summary.csv").exists() else None
    if summary is not None:
        # Per-probe columns (chosen C, edge, warnings), and n_pairs of the metric's subset.
        extra = [c for c in summary.columns if c.startswith("C_") or c == "n_conv_warnings"]
        ci = ci.merge(summary[["position", "layer", *extra]], on=["position", "layer"], how="left")
        long = summary.melt(["position", "layer"], var_name="col", value_name="value")
        n = long[long["col"].str.startswith("n_pairs")].copy()
        n["metric_subset"] = n["col"].str.removeprefix("n_pairs")
        sub = ci["metric"].str.extract(r"(_tfirst|_tafter)$")[0].fillna("")
        lookup = n.set_index(["position", "layer", "metric_subset"])["value"]
        ci["n_pairs_subset"] = [
            lookup.get((p, k, s), np.nan) for p, k, s in zip(ci["position"], ci["layer"], sub)
        ]
        # summary.csv and ci.csv come from the same predictions: they must agree.
        for m in ("acc", "acc_pair", "logloss"):
            s = summary.set_index(["position", "layer"])[m]
            c = ci[ci["metric"] == m].set_index(["position", "layer"])["estimate"]
            if not np.allclose(s.reindex(c.index), c, atol=1e-9, equal_nan=True):
                raise ValueError(f"{folder}: summary.csv and ci.csv disagree on {m}")
    info = run_info(folder, ci, summary)
    fd = cfg["fixed_depth"]
    run = {
        "model": model,
        "bias": bias,
        **info,
        "fixed_layer": fixed_layer(info["n_layers"], cfg),
        "fixed_position": fd["position"],
        "depth_chosen_here": (fd["chosen_on"]["model"], fd["chosen_on"]["bias"]) == (model, bias),
    }
    for col in reversed(RUN_COLUMNS):
        ci.insert(0, col, run[col])
    return ci


def collect(results_root, cfg=None) -> pd.DataFrame:
    cfg = cfg or report_config()
    runs = find_runs(results_root)
    if not runs:
        raise FileNotFoundError(f"no finished run (ci.csv) under {results_root}/<bias>/<model>/")
    return pd.concat([load_run(b, m, f, cfg) for b, m, f in runs], ignore_index=True)


# --------------------------------------------------------------------------- table


def _row(d, position, layer, metric):
    r = d[(d["position"] == position) & (d["layer"] == layer) & (d["metric"] == metric)]
    return r.iloc[0] if len(r) else None


def report_table(all_runs, metric="acc_pair", cfg=None) -> pd.DataFrame:
    """One row per (model, bias): baselines, the probe at the fixed depth with both gaps, the
    split by target position, the best layer (post hoc), gates and flags."""
    cfg = cfg or report_config()
    few = cfg.get("few_targets_below", 20)
    rows = []
    for (bias, model), d in all_runs.groupby(["bias", "model"], sort=False):
        run = d.iloc[0]
        pos, k = run["fixed_position"], int(run["fixed_layer"])
        ell, l0, fx = (
            _row(d, BASELINE, -1, metric),
            _row(d, pos, 0, metric),
            _row(d, pos, k, metric),
        )
        probes = d[(d["position"] != BASELINE) & (d["metric"] == metric)]
        best = probes.loc[probes["estimate"].idxmax()]
        row = {
            "model": model,
            "bias": bias,
            "metric": metric,
            "n_pairs": run["n_pairs"],
            "n_targets": run["n_targets"],
            "few_targets": bool(pd.notna(run["n_targets"]) and run["n_targets"] < few),
            "ell": ell["estimate"],
            "ell_low": ell["low"],
            "ell_high": ell["high"],
            "layer0": l0["estimate"] if l0 is not None else np.nan,
            "layer0_low": l0["low"] if l0 is not None else np.nan,
            "layer0_high": l0["high"] if l0 is not None else np.nan,
            "position": pos,
            "fixed_layer": k,
            "n_layers": run["n_layers"],
            "fixed_rel_depth": k / run["n_layers"],
            "depth_chosen_here": run["depth_chosen_here"],
        }
        for col in ("estimate", "low", "high", "gap", "gap_low", "gap_high"):
            row[f"fixed_{col}"] = fx[col] if fx is not None else np.nan
        for col in ("gap_layer0", "gap_layer0_low", "gap_layer0_high"):
            row[f"fixed_{col}"] = fx[col] if fx is not None else np.nan
        row["fixed_p"] = fx["p"] if fx is not None and "p" in fx else np.nan
        for sub in ("_tfirst", "_tafter"):
            s = _row(d, pos, k, metric + sub)
            for col in ("gap", "gap_low", "gap_high"):
                row[f"{col}{sub}"] = s[col] if s is not None else np.nan
            row[f"n_pairs{sub}"] = s["n_pairs_subset"] if s is not None else np.nan
        row |= {
            "best_position_post_hoc": best["position"],
            "best_layer_post_hoc": int(best["layer"]),
            "best_rel_depth_post_hoc": best["rel_depth"],
            "best_estimate_post_hoc": best["estimate"],
            "best_low_post_hoc": best["low"],
            "best_high_post_hoc": best["high"],
            "best_p_corr_post_hoc": best["p_corr"] if "p_corr" in best else np.nan,
            "B": best["B"] if "B" in best else np.nan,
            "gate1": run["gate1"],
            "gate2": run["gate2"],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _ci(est, low, high, signed=False, note="") -> str:
    """LaTeX cell: the estimate, its CI below in small type (a nested tabular, no packages)."""
    if pd.isna(est):
        return "--"
    f = "{:+.3f}" if signed else "{:.3f}"
    return (
        r"\begin{tabular}[t]{@{}r@{}}"
        + f"{f.format(est)}{note}"
        + r"\\ \scriptsize "
        + f"[{f.format(low)}, {f.format(high)}]"
        + r"\end{tabular}"
    )


def to_latex(table) -> str:
    """booktabs table for the paper (one row per run)."""
    metric = METRIC_NAMES.get(table["metric"].iloc[0], table["metric"].iloc[0])
    head = [
        "Model",
        "Bias",
        "Pairs / targets",
        r"\texttt{ell}",
        "Layer 0",
        "Fixed depth",
        r"Gap vs \texttt{ell}",
        "Gap vs layer 0",
        "Gap, target first",
        "Gap, target after",
        "Best layer (post hoc)",
        "Gates",
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{ll" + "r" * (len(head) - 2) + "}",
        r"\toprule",
        " & ".join(head) + r" \\",
        r"\midrule",
    ]
    for r in table.itertuples():
        mark = r"$^\dagger$" if r.depth_chosen_here else ""
        few = r"$^\ast$" if r.few_targets else ""
        p = (
            ""
            if pd.isna(r.best_p_corr_post_hoc)
            else f", $p_{{corr}}$ {r.best_p_corr_post_hoc:.3f}"
        )
        cells = [
            r.model.replace("_", r"\_") + mark,
            r.bias,
            f"{int(r.n_pairs)} / {int(r.n_targets) if pd.notna(r.n_targets) else '?'}{few}",
            _ci(r.ell, r.ell_low, r.ell_high),
            _ci(r.layer0, r.layer0_low, r.layer0_high),
            _ci(r.fixed_estimate, r.fixed_low, r.fixed_high, note=f" (L{r.fixed_layer})"),
            _ci(r.fixed_gap, r.fixed_gap_low, r.fixed_gap_high, signed=True),
            _ci(r.fixed_gap_layer0, r.fixed_gap_layer0_low, r.fixed_gap_layer0_high, signed=True),
            _ci(r.gap_tfirst, r.gap_low_tfirst, r.gap_high_tfirst, signed=True),
            _ci(r.gap_tafter, r.gap_low_tafter, r.gap_high_tafter, signed=True),
            r"\begin{tabular}[t]{@{}r@{}}"
            + f"{r.best_estimate_post_hoc:.3f} at {r.best_rel_depth_post_hoc:.2f}"
            + r"\\ \scriptsize "
            + p.removeprefix(", ")
            + r"\end{tabular}",
            f"{r.gate1} / {r.gate2}",
        ]
        lines.append(" & ".join(cells) + r" \\")
    fd = report_config()["fixed_depth"]
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{RQ2 linear probes, " + metric + r" on held-out targets, 95\% bootstrap CIs "
        r"over targets. Fixed depth: " + f"{fd['rel_depth']:.3f}" + r" of the model's layers ("
        r"\texttt{" + fd["position"].replace("_", r"\_") + r"}), fixed on " + fd["date"] + r" "
        r"before other runs were looked at; $^\dagger$ the run it was chosen on. Gaps are probe "
        r"minus baseline at that depth; target first/after: pairs where the target comes "
        r"before/after the attribute word (gap vs \texttt{ell}). Best layer: post hoc, as "
        r"relative depth, with the max-over-layers corrected $p$. $^\ast$ few targets "
        r"(coarse CIs).}",
        r"\label{tab:rq2}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- figures


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _model_order(models) -> list:
    """Models in config/default.yaml order (unknown ones last, alphabetically)."""
    known = list((load_config().get("models") or {}).keys())
    return sorted(models, key=lambda m: (known.index(m) if m in known else len(known), m))


def fig_gap_by_depth(all_runs, out, metric="acc_pair", cfg=None):
    """Rows = bias type, columns = model; gap probe - ell with its CI against relative depth
    at the fixed position, shared y-axis, the fixed depth as a vertical line."""
    cfg = cfg or report_config()
    fd = cfg["fixed_depth"]
    biases = [b for b in BIAS_TYPES if b in set(all_runs["bias"])]
    models = _model_order(set(all_runs["model"]))
    d = all_runs[(all_runs["metric"] == metric) & (all_runs["position"] == fd["position"])]
    fig, axes = plt.subplots(
        len(biases),
        len(models),
        figsize=(1.0 + 3.2 * len(models), 0.9 + 2.4 * len(biases)),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    fig.set_facecolor(SURFACE)
    for i, bias in enumerate(biases):
        for j, model in enumerate(models):
            ax = axes[i, j]
            _style(ax)
            r = d[(d["bias"] == bias) & (d["model"] == model)].sort_values("rel_depth")
            if i == 0:
                ax.set_title(model, loc="left", fontsize=10, color=INK, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{bias}\ngap vs ell", color=INK_2, fontsize=9)
            if r.empty:
                ax.text(
                    0.5,
                    0.5,
                    "no run yet",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color=MUTED,
                    fontsize=9,
                )
                continue
            ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
            ax.fill_between(
                r["rel_depth"], r["gap_low"], r["gap_high"], color=VS_ELL, alpha=0.2, lw=0, zorder=2
            )
            ax.plot(r["rel_depth"], r["gap"], color=VS_ELL, lw=2, zorder=3)
            ax.axvline(fd["rel_depth"], color=INK_2, lw=1, ls=(0, (3, 3)), zorder=2)
            run = r.iloc[0]
            notes = []
            if run["depth_chosen_here"]:
                notes.append("fixed depth chosen on this run")
            if pd.notna(run["n_targets"]) and run["n_targets"] < cfg.get("few_targets_below", 20):
                notes.append(f"only {int(run['n_targets'])} targets")
            notes += [f"{g} {run[g]}" for g in ("gate1", "gate2") if run[g] != "PASS"]
            if notes:
                ax.text(
                    0.98,
                    0.04,
                    "; ".join(notes),
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    color=MUTED,
                    fontsize=7.5,
                )
    for ax in axes[-1]:
        ax.set_xlabel("relative depth (layer / n_layers)", color=INK_2, fontsize=9)
        ax.set_xlim(0, 1)
    title = (
        f"Probe minus ell baseline, {METRIC_NAMES.get(metric, metric)}, {fd['position']}. "
        f"Band: 95% CI over targets; dashed: fixed depth {fd['rel_depth']:.3f}."
    )
    chars = int(fig.get_size_inches()[0] * 13)  # ~13 characters per inch at 9 pt
    fig.suptitle(textwrap.fill(title, chars), x=0.01, ha="left", fontsize=9, color=INK_2)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_forest(table, out):
    """One row per run: the gaps vs ell and vs layer 0 at the fixed depth, with CIs; 0 marked."""
    t = table.copy()
    t["bias_rank"] = t["bias"].map({b: i for i, b in enumerate(BIAS_TYPES)})
    order = _model_order(set(t["model"]))
    t["model_rank"] = t["model"].map({m: i for i, m in enumerate(order)})
    t = t.sort_values(["bias_rank", "model_rank"]).reset_index(drop=True)
    n = len(t)
    fig, ax = plt.subplots(figsize=(7.2, 1.3 + 0.42 * n), constrained_layout=True)
    fig.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(True, axis="x", color=GRID, lw=0.8)
    ax.grid(False, axis="y")
    y = np.arange(n)[::-1].astype(float)
    ax.axvline(0, color=INK_2, lw=1, zorder=1)
    for col, color, dy, label in (
        ("fixed_gap", VS_ELL, 0.12, "vs ell (predictability)"),
        ("fixed_gap_layer0", VS_LAYER0, -0.12, "vs layer 0 (word identity)"),
    ):
        ax.errorbar(
            t[col],
            y + dy,
            xerr=[t[col] - t[f"{col}_low"], t[f"{col}_high"] - t[col]],
            fmt="o",
            ms=6,
            color=color,
            mec=SURFACE,
            mew=1.5,
            elinewidth=2,
            capsize=0,
            label=label,
            zorder=3,
        )
    labels = []
    for r in t.itertuples():
        tag = f"{r.model}, {r.bias}"
        if r.depth_chosen_here:
            tag += " †"
        if r.few_targets:
            tag += f" ({int(r.n_targets)} targets)"
        gates = [f"{g} {getattr(r, g)}" for g in ("gate1", "gate2") if getattr(r, g) != "PASS"]
        if gates:
            tag += f" [{'; '.join(gates)}]"
        labels.append(tag)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK, fontsize=9)
    ax.set_ylim(-0.7, n - 0.3)
    metric = METRIC_NAMES.get(t["metric"].iloc[0], t["metric"].iloc[0])
    ax.set_xlabel(f"probe minus baseline, {metric} (95% CI over targets)", color=INK_2, fontsize=9)
    fd = report_config()["fixed_depth"]
    ax.set_title(
        f"At the fixed depth {fd['rel_depth']:.3f} ({fd['position']})"
        + ("; † depth chosen on this run" if t["depth_chosen_here"].any() else ""),
        loc="left",
        fontsize=9,
        color=INK_2,
        fontweight="normal",
        pad=24,
    )
    # Above the plot, under the title: never on top of an interval.
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=2,
        frameon=False,
        fontsize=8.5,
        labelcolor=INK_2,
        borderaxespad=0.2,
    )
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- main


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results_root", default=RESULTS_ROOT)
    ap.add_argument("--metric", default="acc_pair", choices=sorted(METRIC_NAMES))
    args = ap.parse_args(argv)

    root = repo_path(args.results_root)
    cfg = report_config()
    all_runs = collect(root, cfg)
    all_runs.to_csv(root / "all_runs.csv", index=False)
    table = report_table(all_runs, args.metric, cfg)
    table.to_csv(root / "report_table.csv", index=False)
    (root / "report_table.tex").write_text(to_latex(table), encoding="utf-8")
    fig_gap_by_depth(all_runs, root / "fig_gap_by_depth", args.metric, cfg)
    fig_forest(table, root / "fig_forest")

    runs = table[["model", "bias"]].apply(" / ".join, axis=1).tolist()
    print(f"{len(runs)} runs: {', '.join(runs)}")
    print(
        f"wrote all_runs.csv ({len(all_runs)} rows), report_table.csv/.tex, "
        f"fig_gap_by_depth.png/pdf, fig_forest.png/pdf in {root}"
    )


if __name__ == "__main__":
    main()
