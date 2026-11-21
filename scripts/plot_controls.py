"""Plot of the WP6 controls: the real probe curve with its bootstrap CI, the null band and the
ell baseline, per position; plus the gap probe - ell with its paired CI.

Reads ``ci.csv`` and (if present) ``null_scores.csv`` from the run's results folder
(``results/rq2/{bias}/{model}``, written by ``python -m probing.controls stats`` / ``null``) and
writes ``controls_by_layer.png`` there.

Usage:
    uv run python scripts/plot_controls.py --model lapa-12b-instruct --bias gender
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from plot_probes import (  # noqa: E402
    GRID,
    INK_2,
    LABELS,
    MUTED,
    SERIES,
    layer_axis,
    probe_settings,
)

from probing.io import run_paths  # noqa: E402

NULL_FILL = "#d9d8d3"
TITLES = {
    "acc_pair": "Pairwise accuracy",
    "acc": "Per-sentence accuracy",
    "gap_acc_pair": "Pairwise accuracy: probe - ell baseline",
}


def panel(ax, ci, null, pos, metric):
    color = SERIES[pos]
    base = ci[(ci["position"] == "baseline_ell") & (ci["metric"] == metric)].iloc[0]
    d = ci[(ci["position"] == pos) & (ci["metric"] == metric)].sort_values("layer")
    layers = d["layer"].to_numpy()
    ax.axhline(0.5, color=MUTED, lw=0.8, zorder=1)
    if null is not None:
        n = null[null["position"] == pos].groupby("layer")[metric].agg(["min", "max"])
        ax.fill_between(n.index, n["min"], n["max"], color=NULL_FILL, lw=0, zorder=2)
    ax.axhspan(base["low"], base["high"], color=INK_2, alpha=0.08, lw=0, zorder=2)
    ax.axhline(base["estimate"], color=INK_2, lw=1.6, zorder=3)
    ax.annotate(
        f"ell {base['estimate']:.3f}",
        (layers.max() + 0.8, base["estimate"]),
        xytext=(0, 3),
        textcoords="offset points",
        fontsize=8.5,
        color=INK_2,
        va="bottom",
    )
    ax.fill_between(layers, d["low"], d["high"], color=color, alpha=0.2, lw=0, zorder=4)
    ax.plot(layers, d["estimate"], color=color, lw=2, zorder=5)
    layer_axis(ax, layers)


def gap_panel(ax, ci, pos, metric="acc_pair"):
    d = ci[(ci["position"] == pos) & (ci["metric"] == metric)].sort_values("layer")
    layers = d["layer"].to_numpy()
    ax.axhline(0, color=INK_2, lw=1.2, zorder=1)
    ax.fill_between(layers, d["gap_low"], d["gap_high"], color=SERIES[pos], alpha=0.2, lw=0)
    ax.plot(layers, d["gap"], color=SERIES[pos], lw=2, zorder=3)
    layer_axis(ax, layers, margin=1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="lapa-12b-instruct")
    ap.add_argument("--bias", default="gender", help="StereoSet bias type (default: gender)")
    ap.add_argument("--root", default=None, help="data root, as given to probing.controls")
    args = ap.parse_args()

    paths = run_paths(args.model, args.bias, root=args.root)
    results = paths.results
    ci = pd.read_csv(results / "ci.csv")
    probe_settings(paths, ci[ci["layer"] >= 0])
    null_path = results / "null_scores.csv"
    null = pd.read_csv(null_path) if null_path.exists() else None
    n_seeds = 0 if null is None else null["seed"].nunique()
    R = int(ci["R"].iloc[0])
    positions = [p for p in SERIES if p in set(ci["position"])]

    fig, axes = plt.subplots(
        3,
        len(positions),
        figsize=(6 * len(positions), 10),
        sharey="row",
        squeeze=False,
        constrained_layout=True,
    )
    for j, pos in enumerate(positions):
        for i, metric in enumerate(("acc_pair", "acc")):
            panel(axes[i, j], ci, null, pos, metric)
            axes[i, j].set_title(f"{TITLES[metric]}: {LABELS[pos]}")
        gap_panel(axes[2, j], ci, pos)
        axes[2, j].set_title(f"{TITLES['gap_acc_pair']}: {LABELS[pos]}")
    axes[0, 0].set_ylabel("pairwise accuracy")
    axes[1, 0].set_ylabel("per-sentence accuracy")
    axes[2, 0].set_ylabel("probe - ell (difference in accuracy)")

    handles = [plt.Line2D([], [], color=SERIES[p], lw=2) for p in positions]
    labels = [f"probe, {LABELS[p]}" for p in positions]
    handles += [
        plt.Rectangle((0, 0), 1, 1, color=SERIES[positions[0]], alpha=0.2, lw=0),
        plt.Rectangle((0, 0), 1, 1, color=NULL_FILL, lw=0),
        plt.Line2D([], [], color=INK_2, lw=1.6),
        plt.Line2D([], [], color=MUTED, lw=0.8),
    ]
    labels += [
        f"95% CI, {R} resamples of targets (probes fixed)",
        f"shuffled labels, min-max over {n_seeds} seeds" if n_seeds else "no null yet",
        "ell baseline (band: its 95% CI)",
        "chance",
    ]
    fig.legend(
        handles,
        labels,
        loc="upper left",
        ncol=3,
        bbox_to_anchor=(0.01, 0.995),
        fontsize=9,
        labelcolor=INK_2,
    )
    fig.text(
        0.99, 0.995, f"{args.model}, {args.bias}", ha="right", va="top", fontsize=9, color=INK_2
    )
    for ax in axes.ravel():
        ax.grid(True, axis="y", color=GRID)
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.94))
    out = results / "controls_by_layer.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
