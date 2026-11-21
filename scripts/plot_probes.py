"""Basic plots of a probing.probe summary: accuracy by layer next to the ell baseline.

Writes three PNGs next to the summary (or to --out):

    accuracy_by_layer.png    per-sentence accuracy, pairwise accuracy, log-loss by layer
    accuracy_by_subset.png   the same two accuracies, target before vs after the word
    chosen_C.png             C chosen in the 5 outer folds, per layer and position

Usage:
    uv run python scripts/plot_probes.py --summary results/rq2-probing-nov21/summary.csv
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Reference palette (dataviz skill, light mode): categorical slots 1-2, validated.
SERIES = {"first_diff": "#2a78d6", "last_subtok": "#eb6834"}
LABELS = {"first_diff": "first token of the word", "last_subtok": "last token of the word"}
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
CHANCE = {"acc": 0.5, "acc_pair": 0.5, "logloss": float(np.log(2))}
TITLES = {
    "acc": "Per-sentence accuracy",
    "acc_pair": "Pairwise accuracy (stereotype scored higher)",
    "logloss": "Held-out log-loss (lower is better)",
}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "font.size": 10,
        "legend.frameon": False,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    }
)


def layer_axis(ax, layers, margin=7.0):
    ax.set_xlim(layers.min() - 0.5, layers.max() + margin)  # right margin for the baseline label
    step = 8 if layers.max() - layers.min() > 16 else 1
    ax.set_xticks(np.arange(layers.min(), layers.max() + 1, step))
    ax.set_xlabel("layer (0 = embeddings, 48 = after final norm)")


def draw_metric(ax, probes, base, col):
    """Two position lines, the ell baseline (value labelled in the right margin) and chance."""
    layers = probes["layer"].to_numpy()
    ax.axhline(CHANCE[col], color=MUTED, lw=0.8, zorder=1)
    ax.axhline(base[col], color=INK_2, lw=1.6, zorder=2)
    ax.annotate(
        f"ell {base[col]:.3f}",
        (layers.max() + 0.8, base[col]),
        xytext=(0, 3),
        textcoords="offset points",
        fontsize=8.5,
        color=INK_2,
        ha="left",
        va="bottom",
    )
    for pos, color in SERIES.items():
        d = probes[probes["position"] == pos].sort_values("layer")
        if d.empty or d[col].isna().all():
            continue
        ax.plot(d["layer"], d[col], color=color, lw=2, zorder=3, label=LABELS[pos])
    layer_axis(ax, layers)


def legend_top(fig, y=0.995):
    handles = [plt.Line2D([], [], color=c, lw=2) for c in SERIES.values()]
    handles += [plt.Line2D([], [], color=INK_2, lw=1.6), plt.Line2D([], [], color=MUTED, lw=0.8)]
    labels = [LABELS[p] for p in SERIES] + ["ell baseline (predictability)", "chance"]
    fig.legend(
        handles,
        labels,
        loc="upper left",
        ncol=4,
        bbox_to_anchor=(0.01, y),
        fontsize=9,
        handlelength=1.8,
        labelcolor=INK_2,
    )


def plot_overall(probes, base, model, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for ax, col in zip(axes, ("acc", "acc_pair", "logloss")):
        draw_metric(ax, probes, base, col)
        ax.set_title(TITLES[col])
    axes[0].set_ylabel("held-out, unseen targets")
    fig.text(
        0.01,
        0.995,
        f"{model}: probe P(S | activation) by layer, 5-fold CV grouped by target",
        ha="left",
        va="top",
        fontsize=10,
        color=INK_2,
    )
    legend_top(fig, y=0.95)
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.86))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_subsets(probes, base, model, out):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharey="row", constrained_layout=True)
    for j, (suffix, name) in enumerate(
        (("_tfirst", "target before the word"), ("_tafter", "target after the word"))
    ):
        n = int(base.get(f"n_pairs{suffix}", np.nan))
        for i, col in enumerate(("acc", "acc_pair")):
            ax = axes[i, j]
            sub = probes.rename(columns={f"{col}{suffix}": f"{col}_sub"})
            b = {f"{col}_sub": base[f"{col}{suffix}"]}
            CHANCE[f"{col}_sub"] = 0.5
            draw_metric(ax, sub, b, f"{col}_sub")
            ax.set_title(f"{TITLES[col].split(' (')[0]}: {name} ({n} pairs)")
        axes[0, 0].set_ylabel("per-sentence accuracy")
        axes[1, 0].set_ylabel("pairwise accuracy")
    legend_top(fig)
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_C(probes, grid_hint, out):
    folds = [c for c in probes.columns if c.startswith("C_f")]
    grid = sorted(set(np.concatenate([probes[f].to_numpy() for f in folds])) | set(grid_hint))
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True, constrained_layout=True)
    for ax, (pos, color) in zip(axes, SERIES.items()):
        d = probes[probes["position"] == pos].sort_values("layer")
        for _, r in d.iterrows():
            values, counts = np.unique(r[folds].to_numpy(dtype=float), return_counts=True)
            ax.scatter(
                np.full(len(values), r["layer"]),
                values,
                s=12 + 14 * counts,
                color=color,
                edgecolor=SURFACE,
                linewidth=1,
                zorder=3,
            )
        ax.set_yscale("log")
        ax.set_yticks(grid)
        ax.set_yticklabels([f"{g:g}" for g in grid])
        ax.minorticks_off()
        ax.set_title(f"C chosen per fold: {LABELS[pos]}")
        layer_axis(ax, d["layer"].to_numpy(), margin=1.0)
    axes[0].set_ylabel("C (larger = less regularised)")
    sizes = [plt.scatter([], [], s=12 + 14 * k, color=MUTED, edgecolor=SURFACE) for k in (1, 3, 5)]
    fig.legend(
        sizes,
        ["1 fold", "3 folds", "5 folds"],
        loc="upper right",
        ncol=3,
        fontsize=9,
        labelcolor=INK_2,
        bbox_to_anchor=(0.995, 0.995),
    )
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", default=None, help="folder for the PNGs (default: next to summary)")
    ap.add_argument("--model", default="lapa-12b-instruct", help="for the title")
    args = ap.parse_args()

    summary = pd.read_csv(args.summary)
    base = summary[summary["position"] == "baseline_ell"].iloc[0]
    probes = summary[summary["position"] != "baseline_ell"].copy()
    out = Path(args.out) if args.out else Path(args.summary).parent
    out.mkdir(parents=True, exist_ok=True)

    plot_overall(probes, base, args.model, out / "accuracy_by_layer.png")
    plot_subsets(probes, base, args.model, out / "accuracy_by_subset.png")
    plot_C(probes, (1e-4, 1e-3, 1e-2, 1e-1, 1.0), out / "chosen_C.png")
    print(f"wrote accuracy_by_layer.png, accuracy_by_subset.png, chosen_C.png to {out}")


if __name__ == "__main__":
    main()
