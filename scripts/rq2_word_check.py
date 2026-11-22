"""Which attribute words does the layer-0 (word identity) probe find easy? One run, CPU only.

Race: layer 0 alone reaches 0.80-0.82 pairwise accuracy on held-out targets, the deeper probe
only ~0.07 more. This looks at the run's out-of-fold predictions (folds are grouped by target,
so every prediction comes from a probe that never saw that sentence's target):

1. pairwise accuracy of ell, layer 0 and the fixed-depth probe, and a 2x2 of layer 0 vs the
   fixed depth (the pairs depth fixes, and those it breaks);
2. memorised words or a general feature: pairs whose words also occur under other targets
   (the probe can have learned them) vs words that occur under this target only;
3. the most frequent words on each side, how consistently they sit there, layer 0's score;
4. example pairs only the fixed depth gets right.

Inputs, from the run's data folder ({rq2_root}/{bias}/{model}/ on JUPITER), in --data:
    meta.csv, probes/baseline_ell.csv, probes/{position}/preds_layer_00.csv and
    probes/{position}/preds_layer_{fixed:02d}.csv (the fixed layer: results/rq2/all_runs.csv).

Writes results/rq2/word_check/{bias}_{model}.md and {bias}_{model}_words.csv.

Usage:
    # laptop, after copying the four files from the team folder into --data (same layout)
    uv run python scripts/rq2_word_check.py --model llama31-8b-instruct --bias race \\
        --data ~/rq2-preds/race/llama31-8b-instruct
    # JUPITER login node (reads the run's data folder directly)
    source slurm/env.sh && python scripts/rq2_word_check.py --model llama31-8b-instruct
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from probing.io import run_paths
from rmu.config import repo_path

OUT = "results/rq2/word_check"
N_WORDS = 25
N_EXAMPLES = 15


def fixed_layer_of(model, bias) -> int:
    runs = pd.read_csv(repo_path("results/rq2/all_runs.csv"))
    r = runs[(runs["model"] == model) & (runs["bias"] == bias)]
    if r.empty:
        raise ValueError(f"{model}/{bias} not in results/rq2/all_runs.csv; run rq2_report.py")
    return int(r["fixed_layer"].iloc[0])


def load(data: Path, position: str, k: int) -> pd.DataFrame:
    """Meta rows with the out-of-fold scores z of ell, layer 0 and layer k."""
    meta = pd.read_csv(data / "meta.csv", keep_default_na=False)
    files = {
        "z_ell": data / "probes" / "baseline_ell.csv",
        "z_0": data / "probes" / position / "preds_layer_00.csv",
        "z_k": data / "probes" / position / f"preds_layer_{k:02d}.csv",
    }
    for col, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"{path} missing (see the usage in this script's docstring)")
        d = pd.read_csv(path)
        if len(d) != len(meta) or not (d["row"].to_numpy() == meta["row"].to_numpy()).all():
            raise ValueError(f"{path}: rows do not match meta.csv")
        meta[col] = d["z"].to_numpy()
    meta["word"] = meta["word"].str.lower()
    return meta


def pairs_table(meta: pd.DataFrame) -> pd.DataFrame:
    """One row per pair: both words and, per score, whether the stereotype scored higher."""
    s = meta[meta["S"] == 1].set_index("item_id")
    a = meta[meta["S"] == 0].set_index("item_id")
    p = pd.DataFrame(
        {
            "target": s["target"],
            "word_stereo": s["word"],
            "word_anti": a["word"].reindex(s.index),
            "target_after_word": s["target_after_word"],
            "sentence_stereo": s["sentence"],
        }
    )
    for col in ("z_ell", "z_0", "z_k"):
        d = s[col] - a[col].reindex(s.index)
        p[col.replace("z_", "right_")] = np.where(d > 0, 1.0, np.where(d == 0, 0.5, 0.0))
        p[col.replace("z_", "margin_")] = d
    return p.reset_index()


def seen_elsewhere(p: pd.DataFrame, side: str) -> np.ndarray:
    """Whether the pair's ``side`` word occurs (either side) in a pair of another target."""
    long = pd.concat(
        [
            p[["target", "word_stereo"]].set_axis(["target", "w"], axis=1),
            p[["target", "word_anti"]].set_axis(["target", "w"], axis=1),
        ]
    )
    n_targets = long.groupby("w")["target"].nunique()
    return np.array([n_targets.get(w, 0) > 1 for w in p[f"word_{side}"]], dtype=bool)


def words_table(meta: pd.DataFrame) -> pd.DataFrame:
    g = meta.groupby("word")
    w = pd.DataFrame(
        {
            "n_stereo": g["S"].sum(),
            "n_anti": g["S"].count() - g["S"].sum(),
            "n_targets": g["target"].nunique(),
            "mean_z0": g["z_0"].mean(),
            "mean_zk": g["z_k"].mean(),
        }
    )
    w["n"] = w["n_stereo"] + w["n_anti"]
    w["stereo_share"] = w["n_stereo"] / w["n"]
    return w.sort_values("n", ascending=False)


def acc(x) -> str:
    return f"{np.mean(x):.3f} (n={len(x)})" if len(x) else "-"


def report(model, bias, position, k, p, w) -> str:
    L = [f"# Word check: {model} / {bias}", ""]
    L += [
        f"Out-of-fold pairwise accuracy (folds grouped by target), {len(p)} pairs, "
        f"{p['target'].nunique()} targets; probe position `{position}`, fixed layer {k}.",
        "",
        "| score | pairwise accuracy |",
        "|---|---|",
        f"| ell (predictability) | {acc(p['right_ell'])} |",
        f"| layer 0 (word identity) | {acc(p['right_0'])} |",
        f"| layer {k} (fixed depth) | {acc(p['right_k'])} |",
        "",
        f"## Layer 0 vs layer {k}",
        "",
        f"| | layer {k} right | layer {k} wrong |",
        "|---|---|---|",
    ]
    r0, rk = p["right_0"] >= 1, p["right_k"] >= 1
    L += [
        f"| layer 0 right | {(r0 & rk).sum()} | {(r0 & ~rk).sum()} |",
        f"| layer 0 wrong | {(~r0 & rk).sum()} | {(~r0 & ~rk).sum()} |",
        "",
        f"Depth fixes {(~r0 & rk).sum()} pairs and breaks {(r0 & ~rk).sum()}.",
        "",
        "## Memorised words or a general feature?",
        "",
        "A word that also occurs under other targets can have been learned by the probe "
        "(those targets were in its training folds). A word that occurs under this target only "
        "was never seen in training: accuracy above 0.5 on those pairs means layer 0 uses a "
        "feature of the word (e.g. valence), not the word itself.",
        "",
        "| words of the pair | pairs | layer 0 | layer k | ell |",
        "|---|---|---|---|---|",
    ]
    es, ea = seen_elsewhere(p, "stereo"), seen_elsewhere(p, "anti")
    for name, m in (
        ("both seen under other targets", es & ea),
        ("one seen under other targets", es ^ ea),
        ("neither: unique to this target", ~es & ~ea),
    ):
        d = p[m]
        L.append(
            f"| {name} | {len(d)} | {np.mean(d['right_0']):.3f} | {np.mean(d['right_k']):.3f} "
            f"| {np.mean(d['right_ell']):.3f} |"
            if len(d)
            else f"| {name} | 0 | - | - | - |"
        )
    L += ["", "By target position (target after the attribute word or before):", ""]
    L += ["| | pairs | layer 0 | layer k | ell |", "|---|---|---|---|---|"]
    for name, m in (
        ("target first", p["target_after_word"] == 0),
        ("target after", p["target_after_word"] == 1),
    ):
        d = p[m]
        L.append(
            f"| {name} | {len(d)} | {np.mean(d['right_0']):.3f} | {np.mean(d['right_k']):.3f} "
            f"| {np.mean(d['right_ell']):.3f} |"
        )

    def word_rows(d):
        out = [
            "| word | as stereo | as anti | targets | layer-0 z | layer-k z |",
            "|---|---|---|---|---|---|",
        ]
        for word, r in d.iterrows():
            out.append(
                f"| {word} | {int(r.n_stereo)} | {int(r.n_anti)} | {int(r.n_targets)} "
                f"| {r.mean_z0:+.2f} | {r.mean_zk:+.2f} |"
            )
        return out

    L += [
        "",
        f"## Most frequent words, by side (top {N_WORDS})",
        "",
        "Layer-0 z: the probe's mean out-of-fold score for the word's sentences (> 0: called "
        "stereotype). A word used only on one side and scored with that sign is what a "
        "word-identity probe can exploit.",
        "",
        "### Mostly the stereotype word (stereo share >= 0.5)",
        "",
        *word_rows(w[w["stereo_share"] >= 0.5].head(N_WORDS)),
        "",
        "### Mostly the anti-stereotype word (stereo share < 0.5)",
        "",
        *word_rows(w[w["stereo_share"] < 0.5].head(N_WORDS)),
        "",
        f"Words on both sides (each side at least twice): "
        f"{', '.join(w[(w['n_stereo'] >= 2) & (w['n_anti'] >= 2)].index[:30]) or 'none'}",
        "",
        f"## Pairs layer 0 gets wrong and layer {k} right (largest layer-{k} margin first)",
        "",
        "| target | stereotype word | anti word | sentence (stereotype) |",
        "|---|---|---|---|",
    ]
    fixed = p[~r0 & rk].sort_values("margin_k", ascending=False).head(N_EXAMPLES)
    for r in fixed.itertuples():
        L.append(f"| {r.target} | {r.word_stereo} | {r.word_anti} | {r.sentence_stereo} |")
    L += [
        "",
        "## Pairs both get wrong (largest layer-0 error first)",
        "",
        "| target | stereotype word | anti word | sentence (stereotype) |",
        "|---|---|---|---|",
    ]
    both = p[~r0 & ~rk].sort_values("margin_0").head(N_EXAMPLES)
    for r in both.itertuples():
        L.append(f"| {r.target} | {r.word_stereo} | {r.word_anti} | {r.sentence_stereo} |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--bias", default="race")
    ap.add_argument("--position", default="last_subtok")
    ap.add_argument("--layer", type=int, default=None, help="default: the run's fixed layer")
    ap.add_argument("--data", default=None, help="default: the run's data folder (JUPITER)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)

    k = args.layer if args.layer is not None else fixed_layer_of(args.model, args.bias)
    data = Path(args.data).expanduser() if args.data else run_paths(args.model, args.bias).data
    meta = load(data, args.position, k)
    p, w = pairs_table(meta), words_table(meta)
    out = repo_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.bias}_{args.model}"
    (out / f"{stem}.md").write_text(report(args.model, args.bias, args.position, k, p, w))
    w.round(4).to_csv(out / f"{stem}_words.csv")
    print(f"wrote {out / stem}.md and {stem}_words.csv")
    print(
        f"pairwise: ell {p['right_ell'].mean():.3f}, layer 0 {p['right_0'].mean():.3f}, "
        f"layer {k} {p['right_k'].mean():.3f}"
    )


if __name__ == "__main__":
    main()
