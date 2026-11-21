"""Sanity check of the probing setup on fake activations (from scripts/make_fake_acts.py).

Per layer and position: StandardScaler + LogisticRegression, mean accuracy over GroupKFold
grouped by StereoSet target, once with the real labels and once with labels swapped within a
random half of the pairs (rows 2i, 2i+1). Expected: ~0.50 on unplanted layers and with flipped
labels, clearly above 0.50 on planted layers (with N(0, 1) noise, at most Phi(delta)). With 40
targets, accuracies within about 0.47-0.53 are chance.

This is a check of the data layout and CV wiring, not the RQ2 probe: C is fixed (no inner CV).

Usage:
    uv run python scripts/make_fake_acts.py --root /tmp/rq2-fake --layers 19-23 --signal 20,21,22
    uv run python scripts/check_fake_probe.py --root /tmp/rq2-fake
    # options: --model lapa-12b-instruct  --bias gender
    #          --layers 19,21,23 (default: every layer file found)
    #          --positions first_diff,last_subtok  --n_splits 5  --seed 0
"""

import argparse
import re

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from probing.controls import flip_within_pairs
from probing.io import POSITIONS, load_layer, load_meta, run_paths


def probe_accuracy(X, y, groups, n_splits=5):
    """Mean accuracy over GroupKFold folds of a scaled logistic probe."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cv = GroupKFold(n_splits=n_splits)
    return cross_val_score(clf, X, y, groups=groups, cv=cv).mean()


def layers_on_disk(paths, position):
    folder = paths.acts / position
    found = (re.fullmatch(r"layer_(\d+)\.npy", p.name) for p in folder.glob("layer_*.npy"))
    return sorted(int(m.group(1)) for m in found if m)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=None, help="default: paths.rq2_root")
    ap.add_argument("--model", default="lapa-12b-instruct")
    ap.add_argument("--bias", default="gender", help="StereoSet bias type (default: gender)")
    ap.add_argument("--layers", default=None, help="comma-separated (default: all found)")
    ap.add_argument("--positions", default=",".join(POSITIONS), help="comma-separated")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0, help="seed of the within-pair flips")
    args = ap.parse_args()

    paths = run_paths(args.model, args.bias, root=args.root)
    meta = load_meta(paths)
    y = meta["S"].to_numpy()
    y_null = flip_within_pairs(y, args.seed)
    groups = meta["target"].to_numpy()
    print(f"{len(meta)} rows, {len(np.unique(groups))} targets, GroupKFold({args.n_splits})")

    print("position     layer  acc    acc_flipped")
    for position in [p for p in args.positions.split(",") if p]:
        if args.layers:
            layers = [int(k) for k in args.layers.split(",") if k]
        else:
            layers = layers_on_disk(paths, position)
        if not layers:
            print(f"{position:<12} (no layer files)")
        for k in layers:
            X = np.asarray(load_layer(paths, position, k), dtype=np.float32)
            if X.shape[0] != len(meta):
                raise ValueError(f"{position} layer {k}: {X.shape[0]} rows, meta has {len(meta)}")
            acc = probe_accuracy(X, y, groups, args.n_splits)
            acc0 = probe_accuracy(X, y_null, groups, args.n_splits)
            print(f"{position:<12} {k:5d}  {acc:.3f}  {acc0:.3f}")


if __name__ == "__main__":
    main()
