"""Fake RQ2 activations for testing the probing pipeline without a GPU.

Writes the real meta table (``probing.meta.build_meta``) and Gaussian float16 activations in
the same layout as the extraction step, for both positions. Optionally plants a linear signal
``delta * (2S - 1) * u`` at some layers (u a fixed seeded random unit vector), so a probe should
find S there and nowhere else.

The noise is N(0, 1) per dimension, so the best possible accuracy along u is Phi(delta): about
54% at delta 0.1, which a 3840-dim probe on ~1.6k sentences cannot pick up. With GroupKFold(5)
by target, a scaled logistic probe gets ~0.67 at delta 1, ~0.92 at delta 2 and ~0.99 at delta 3
on planted layers, ~0.50 elsewhere and ~0.50 with labels flipped within pairs.

Usage:
    uv run python scripts/make_fake_acts.py --root /tmp/rq2-fake --signal 20,21,22 --delta 2
    # options: --model lapa-12b-instruct  --layers 0-48 (or 18-24, or 0,20,21)  --hidden 3840
    #          --bias_types gender  --seed 42  --root PATH (default: paths.rq2_root)
"""

import argparse

import numpy as np

from probing.io import POSITIONS, meta_path, save_layer, write_meta
from probing.meta import build_meta

N_HIDDEN_STATES = 49  # lapa-12b-instruct: embeddings + 48 decoder layers


def parse_layers(text):
    """'0-48' / '18-24' / '0,20,21' / '0-3,40' -> sorted list of ints."""
    layers = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            layers.update(range(int(a), int(b) + 1))
        else:
            layers.add(int(part))
    return sorted(layers)


def fake_layer(labels, hidden, k, seed, signal_layers, delta, u, position):
    """N(0, 1) activations; plus ``delta * (2S - 1) * u`` if ``k`` is a planted layer."""
    rng = np.random.default_rng([seed, k, POSITIONS.index(position)])
    x = rng.standard_normal((len(labels), hidden), dtype=np.float32)
    if k in signal_layers:
        x += delta * (2 * labels[:, None] - 1) * u[None, :]
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="lapa-12b-instruct")
    ap.add_argument("--bias_types", default="gender", help="comma-separated")
    ap.add_argument("--layers", default=f"0-{N_HIDDEN_STATES - 1}", help="e.g. 0-48, 18-24, 0,20")
    ap.add_argument("--hidden", type=int, default=3840)
    ap.add_argument("--signal", default="", help="planted layers, e.g. 20,21,22 (default: none)")
    ap.add_argument("--delta", type=float, default=2.0, help="signal size along u")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--root", default=None, help="default: paths.rq2_root")
    args = ap.parse_args()

    meta = build_meta(args.model, [b for b in args.bias_types.split(",") if b])
    write_meta(meta, meta_path(args.model, args.root))
    labels = meta["S"].to_numpy().astype(np.float32)

    layers = parse_layers(args.layers)
    signal = set(parse_layers(args.signal))
    u = np.random.default_rng(args.seed).standard_normal(args.hidden).astype(np.float32)
    u /= np.linalg.norm(u)

    for position in POSITIONS:
        for k in layers:
            x = fake_layer(labels, args.hidden, k, args.seed, signal, args.delta, u, position)
            path = save_layer(x, args.model, position, k, args.root)
        print(f"{position}: {len(layers)} layers, {x.shape} -> {path.parent}")
    planted = sorted(signal & set(layers))
    print(f"meta: {meta_path(args.model, args.root)} ({len(meta)} rows)")
    print(f"planted layers: {planted or 'none'} (delta = {args.delta}, seed = {args.seed})")
    if signal - set(layers):
        print(f"note: --signal layers {sorted(signal - set(layers))} are not in --layers")


if __name__ == "__main__":
    main()
