"""Fake RQ2 activations for testing the probing pipeline without a GPU.

Writes the real meta table (``probing.meta.build_meta``) and Gaussian float16 activations in
the same layout as the extraction step (``probing.io.run_paths``), for both positions.
Optionally plants a linear signal ``delta * (2S - 1) * u`` at some layers (u a fixed seeded
random unit vector), so a probe should find S there and nowhere else.

The noise is N(0, 1) per dimension, so the best possible accuracy along u is Phi(delta): about
54% at delta 0.1, which a 3840-dim probe on ~1.6k sentences cannot pick up. With GroupKFold(5)
by target, a scaled logistic probe gets ~0.67 at delta 1, ~0.92 at delta 2 and ~0.99 at delta 3
on planted layers, ~0.50 elsewhere and ~0.50 with labels flipped within pairs.

Usage:
    uv run python scripts/make_fake_acts.py --root /tmp/rq2-fake --signal 20,21,22 --delta 2
    # options: --model lapa-12b-instruct  --layers 0-48 (or 18-24, or 0,20,21)  --hidden 3840
    #          --bias gender  --seed 42  --root PATH (default: paths.rq2_root)
    #          --ell_delta 0 (fake ell ~ N(-10, 2), shifted by +ell_delta on stereotype rows)
"""

import argparse
import json

import numpy as np

from probing.io import POSITIONS, run_paths, save_layer, save_meta
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
    ap.add_argument("--bias", default="gender", help="StereoSet bias type (default: gender)")
    ap.add_argument("--layers", default=f"0-{N_HIDDEN_STATES - 1}", help="e.g. 0-48, 18-24, 0,20")
    ap.add_argument("--hidden", type=int, default=3840)
    ap.add_argument("--signal", default="", help="planted layers, e.g. 20,21,22 (default: none)")
    ap.add_argument("--delta", type=float, default=2.0, help="signal size along u")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--ell_delta", type=float, default=0.0, help="fake ell shift on S=1 rows (0: no signal)"
    )
    ap.add_argument("--root", default=None, help="default: paths.rq2_root")
    args = ap.parse_args()

    paths = run_paths(args.model, args.bias, root=args.root)
    meta = build_meta(args.model, args.bias)
    # Fake ell for the predictability baseline: N(-10, 2), +ell_delta on stereotype rows.
    ell = np.random.default_rng([args.seed, 999]).normal(-10.0, 2.0, len(meta))
    meta["ell"] = np.minimum(ell + args.ell_delta * meta["S"].to_numpy(), -1e-3)
    save_meta(meta, paths)
    labels = meta["S"].to_numpy().astype(np.float32)

    layers = parse_layers(args.layers)
    signal = set(parse_layers(args.signal))
    u = np.random.default_rng(args.seed).standard_normal(args.hidden).astype(np.float32)
    u /= np.linalg.norm(u)

    for position in POSITIONS:
        for k in layers:
            x = fake_layer(labels, args.hidden, k, args.seed, signal, args.delta, u, position)
            path = save_layer(x, paths, position, k)
        print(f"{position}: {len(layers)} layers, {x.shape} -> {path.parent}")
    # As probing.extract writes it; the probes read the number of layers from here.
    info = {"model": args.model, "bias": args.bias, "fake": True}
    info |= {"n_hidden_states": N_HIDDEN_STATES, "hidden_size": args.hidden}
    info["store_dtype"] = "float16"
    (paths.acts / "run.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    planted = sorted(signal & set(layers))
    print(f"meta: {paths.meta} ({len(meta)} rows)")
    print(f"planted layers: {planted or 'none'} (delta = {args.delta}, seed = {args.seed})")
    if signal - set(layers):
        print(f"note: --signal layers {sorted(signal - set(layers))} are not in --layers")


if __name__ == "__main__":
    main()
