"""Peak memory of one null fit (probing.controls.fit_task) on a real run, in copies of the layer.

The worker cap (probing.probe.n_workers) assumes FIT_COPIES float64 copies of the layer per
fit. On JUPITER, profession/race workers used ~3x what a laptop measures, so measure it where
it runs. Single-threaded, a few GB and a few minutes: fine on a login node.

Runs fit_layer as the null does (real labels flipped with seed 0, the real grid and inner CV)
but only for the first outer fold, which is where the peak is reached.

Usage (JUPITER login node, from your clone):
    source slurm/env.sh
    python scripts/measure_fit_memory.py --model qwen3-8b --bias profession [--layer 18]
"""

import argparse
import os
import resource
import time

for var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "FLEXIBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(var, "1")  # as the workers run (probing.probe.THREAD_VARS)

import numpy as np  # noqa: E402

from probing.controls import flip_within_pairs  # noqa: E402
from probing.io import load_layer, load_meta, run_paths  # noqa: E402
from probing.probe import FIT_COPIES, FIT_OVERHEAD, GRID, fit_layer, outer_folds  # noqa: E402


def peak_gb() -> float:
    """Peak RSS of this process so far (ru_maxrss: KiB on Linux, bytes on macOS)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r * 1024 / 1e9 if os.uname().sysname == "Linux" else r / 1e9


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--bias", required=True)
    ap.add_argument("--position", default="last_subtok")
    ap.add_argument("--layer", type=int, default=None, help="default: the middle layer")
    args = ap.parse_args()

    paths = run_paths(args.model, args.bias)
    meta = load_meta(paths)
    y, groups = meta["S"].to_numpy(), meta["target"].to_numpy()
    folds = outer_folds(meta)
    k = args.layer
    if k is None:
        k = len(list((paths.data / "acts" / args.position).glob("layer_*.npy"))) // 2
    base = peak_gb()
    print(f"after imports and meta: peak {base:.2f} GB")

    X = np.asarray(load_layer(paths, args.position, k), dtype=np.float64)
    x_gb = X.nbytes / 1e9
    print(f"{args.position} layer {k}: X {X.shape}, float64 {x_gb:.3f} GB; peak {peak_gb():.2f} GB")

    one_fold = np.where(folds == 0, 0, 1)  # fold 0 = the real first outer fold; 1 = the rest
    t = time.time()
    fit_layer(X, flip_within_pairs(y, 0), groups, one_fold, GRID)
    peak = peak_gb()
    copies = (peak - base) / x_gb
    est = FIT_COPIES * x_gb + FIT_OVERHEAD / 1e9
    print(
        f"fit (first outer fold + a small second): {time.time() - t:.0f} s\n"
        f"peak {peak:.2f} GB = {peak - base:.2f} GB above the start = {copies:.1f} copies of X\n"
        f"probing.probe estimates {est:.2f} GB per worker (FIT_COPIES = {FIT_COPIES})"
    )


if __name__ == "__main__":
    main()
