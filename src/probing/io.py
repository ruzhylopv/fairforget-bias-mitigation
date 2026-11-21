"""File layout of the RQ2 probing data: activations (X) and the metadata table (y, groups, ...).

Everything lives under ``paths.rq2_root`` (config/default.yaml)::

    {rq2_root}/acts/{model}/{position}/layer_{kk:02d}.npy   float16 (or float32), (N, hidden)
    {rq2_root}/meta/{model}.csv                              one row per sentence

``kk`` indexes ``hidden_states`` (00 = embeddings). Row i of the meta CSV is row i of every
layer file. Rows come pair by pair, stereotype first, so rows 2i and 2i+1 are one StereoSet item.
Every path function takes an optional ``root`` that replaces ``rq2_root()`` (tests, fake data).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from rmu.config import get

# Token positions with activations on disk, and the meta column holding each one's index.
# Indices are negative (counted from the end), so they work on left-padded batches unchanged.
POSITIONS = ("first_diff", "last_subtok")
POSITION_COLUMNS = {"first_diff": "idx_first_diff", "last_subtok": "idx_last_subtok"}

META_COLUMNS = [
    "row",
    "sentence_id",
    "item_id",
    "split",
    "bias_type",
    "target",
    "S",
    "sentence",
    "word",
    "n_tokens",
    "idx_first_diff",
    "idx_last_subtok",
    "n_word_tokens",
    "target_after_word",
    "ends_with_punct",
    "ell",
    "model",
]
_STR = ["sentence_id", "item_id", "split", "bias_type", "target", "sentence", "word", "model"]
_INT = ["row", "S", "n_tokens", "idx_first_diff", "idx_last_subtok", "n_word_tokens"]
_BOOL = ["target_after_word", "ends_with_punct"]
META_DTYPES = {
    **dict.fromkeys(_STR, str),
    **dict.fromkeys(_INT, "int64"),
    **dict.fromkeys(_BOOL, bool),
    "ell": "float64",
}


# --------------------------------------------------------------------------- paths


def rq2_root() -> Path:
    """``paths.rq2_root`` from the config (team folder on JUPITER)."""
    return get("paths.rq2_root")


def meta_path(model: str, root=None) -> Path:
    return Path(root or rq2_root()) / "meta" / f"{model}.csv"


def layer_path(model: str, position: str, k: int, root=None) -> Path:
    """Activations of ``hidden_states[k]`` at ``position`` (k = 0 is the embedding output)."""
    if position not in POSITIONS:
        raise ValueError(f"unknown position {position!r}; expected one of {POSITIONS}")
    if int(k) != k or k < 0:
        raise ValueError(f"layer index must be a non-negative integer, got {k!r}")
    return Path(root or rq2_root()) / "acts" / model / position / f"layer_{int(k):02d}.npy"


# --------------------------------------------------------------------------- metadata


def read_meta(path) -> pd.DataFrame:
    """Read a meta CSV. Only ``ell`` may hold NaN; text is kept exactly as written."""
    return pd.read_csv(
        path,
        dtype=META_DTYPES,
        keep_default_na=False,
        na_values={"ell": [""]},
        encoding="utf-8",
    )


def write_meta(df: pd.DataFrame, path) -> Path:
    """Validate ``df`` and write it to ``path`` (folders are created)."""
    validate_meta(df)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def load_meta(model: str, root=None) -> pd.DataFrame:
    return read_meta(meta_path(model, root))


def save_meta(df: pd.DataFrame, model: str, root=None) -> Path:
    models = set(df["model"]) if "model" in df else set()
    if models != {model}:
        raise ValueError(f"meta for {model!r} has model column values {sorted(models)}")
    return write_meta(df, meta_path(model, root))


def _rows(mask) -> str:
    bad = np.flatnonzero(np.asarray(mask))
    more = f" (+{len(bad) - 5} more)" if len(bad) > 5 else ""
    return f"{bad[:5].tolist()}{more}"


def validate_meta(df: pd.DataFrame) -> None:
    """Raise ValueError if ``df`` breaks the meta schema or the row-order rule."""
    if list(df.columns) != META_COLUMNS:
        raise ValueError(f"meta columns must be exactly {META_COLUMNS}, got {list(df.columns)}")
    n = len(df)
    if n == 0:
        raise ValueError("meta is empty")
    nan_cols = [c for c in META_COLUMNS if c != "ell" and df[c].isna().any()]
    if nan_cols:
        raise ValueError(f"NaN in columns {nan_cols} (only ell may be NaN)")

    if not np.array_equal(df["row"].to_numpy(), np.arange(n)):
        raise ValueError(
            f"row must be 0..{n - 1} in order; mismatch at rows {_rows(df['row'] != np.arange(n))}"
        )
    s = df["S"].to_numpy()
    if not np.isin(s, [0, 1]).all():
        raise ValueError(f"S must be 0 or 1; bad rows {_rows(~np.isin(s, [0, 1]))}")
    if (s == 1).sum() != (s == 0).sum():
        raise ValueError(
            f"labels are unbalanced: {(s == 1).sum()} with S=1, {(s == 0).sum()} with S=0"
        )

    # Pair rule: rows 2i (S=1) and 2i+1 (S=0) share an item_id that appears nowhere else.
    if n % 2:
        raise ValueError(f"meta must have an even number of rows (pairs), got {n}")
    items = df["item_id"].to_numpy()
    if not (items[0::2] == items[1::2]).all():
        bad = np.flatnonzero(items[0::2] != items[1::2]) * 2
        raise ValueError(
            f"rows 2i and 2i+1 must share item_id; first bad pairs start at {bad[:5].tolist()}"
        )
    if not ((s[0::2] == 1) & (s[1::2] == 0)).all():
        bad = np.flatnonzero((s[0::2] != 1) | (s[1::2] != 0)) * 2
        raise ValueError(
            f"each pair must be S=1 then S=0; first bad pairs start at {bad[:5].tolist()}"
        )
    dup_items = pd.Series(items).value_counts()
    dup_items = dup_items[dup_items != 2]
    if len(dup_items):
        raise ValueError(
            f"each item_id must have exactly 2 rows; offenders: {dup_items.index[:5].tolist()}"
        )

    # Negative indices, never pointing at BOS (index -n_tokens).
    n_tok = df["n_tokens"].to_numpy()
    for col in POSITION_COLUMNS.values():
        idx = df[col].to_numpy()
        if not (idx < 0).all():
            raise ValueError(
                f"{col} must be negative (counted from the end); bad rows {_rows(idx >= 0)}"
            )
        if not (idx >= -n_tok + 1).all():
            raise ValueError(
                f"{col} points at BOS or before the sentence; bad rows {_rows(idx < -n_tok + 1)}"
            )
    if not (df["idx_last_subtok"] >= df["idx_first_diff"]).all():
        bad = df["idx_last_subtok"] < df["idx_first_diff"]
        raise ValueError(f"idx_last_subtok must be >= idx_first_diff; bad rows {_rows(bad)}")

    dup_ids = df["sentence_id"][df["sentence_id"].duplicated()]
    if len(dup_ids):
        raise ValueError(f"sentence_id must be unique; duplicates: {dup_ids.head().tolist()}")


# --------------------------------------------------------------------------- activations


def load_layer(model: str, position: str, k: int, mmap: bool = True, root=None) -> np.ndarray:
    """(N, hidden) activations as stored (float16, float32 for models that overflow it).

    Memory-mapped read-only by default.
    """
    return np.load(layer_path(model, position, k, root), mmap_mode="r" if mmap else None)


def save_layer(arr, model: str, position: str, k: int, root=None, dtype="float16") -> Path:
    """Write (N, hidden) activations as ``dtype`` (float16 by default, or float32).

    float16 tops out at 65504; models with larger outlier dimensions are stored as float32
    (all layers of that model). The file appears only once fully written.
    """
    if np.dtype(dtype) not in (np.float16, np.float32):
        raise ValueError(f"dtype must be float16 or float32, got {dtype!r}")
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"activations must be 2-D (N, hidden), got shape {arr.shape}")
    with np.errstate(over="ignore"):
        half = arr.astype(dtype, copy=False)
    if np.isfinite(arr).all() and not np.isfinite(half).all():
        raise ValueError(
            f"{model} {position} layer {k}: {int((~np.isfinite(half)).sum())} values overflow "
            f"float16 (max |x| = {float(np.abs(arr).max()):.4g})"
        )
    path = layer_path(model, position, k, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, half)
    os.replace(tmp, path)
    return path
