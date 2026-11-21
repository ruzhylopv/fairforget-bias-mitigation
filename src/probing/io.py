"""File layout of the RQ2 probing data: activations (X) and the metadata table (y, groups, ...).

One run = one model on one StereoSet bias type. ``run_paths(model, bias)`` gives all its paths,
and every step reads and writes through it::

    {rq2_root}/{bias}/{model}/meta.csv                            one row per sentence
    {rq2_root}/{bias}/{model}/acts/{position}/layer_{kk:02d}.npy  float16 or float32 (N, hidden)
    {rq2_root}/{bias}/{model}/acts/run.json                       extraction provenance + stats
    {rq2_root}/{bias}/{model}/probes/...                          folds, predictions, weights
    results/rq2/{bias}/{model}/                                   small results, committed
    results/rq2/{bias}/{model}/run.json                           date, commit, ... of each step

``kk`` indexes ``hidden_states`` (00 = embeddings). Row i of the meta CSV is row i of every
layer file. Rows come pair by pair, stereotype first, so rows 2i and 2i+1 are one StereoSet item.
Dates and commits go into ``run.json``, never into folder names.

Per-bias settings (items left out, misspelled words) are in ``config/rq2/{bias}.yaml``; the
fixed reporting depth is in ``config/rq2/report.yaml``.

Depth across models: ``rel_depth = k / n_layers``, with k the ``hidden_states`` index and
n_layers the number of decoder layers (48 for lapa and gemma3, 36 for qwen3, 32 for Llama).
"""

import datetime
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rmu.config import PROJECT_ROOT, get, repo_path, resolve_model

# Token positions with activations on disk, and the meta column holding each one's index.
# Indices are negative (counted from the end), so they work on left-padded batches unchanged.
POSITIONS = ("first_diff", "last_subtok")
POSITION_COLUMNS = {"first_diff": "idx_first_diff", "last_subtok": "idx_last_subtok"}
# StereoSet intrasentence bias types.
BIAS_TYPES = ("gender", "profession", "race", "religion")
RESULTS_ROOT = "results/rq2"  # in the repo
BIAS_CONFIG_DIR = "config/rq2"  # in the repo
REPORT_CONFIG = "config/rq2/report.yaml"  # in the repo

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


@dataclass(frozen=True)
class RunPaths:
    """Where one run (``model`` on bias type ``bias``) keeps its files; see the module doc."""

    model: str
    bias: str
    data: Path  # big files: {rq2_root}/{bias}/{model}
    results: Path  # small, committed: results/rq2/{bias}/{model}

    @property
    def meta(self) -> Path:
        return self.data / "meta.csv"

    @property
    def acts(self) -> Path:
        return self.data / "acts"

    @property
    def probes(self) -> Path:
        return self.data / "probes"

    @property
    def summary(self) -> Path:
        return self.results / "summary.csv"

    @property
    def run_json(self) -> Path:
        return self.results / "run.json"

    def layer(self, position: str, k: int) -> Path:
        """Activations of ``hidden_states[k]`` at ``position`` (k = 0 is the embedding output)."""
        if position not in POSITIONS:
            raise ValueError(f"unknown position {position!r}; expected one of {POSITIONS}")
        if int(k) != k or k < 0:
            raise ValueError(f"layer index must be a non-negative integer, got {k!r}")
        return self.acts / position / f"layer_{int(k):02d}.npy"


def run_paths(model: str, bias: str, root=None, results_root=None) -> RunPaths:
    """Paths of the run ``model`` x ``bias``.

    ``root`` replaces ``rq2_root()`` (fake data, dry runs). The committed results then go to
    ``{root}/results/{bias}/{model}`` instead of the repo, so a test run never overwrites real
    results; ``results_root`` sets that folder's parent explicitly.
    """
    if not model or any(c in model for c in "/\\") or model.startswith("."):
        raise ValueError(f"model must be a short name (no path), got {model!r}")
    if bias not in BIAS_TYPES:
        raise ValueError(f"unknown bias type {bias!r}; expected one of {BIAS_TYPES}")
    if results_root is None:
        results_root = repo_path(RESULTS_ROOT) if root is None else Path(root) / "results"
    data_root = rq2_root() if root is None else Path(root)
    return RunPaths(model, bias, Path(data_root) / bias / model, Path(results_root) / bias / model)


# --------------------------------------------------------------------------- per-bias settings


def bias_config(bias: str) -> dict:
    """Settings of a bias type from ``config/rq2/{bias}.yaml``:

    * ``exclude_items``: {item_id: reason}, pairs left out of the meta (both rows);
    * ``misspelled``: attribute words whose pairs the robustness check drops.
    """
    if bias not in BIAS_TYPES:
        raise ValueError(f"unknown bias type {bias!r}; expected one of {BIAS_TYPES}")
    path = repo_path(f"{BIAS_CONFIG_DIR}/{bias}.yaml")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing: check the {bias} pairs and list the items to exclude and the "
            "misspelled words there (empty lists are fine), see config/rq2/gender.yaml"
        )
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    unknown = set(raw) - {"exclude_items", "misspelled"}
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    exclude = raw.get("exclude_items") or {}
    misspelled = raw.get("misspelled") or []
    if not isinstance(exclude, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in exclude.items()
    ):
        raise ValueError(f"{path}: exclude_items must map item_id -> reason (both strings)")
    if not isinstance(misspelled, list) or not all(isinstance(w, str) for w in misspelled):
        raise ValueError(f"{path}: misspelled must be a list of words")
    return {"exclude_items": dict(exclude), "misspelled": list(misspelled)}


# --------------------------------------------------------------------------- depth


def n_decoder_layers(paths: RunPaths) -> int:
    """Decoder layers of the run's model (= hidden states - 1; k = 0 is the embedding output).

    From ``{acts}/run.json`` (written by probing.extract), else from the model's config.json.
    """
    run = paths.acts / "run.json"
    if run.exists():
        info = json.loads(run.read_text(encoding="utf-8"))
        if "n_hidden_states" in info:
            return int(info["n_hidden_states"]) - 1
    cfg = Path(resolve_model(paths.model)) / "config.json"
    if cfg.exists():
        c = json.loads(cfg.read_text(encoding="utf-8"))
        return int(c.get("text_config", c)["num_hidden_layers"])  # Gemma 3: text_config
    raise FileNotFoundError(
        f"number of layers of {paths.model} unknown: no n_hidden_states in {run} and no {cfg}"
    )


def rel_depth(layer, n_layers: int):
    """``layer / n_layers`` (NaN for negative layers, i.e. the ell baseline row)."""
    layer = np.asarray(layer, dtype=float)
    out = np.where(layer >= 0, layer / n_layers, np.nan)
    return float(out) if out.ndim == 0 else out


def report_config() -> dict:
    """``config/rq2/report.yaml``. ``fixed_depth`` gets ``rel_depth`` = layer / n_layers."""
    path = repo_path(REPORT_CONFIG)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    fd = cfg.get("fixed_depth") or {}
    need = {"layer", "n_layers", "position", "chosen_on", "date", "rule"}
    if need - set(fd):
        raise ValueError(f"{path}: fixed_depth needs {sorted(need - set(fd))}")
    if fd["position"] not in POSITIONS:
        raise ValueError(f"{path}: fixed_depth.position must be one of {POSITIONS}")
    if not 0 <= fd["layer"] <= fd["n_layers"]:
        raise ValueError(f"{path}: fixed_depth.layer must be in 0..n_layers")
    fd["rel_depth"] = fd["layer"] / fd["n_layers"]
    fd["date"] = str(fd["date"])
    return cfg


def fixed_layer(n_layers: int, cfg: dict | None = None) -> int:
    """The layer nearest the fixed depth in a model with ``n_layers`` decoder layers (halves
    round up)."""
    fd = (cfg or report_config())["fixed_depth"]
    return int(np.floor(fd["layer"] * n_layers / fd["n_layers"] + 0.5))


def chosen_on(paths: RunPaths, cfg: dict | None = None) -> bool:
    """True if the fixed depth was chosen on this run (the report must say so)."""
    on = (cfg or report_config())["fixed_depth"]["chosen_on"]
    return (on.get("model"), on.get("bias")) == (paths.model, paths.bias)


# --------------------------------------------------------------------------- run.json


def git_commit(repo: Path = PROJECT_ROOT):
    """HEAD commit; read from .git directly when git is missing (compute nodes)."""
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    try:
        git = repo / ".git"
        if git.is_file():  # worktree: "gitdir: <path>"
            git = Path(git.read_text().split(":", 1)[1].strip())
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head
        ref = head.split(":", 1)[1].strip()
        common = git / "commondir"
        roots = [git] + ([git / common.read_text().strip()] if common.exists() else [])
        for root in roots:
            if (root / ref).exists():
                return (root / ref).read_text().strip()
            packed = root / "packed-refs"
            if packed.exists():
                for line in packed.read_text().splitlines():
                    if line.endswith(" " + ref):
                        return line.split()[0]
    except OSError:
        pass
    return None


def _write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def record_run(paths: RunPaths, step: str, info: dict | None = None, stamp: bool = True) -> Path:
    """Add or replace ``step`` in ``results/.../run.json``: {model, bias, steps: {step: ...}}.

    With ``stamp`` the entry starts with the current date, git commit and Slurm job id;
    ``info`` (JSON-serialisable) follows. A run.json of another model or bias is an error.
    """
    path = paths.run_json
    run = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if run and (run.get("model"), run.get("bias")) != (paths.model, paths.bias):
        raise ValueError(
            f"{path} belongs to {run.get('model')!r} / {run.get('bias')!r}, "
            f"not {paths.model!r} / {paths.bias!r}"
        )
    entry = {}
    if stamp:
        entry = {
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
    entry |= info or {}
    steps = run.get("steps", {})
    steps[step] = entry
    return _write_json(path, {"model": paths.model, "bias": paths.bias, "steps": steps})


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


def _check_run(df: pd.DataFrame, paths: RunPaths) -> None:
    """The meta's ``model`` and ``bias_type`` columns must name the run."""
    for col, want in (("model", paths.model), ("bias_type", paths.bias)):
        got = set(df[col]) if col in df else set()
        if got != {want}:
            raise ValueError(f"meta for {paths.model} / {paths.bias} has {col} {sorted(got)}")


def load_meta(paths: RunPaths) -> pd.DataFrame:
    df = read_meta(paths.meta)
    _check_run(df, paths)
    return df


def save_meta(df: pd.DataFrame, paths: RunPaths) -> Path:
    _check_run(df, paths)
    return write_meta(df, paths.meta)


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


def load_layer(paths: RunPaths, position: str, k: int, mmap: bool = True) -> np.ndarray:
    """(N, hidden) activations as stored (float16, float32 for models that overflow it).

    Memory-mapped read-only by default.
    """
    return np.load(paths.layer(position, k), mmap_mode="r" if mmap else None)


def save_layer(arr, paths: RunPaths, position: str, k: int, dtype="float16") -> Path:
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
            f"{paths.model} {position} layer {k}: {int((~np.isfinite(half)).sum())} values "
            f"overflow float16 (max |x| = {float(np.abs(arr).max()):.4g})"
        )
    path = paths.layer(position, k)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, half)
    os.replace(tmp, path)
    return path
