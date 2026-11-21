"""Build the RQ2 metadata table (one row per sentence) from the tokenization check.

The token positions come from ``results/tokenization-check-nov17/pairs.csv``; nothing is
re-tokenized here. Positions depend on the tokenizer, so there is one meta file per model and
bias type: ``run_paths(model, bias).meta``. Items left out are listed in ``config/rq2/{bias}.yaml``.

Usage:
    python -m probing.meta --model lapa-12b-instruct --bias gender [--root PATH]
"""

import argparse

import numpy as np
import pandas as pd

from probing.io import META_COLUMNS, bias_config, record_run, run_paths, save_meta, validate_meta
from rmu.config import repo_path

PAIRS_CSV = "results/tokenization-check-nov17/pairs.csv"
PUNCT = (".", "!", "?")


def word_start(sentence: str, word: str, other: str) -> int:
    """Char index where ``word`` starts in ``sentence``.

    ``other`` is the paired sentence: the attribute word is the occurrence that contains the
    first character where the two sentences differ (the word can also appear elsewhere).
    """
    p = 0
    while p < min(len(sentence), len(other)) and sentence[p] == other[p]:
        p += 1
    starts = []
    i = sentence.find(word)
    while i != -1:
        starts.append(i)
        i = sentence.find(word, i + 1)
    if not starts:
        raise ValueError(f"word {word!r} not found in {sentence!r}")
    hits = [s for s in starts if s <= p <= s + len(word)]
    return hits[-1] if hits else starts[0]


def target_after_word(sentence: str, target: str, start: int) -> bool:
    """True if the first case-insensitive occurrence of ``target`` starts after the word."""
    pos = sentence.lower().find(target.lower())
    return pos > start


def _side_rows(pairs: pd.DataFrame, side: str, other: str, label: int, model: str):
    sentences = pairs[f"sentence_{side}"]
    words = pairs[f"word_{side}"]
    starts = [word_start(s, w, o) for s, w, o in zip(sentences, words, pairs[f"sentence_{other}"])]
    return pd.DataFrame(
        {
            "sentence_id": pairs[f"sentence_id_{side}"].to_numpy(),
            "item_id": pairs["item_id"].to_numpy(),
            "split": pairs["split"].to_numpy(),
            "bias_type": pairs["bias_type"].to_numpy(),
            "target": pairs["target"].to_numpy(),
            "S": label,
            "sentence": sentences.to_numpy(),
            "word": words.to_numpy(),
            "n_tokens": pairs[f"len_{side}"].to_numpy(),
            "idx_first_diff": (pairs[f"first_diff_{side}"] - pairs[f"len_{side}"]).to_numpy(),
            "idx_last_subtok": pairs[f"last_subtok_from_end_{side}"].to_numpy(),
            "n_word_tokens": pairs[f"n_tok_{side}"].to_numpy(),
            "target_after_word": [
                target_after_word(s, t, a) for s, t, a in zip(sentences, pairs["target"], starts)
            ],
            "ends_with_punct": [s.strip().endswith(PUNCT) for s in sentences],
            "ell": np.nan,
            "model": model,
        }
    )


def build_meta(model: str, bias: str, pairs_csv=PAIRS_CSV, exclude=None) -> pd.DataFrame:
    """Meta table for ``model`` on bias type ``bias``: pairs.csv order, stereotype row first in
    each pair.

    Items in ``exclude`` (item_id -> reason; default: ``exclude_items`` of
    ``config/rq2/{bias}.yaml``) are dropped, both sentences together.
    """
    if exclude is None:
        exclude = bias_config(bias)["exclude_items"]
    pairs = pd.read_csv(repo_path(pairs_csv), keep_default_na=False, dtype=str)
    pairs = pairs[(pairs["model"] == model) & (pairs["bias_type"] == bias)]
    pairs = pairs[~pairs["item_id"].isin(list(exclude))]
    if pairs.empty:
        raise ValueError(f"no pairs for model {model!r} and bias type {bias!r}")
    pairs = pairs.reset_index(drop=True)
    for prefix in ("n_tok", "len", "first_diff", "last_subtok_from_end"):
        for side in ("stereo", "anti"):
            pairs[f"{prefix}_{side}"] = pairs[f"{prefix}_{side}"].astype("int64")

    stereo = _side_rows(pairs, "stereo", "anti", 1, model)
    anti = _side_rows(pairs, "anti", "stereo", 0, model)
    # Interleave: pair i -> rows 2i (stereotype) and 2i+1 (anti-stereotype).
    stereo.index = np.arange(len(pairs)) * 2
    anti.index = np.arange(len(pairs)) * 2 + 1
    df = pd.concat([stereo, anti]).sort_index().reset_index(drop=True)
    df.insert(0, "row", np.arange(len(df)))
    df = df[META_COLUMNS]
    validate_meta(df)
    return df


def describe(df: pd.DataFrame, n_excluded: int = 0) -> str:
    """Short summary of a meta table, for the CLI and quick checks."""
    stereo = df[df["S"] == 1]
    bias = ", ".join(sorted(set(df["bias_type"])))
    lines = [
        f"rows: {len(df)} (excluded items: {n_excluded}, see config/rq2/{bias}.yaml)",
        f"targets: {df['target'].nunique()}",
        f"S=1 / S=0: {(df['S'] == 1).sum()} / {(df['S'] == 0).sum()}",
        f"pairs with target_after_word (stereotype row): {int(stereo['target_after_word'].sum())}",
        f"pairs with target_after_word (either row): "
        f"{df.groupby('item_id')['target_after_word'].any().sum()}",
        f"sentences without ends_with_punct: {(~df['ends_with_punct']).sum()} "
        f"({100 * (~df['ends_with_punct']).mean():.1f}%)",
        f"multi-token words: {(df['n_word_tokens'] > 1).sum()}",
        f"first_diff != last_subtok: {(df['idx_first_diff'] != df['idx_last_subtok']).sum()}",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="model name as in pairs.csv")
    ap.add_argument("--bias", default="gender", help="StereoSet bias type (default: gender)")
    ap.add_argument("--pairs_csv", default=PAIRS_CSV)
    ap.add_argument("--root", default=None, help="data root (default: paths.rq2_root)")
    args = ap.parse_args()

    paths = run_paths(args.model, args.bias, root=args.root)
    exclude = bias_config(args.bias)["exclude_items"]
    df = build_meta(args.model, args.bias, args.pairs_csv, exclude)
    out = save_meta(df, paths)
    record_run(
        paths,
        "meta",
        {"pairs_csv": str(args.pairs_csv), "n_rows": len(df), "excluded_items": exclude},
    )
    print(describe(df, len(exclude)))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
