"""RQ2 data check for one StereoSet bias type, from the tokenization check (CPU, no model).

Run before a bias type's first GPU run; the findings go into ``config/rq2/{bias}.yaml``
(``exclude_items`` with a reason, ``misspelled``). Prints a report and writes
``results/rq2/data_check/{bias}.md`` and ``{bias}_flags.csv`` (one row per flagged pair).

* misspelling candidates: attribute words with ``wordfreq.zipf_frequency(word, "en") == 0``
  (check by hand: misspelling, or just a rare word?);
* broken pairs, found automatically: the same word on both sides, the word not in its
  sentence, sentences that differ outside the attribute word, a word that is a prefix of the
  other ("are" / "arent": the first differing token lies after the shorter word);
* shares: target after the word, multi-token words (per model), no final punctuation;
* items per target (smallest, median): few items per target make the grouped CV noisy.

After the first extraction, also look at the lowest-``ell`` words (``--ell`` with the run's
meta.csv), as was done for gender.

Usage:
    uv run python scripts/rq2_data_check.py --bias profession
    uv run python scripts/rq2_data_check.py --bias gender --ell path/to/meta.csv
"""

import argparse
import re
from pathlib import Path

import pandas as pd
from wordfreq import zipf_frequency

from probing.meta import PAIRS_CSV, build_meta, word_start
from rmu.config import repo_path

OUT_DIR = "results/rq2/data_check"
PUNCT = ".,!?;:'\"()"


def norm(word: str) -> str:
    return word.strip().strip(PUNCT).lower()


def zipf(word: str) -> float:
    return zipf_frequency(norm(word), "en")


def rest_of_sentence(sentence: str, word: str, other: str) -> str:
    """The sentence with the attribute word cut out, lower case. The occurrence cut is the one
    at the first character where the pair differs (``probing.meta.word_start``): the word can
    also appear earlier ("clean her clean house", "stepmother ... mother")."""
    try:
        i = word_start(sentence, word, other)
    except ValueError:
        return sentence.lower()
    s = sentence[:i] + sentence[i + len(word) :]
    return re.sub(r"\s+", " ", s).strip().lower()


def pair_flags(pairs: pd.DataFrame) -> pd.DataFrame:
    """One row per item with automatic flags (model-independent: one model's rows suffice)."""
    rows = []
    for r in pairs.itertuples(index=False):
        ws, wa, ss, sa = r.word_stereo, r.word_anti, r.sentence_stereo, r.sentence_anti
        flags = []
        if norm(ws) == norm(wa):
            flags.append("same word on both sides")
        if ss.find(ws) < 0 or sa.find(wa) < 0:
            flags.append("attribute word not in its sentence")
        if rest_of_sentence(ss, ws, sa) != rest_of_sentence(sa, wa, ss):
            flags.append("sentences differ outside the attribute word")
        if norm(ws) != norm(wa) and (
            norm(ws).startswith(norm(wa)) or norm(wa).startswith(norm(ws))
        ):
            flags.append("one word is a prefix of the other")
        zs, za = zipf(ws), zipf(wa)
        if zs == 0:
            flags.append(f"rare/misspelled? {norm(ws)!r} (stereo)")
        if za == 0:
            flags.append(f"rare/misspelled? {norm(wa)!r} (anti)")
        rows.append(
            {
                "item_id": r.item_id,
                "target": r.target,
                "word_stereo": ws,
                "word_anti": wa,
                "zipf_stereo": zs,
                "zipf_anti": za,
                "flags": "; ".join(flags),
                "sentence_stereo": ss,
                "sentence_anti": sa,
            }
        )
    return pd.DataFrame(rows)


def model_shares(bias: str, models) -> list:
    """Per model: pairs kept by build_meta (no exclusions), target after the word, multi-token
    words, no final punctuation; or the error build_meta raises."""
    out = []
    for m in models:
        try:
            meta = build_meta(m, bias, exclude={})
        except ValueError as e:
            out.append({"model": m, "error": str(e)})
            continue
        stereo = meta[meta["S"] == 1]
        n = len(stereo)
        out.append(
            {
                "model": m,
                "pairs": n,
                "dropped_no_context": len(meta.attrs.get("dropped_no_context", [])),
                "target_after_word": meta.groupby("item_id")["target_after_word"].any().mean(),
                "multi_token_word": (meta["n_word_tokens"] > 1).mean(),
                "no_final_punct": (~meta["ends_with_punct"]).mean(),
            }
        )
    return out


def ell_report(meta_csv, n=15) -> str:
    """The lowest-ell attribute words of an extracted meta.csv (misspellings sink to the bottom)."""
    meta = pd.read_csv(meta_csv, keep_default_na=False, na_values={"ell": [""]})
    low = meta.nsmallest(n, "ell")[["ell", "word", "sentence"]]
    lines = [f"lowest ell in {meta_csv}:"]
    lines += [f"  {r.ell:8.2f}  {r.word!r:18}  {r.sentence}" for r in low.itertuples()]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bias", required=True)
    ap.add_argument("--pairs_csv", default=PAIRS_CSV)
    ap.add_argument("--ell", default=None, help="an extracted meta.csv: list the lowest-ell words")
    args = ap.parse_args(argv)

    pairs = pd.read_csv(repo_path(args.pairs_csv), keep_default_na=False, dtype=str)
    pairs = pairs[pairs["bias_type"] == args.bias]
    models = sorted(pairs["model"].unique())
    one = pairs[pairs["model"] == models[0]]  # words and sentences do not depend on the model
    flags = pair_flags(one)
    per_target = one.groupby("target").size().sort_values()

    lines = [
        f"# Data check: {args.bias}",
        "",
        f"pairs: {len(one)}, targets: {len(per_target)}; items per target: smallest "
        f"{per_target.iloc[0]} ({per_target.index[0]}), median {per_target.median():.0f}, "
        f"largest {per_target.iloc[-1]} ({per_target.index[-1]})",
        f"targets with < 10 items: {int((per_target < 10).sum())}"
        + (
            f" ({', '.join(f'{t} {k}' for t, k in per_target[per_target < 10].items())})"
            if (per_target < 10).any()
            else ""
        ),
        "",
        "## Per model (build_meta, no exclusions)",
        "",
    ]
    for s in model_shares(args.bias, models):
        if "error" in s:
            lines.append(f"- {s['model']}: build_meta fails: {s['error']}")
        else:
            lines.append(
                f"- {s['model']}: {s['pairs']} pairs ({s['dropped_no_context']} dropped, word "
                f"first without BOS); target after the word {100 * s['target_after_word']:.1f}% "
                f"of pairs; multi-token words {100 * s['multi_token_word']:.1f}% of sentences; "
                f"no final punctuation {100 * s['no_final_punct']:.1f}%"
            )
    flagged = flags[flags["flags"] != ""]
    lines += ["", f"## Flagged pairs ({len(flagged)})", ""]
    for kind in (
        "same word on both sides",
        "attribute word not in its sentence",
        "sentences differ outside the attribute word",
        "one word is a prefix of the other",
        "rare/misspelled?",
    ):
        f = flagged[flagged["flags"].str.contains(re.escape(kind))]
        lines.append(f"### {kind} ({len(f)})")
        for r in f.itertuples():
            lines.append(
                f"- `{r.item_id}` {r.target}: {r.word_stereo!r} / {r.word_anti!r} "
                f'(zipf {r.zipf_stereo:.2f} / {r.zipf_anti:.2f}): "{r.sentence_stereo}" / '
                f'"{r.sentence_anti}"'
            )
        lines.append("")
    if args.ell:
        lines += ["## Lowest ell", "", "```", ell_report(args.ell), "```", ""]

    text = "\n".join(lines)
    print(text)
    out = repo_path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.bias}.md").write_text(text + "\n", encoding="utf-8")
    flagged.to_csv(out / f"{args.bias}_flags.csv", index=False)
    print(f"\nwrote {Path(OUT_DIR) / (args.bias + '.md')} and {args.bias}_flags.csv")


if __name__ == "__main__":
    main()
