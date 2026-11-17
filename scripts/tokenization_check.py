"""Tokenization check for the RQ2 probing positions (CPU only, tokenizers only).

For every StereoSet intrasentence item (dev + test), take the stereotype / anti-stereotype
pair and, per model tokenizer, record:

* how many tokens each attribute word (the filler of BLANK) takes,
* whether the first token that differs between the two full sentences (as found by
  ``rmu.unlearn.from_first_difference``) is the first token of the attribute word or falls
  mid-word (shared subtoken prefix, e.g. "▁un" + "caring" vs "▁un" + "kind"),
* the index of the attribute word's last subtoken in each sentence, for activation extraction.

Tokenizers are loaded like ``rmu.utils.load_model`` does (same ``from_pretrained`` args, left
padding, pad = eos). Sentences are tokenized one at a time with special tokens, which are then
masked out, like ``rmu.unlearn.loss_positions``.

Outputs (in --out_dir, default results/tokenization-check-nov17):
    pairs.csv          one row per model x pair
    summary.csv        one row per model x bias type (plus "all")
    examples.md        5 example pairs per model x category, to eyeball
    positions/<model>.json   per-sentence token positions for extraction
    meta.json          tokenizer sources, transformers version, command line

Usage:
    # JUPITER login node, from your clone
    source slurm/env.sh && python scripts/tokenization_check.py
    # laptop (tokenizers downloaded from the hub; Gemma 3 is gated, needs `hf auth login`)
    uv run python scripts/tokenization_check.py
    # options: --models qwen3-8b  --hub lapa-12b-instruct=org/repo  --limit 200  --out_dir ...
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer

from rmu.config import load_config, repo_path, resolve_model
from rmu.unlearn import from_first_difference

DEFAULT_MODELS = ["lapa-12b-instruct", "gemma3-12b-it", "qwen3-8b"]
# Used only when the shared models folder is not there (e.g. on a laptop).
HUB_FALLBACK = {
    "lapa-12b-instruct": "lapa-llm/lapa-v0.1.2-instruct",
    "gemma3-12b-it": "google/gemma-3-12b-it",
    "qwen3-8b": "Qwen/Qwen3-8B",
}
N_EXAMPLES = 5


# --------------------------------------------------------------------------- data


def word_span_from_context(context, sentence):
    """Char span of the BLANK filler in ``sentence``, or None if the context doesn't line up."""
    if context.count("BLANK") != 1:
        return None
    pre, post = context.split("BLANK")
    low = sentence.lower()
    if len(sentence) < len(pre) + len(post):
        return None
    if not (low.startswith(pre.lower()) and low.endswith(post.lower())):
        return None
    start, end = len(pre), len(sentence) - len(post)
    return (start, end) if start < end else None


def word_span_from_pair(sentence, other):
    """Fallback: char span where ``sentence`` differs from ``other``, widened to whole words."""
    p = 0
    while p < min(len(sentence), len(other)) and sentence[p] == other[p]:
        p += 1
    s = 0
    while (
        s < min(len(sentence), len(other)) - p
        and sentence[len(sentence) - 1 - s] == other[len(other) - 1 - s]
    ):
        s += 1
    start, end = p, len(sentence) - s
    while start > 0 and sentence[start - 1].isalnum():
        start -= 1
    while end < len(sentence) and sentence[end].isalnum():
        end += 1
    return (start, end) if start < end else None


def load_pairs(files):
    pairs = []
    for f in files:
        split = Path(f).stem
        items = json.loads(Path(f).read_text(encoding="utf-8"))["data"]["intrasentence"]
        for item in items:
            by_label = {s["gold_label"]: s for s in item["sentences"]}
            if "stereotype" not in by_label or "anti-stereotype" not in by_label:
                continue
            st, an = by_label["stereotype"], by_label["anti-stereotype"]
            span_st = word_span_from_context(item["context"], st["sentence"])
            span_an = word_span_from_context(item["context"], an["sentence"])
            method = "context"
            if span_st is None or span_an is None:
                method = "pair_diff"
                span_st = word_span_from_pair(st["sentence"], an["sentence"])
                span_an = word_span_from_pair(an["sentence"], st["sentence"])
            if span_st is None or span_an is None:
                print(f"skip {item['id']}: no attribute span", file=sys.stderr)
                continue
            pairs.append(
                {
                    "split": split,
                    "item_id": item["id"],
                    "bias_type": item["bias_type"],
                    "target": item["target"],
                    "context": item["context"],
                    "span_method": method,
                    "stereo": {**st, "span": span_st},
                    "anti": {**an, "span": span_an},
                }
            )
    return pairs


# --------------------------------------------------------------------------- tokenizers


def load_tokenizer(name, hub_overrides, local_files_only):
    """Same settings as rmu.utils.load_model; falls back to the hub if the folder is missing."""
    path = resolve_model(name)
    if Path(path).exists():
        source = str(path)
    else:
        source = hub_overrides.get(name) or HUB_FALLBACK.get(name) or name
        print(f"[{name}] {path} not found, loading tokenizer from the hub: {source}")
    tok = AutoTokenizer.from_pretrained(
        source, trust_remote_code=True, use_fast=False, local_files_only=local_files_only
    )
    tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    tok.mask_token_id = tok.eos_token_id
    tok.sep_token_id = tok.eos_token_id
    tok.cls_token_id = tok.eos_token_id
    return tok, source


def encode(tok, sentence):
    """input_ids, special-token mask and per-token char spans (None for special tokens)."""
    try:
        enc = tok(sentence, return_special_tokens_mask=True, return_offsets_mapping=True)
        offsets = [tuple(o) for o in enc["offset_mapping"]]
    except (NotImplementedError, ValueError, TypeError):
        enc = tok(sentence, return_special_tokens_mask=True)
        offsets = None
    ids = list(enc["input_ids"])
    special = [bool(x) for x in enc["special_tokens_mask"]]
    if offsets is None:
        offsets = offsets_by_decoding(tok, ids, special, sentence)
    spans = [None if sp else off for sp, off in zip(special, offsets)]
    return ids, special, spans


def offsets_by_decoding(tok, ids, special, sentence):
    """Char spans from cumulative decoding of the word tokens (for tokenizers without offsets)."""
    word_ids = [i for i, sp in zip(ids, special) if not sp]
    out, prev = [], 0
    k = 0
    for sp in special:
        if sp:
            out.append((0, 0))
            continue
        k += 1
        text = tok.decode(word_ids[:k], clean_up_tokenization_spaces=False)
        end = min(len(text), len(sentence))
        out.append((prev, end))
        prev = end
    return out


def word_token_positions(spans, word_span):
    """Indices (into input_ids) of tokens overlapping the attribute word's char span.

    Offsets of SentencePiece/BPE tokens may include the leading space, so overlap is tested
    against the word's characters only.
    """
    a, b = word_span
    return [
        i
        for i, sp in enumerate(spans)
        if sp is not None and sp[1] > sp[0] and sp[0] < b and sp[1] > a
    ]


def first_difference(ids_a, special_a, ids_b, special_b):
    """Index into ids_a of the first differing word token, via rmu.unlearn.from_first_difference."""
    width = max(len(ids_a), len(ids_b))

    def pad(xs, fill):  # left padding, like the tokenizer
        return [fill] * (width - len(xs)) + list(xs)

    a = torch.tensor([pad(ids_a, -1)])
    b = torch.tensor([pad(ids_b, -1)])
    wa = torch.tensor([pad([not s for s in special_a], False)])
    wb = torch.tensor([pad([not s for s in special_b], False)])
    keep = from_first_difference(a, wa, b, wb)
    first_padded = int(keep[0].nonzero()[0])
    return first_padded - (width - len(ids_a))


def classify(first, word_pos):
    if not word_pos:
        return "no_word_tokens"
    if first == word_pos[0]:
        return "word_start"
    if first in word_pos:
        return "mid_word"
    return "before_word" if first < word_pos[0] else "after_word"


# --------------------------------------------------------------------------- main


def analyse(model, tok, pairs):
    rows, positions = [], {}
    for p in pairs:
        side = {}
        for key in ("stereo", "anti"):
            s = p[key]
            ids, special, spans = encode(tok, s["sentence"])
            pos = word_token_positions(spans, s["span"])
            side[key] = (s, ids, special, pos)
        (st, ids_s, sp_s, pos_s), (an, ids_a, sp_a, pos_a) = side["stereo"], side["anti"]
        first_s = first_difference(ids_s, sp_s, ids_a, sp_a)
        first_a = first_difference(ids_a, sp_a, ids_s, sp_s)
        cls_s, cls_a = classify(first_s, pos_s), classify(first_a, pos_a)
        n_s, n_a = len(pos_s), len(pos_a)
        row = {
            "model": model,
            "split": p["split"],
            "item_id": p["item_id"],
            "bias_type": p["bias_type"],
            "target": p["target"],
            "span_method": p["span_method"],
            "word_stereo": st["sentence"][slice(*st["span"])],
            "word_anti": an["sentence"][slice(*an["span"])],
            "n_tok_stereo": n_s,
            "n_tok_anti": n_a,
            "tokens_stereo": "|".join(tok.convert_ids_to_tokens([ids_s[i] for i in pos_s])),
            "tokens_anti": "|".join(tok.convert_ids_to_tokens([ids_a[i] for i in pos_a])),
            "both_single_token": n_s == 1 and n_a == 1,
            "len_relation": "fewer" if n_s < n_a else "more" if n_s > n_a else "same",
            "first_diff_stereo": first_s,
            "first_diff_anti": first_a,
            "first_diff_class_stereo": cls_s,
            "first_diff_class_anti": cls_a,
            "first_diff_mid_word": "mid_word" in (cls_s, cls_a),
            "first_diff_at_word_start": cls_s == "word_start" and cls_a == "word_start",
            "len_stereo": len(ids_s),
            "len_anti": len(ids_a),
            "last_subtok_stereo": pos_s[-1] if pos_s else -1,
            "last_subtok_anti": pos_a[-1] if pos_a else -1,
            # Negative index: unchanged by left padding, so x[:, idx] works on padded batches.
            "last_subtok_from_end_stereo": pos_s[-1] - len(ids_s) if pos_s else 0,
            "last_subtok_from_end_anti": pos_a[-1] - len(ids_a) if pos_a else 0,
            "sentence_stereo": st["sentence"],
            "sentence_anti": an["sentence"],
            "sentence_id_stereo": st["id"],
            "sentence_id_anti": an["id"],
        }
        rows.append(row)
        for key, s, ids, pos in (("stereo", st, ids_s, pos_s), ("anti", an, ids_a, pos_a)):
            positions[s["id"]] = {
                "item_id": p["item_id"],
                "label": "stereotype" if key == "stereo" else "anti-stereotype",
                "bias_type": p["bias_type"],
                "sentence": s["sentence"],
                "n_tokens": len(ids),
                "word_token_idx": pos,
                "last_subtok_idx": pos[-1] if pos else None,
                "last_subtok_idx_from_end": pos[-1] - len(ids) if pos else None,
                "first_diff_idx": first_s if key == "stereo" else first_a,
            }
    return rows, positions


def pct(xs):
    return round(100 * sum(xs) / len(xs), 1) if xs else float("nan")


def mean(xs):
    return round(sum(xs) / len(xs), 3) if xs else float("nan")


def summarise(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["bias_type"])].append(r)
        groups[(r["model"], "all")].append(r)
    out = []
    for (model, bias), rs in sorted(groups.items()):
        out.append(
            {
                "model": model,
                "bias_type": bias,
                "n_pairs": len(rs),
                "pct_both_single_token": pct([r["both_single_token"] for r in rs]),
                "pct_first_diff_mid_word": pct([r["first_diff_mid_word"] for r in rs]),
                "pct_first_diff_other": pct(
                    [not r["first_diff_mid_word"] and not r["first_diff_at_word_start"] for r in rs]
                ),
                "mean_n_tok_stereo": mean([r["n_tok_stereo"] for r in rs]),
                "mean_n_tok_anti": mean([r["n_tok_anti"] for r in rs]),
                "pct_stereo_fewer": pct([r["len_relation"] == "fewer" for r in rs]),
                "pct_stereo_more": pct([r["len_relation"] == "more" for r in rs]),
                "pct_same": pct([r["len_relation"] == "same" for r in rs]),
            }
        )
    return out


def examples_md(rows, models):
    categories = {
        "stereo multi-token": lambda r: r["n_tok_stereo"] > 1,
        "anti multi-token": lambda r: r["n_tok_anti"] > 1,
        "stereo fewer tokens than anti": lambda r: r["len_relation"] == "fewer",
        "stereo more tokens than anti": lambda r: r["len_relation"] == "more",
        "first difference mid-word": lambda r: r["first_diff_mid_word"],
        "first difference outside the word": lambda r: (
            not r["first_diff_mid_word"] and not r["first_diff_at_word_start"]
        ),
    }
    lines = ["# Tokenization check: example pairs", ""]
    for model in models:
        mr = [r for r in rows if r["model"] == model]
        lines += [f"## {model}", ""]
        for name, pred in categories.items():
            hits = [r for r in mr if pred(r)]
            lines += [f"### {name} ({len(hits)} / {len(mr)} pairs)", ""]
            if not hits:
                lines += ["(none)", ""]
                continue
            lines += [
                "| bias | stereo | anti | stereo tokens | anti tokens | first diff (st/an) |",
                "|---|---|---|---|---|---|",
            ]
            # Spread the examples over bias types instead of taking the first five.
            by_bias = defaultdict(list)
            for r in hits:
                by_bias[r["bias_type"]].append(r)
            picked, i = [], 0
            while len(picked) < N_EXAMPLES and any(i < len(v) for v in by_bias.values()):
                for v in by_bias.values():
                    if i < len(v) and len(picked) < N_EXAMPLES:
                        picked.append(v[i])
                i += 1
            for r in picked:
                lines.append(
                    f"| {r['bias_type']} | {r['word_stereo']} | {r['word_anti']} "
                    f"| `{r['tokens_stereo']}` | `{r['tokens_anti']}` "
                    f"| {r['first_diff_class_stereo']} / {r['first_diff_class_anti']} |"
                )
            lines.append("")
    return "\n".join(lines)


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS), help="comma-separated")
    ap.add_argument(
        "--hub",
        action="append",
        default=[],
        metavar="NAME=HUB_ID",
        help="override the hub fallback for a model, e.g. lapa-12b-instruct=org/repo",
    )
    ap.add_argument("--local_files_only", action="store_true", help="never hit the hub")
    ap.add_argument("--out_dir", default="results/tokenization-check-nov17")
    ap.add_argument("--limit", type=int, default=None, help="first N pairs only (debugging)")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    hub = dict(h.split("=", 1) for h in args.hub)
    out = repo_path(args.out_dir)
    (out / "positions").mkdir(parents=True, exist_ok=True)

    files = [repo_path(f) for f in load_config()["data"]["stereoset_files"]]
    pairs = load_pairs(files)[: args.limit]
    print(f"{len(pairs)} stereotype/anti-stereotype pairs from {[f.name for f in files]}")

    all_rows, sources = [], {}
    for model in models:
        tok, source = load_tokenizer(model, hub, args.local_files_only)
        sources[model] = {"source": source, "class": type(tok).__name__}
        print(f"[{model}] {type(tok).__name__} from {source}")
        rows, positions = analyse(model, tok, pairs)
        missing = sum(r["n_tok_stereo"] == 0 or r["n_tok_anti"] == 0 for r in rows)
        if missing:
            print(f"[{model}] WARNING: {missing} pairs with no tokens on the attribute word")
        all_rows += rows
        (out / "positions" / f"{model}.json").write_text(
            json.dumps(positions, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    summary = summarise(all_rows)
    write_csv(out / "pairs.csv", all_rows)
    write_csv(out / "summary.csv", summary)
    (out / "examples.md").write_text(examples_md(all_rows, models), encoding="utf-8")
    (out / "meta.json").write_text(
        json.dumps(
            {
                "argv": sys.argv,
                "transformers": transformers.__version__,
                "tokenizers": sources,
                "n_pairs": len(pairs),
                "files": [str(f) for f in files],
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    cols = list(summary[0])
    print("\n" + "\t".join(cols))
    for s in summary:
        print("\t".join(str(s[c]) for c in cols))
    print(f"\nWrote {out}/pairs.csv, summary.csv, examples.md, positions/*.json, meta.json")


if __name__ == "__main__":
    main()
