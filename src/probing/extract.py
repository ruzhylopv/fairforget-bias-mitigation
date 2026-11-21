"""RQ2 extraction: one frozen forward pass per sentence of the meta table (GPU).

For every row of the meta table (``run_paths(model, bias).meta``) this saves
``hidden_states[k]`` at the attribute word (positions ``first_diff`` and ``last_subtok``) for
every k, and fills ``ell``, the attribute word's log-probability under the model:

    ell = sum_{p = first_diff .. last_subtok} log softmax(z_{p-1})[x_p]

The model is only read, never trained. Outputs (see docs/rq2_probing.md)::

    {rq2_root}/{bias}/{model}/acts/{position}/layer_{kk:02d}.npy  (N, hidden), float16 (float32
                                                                  on overflow)
    {rq2_root}/{bias}/{model}/meta.csv                            same table, ``ell`` filled
    {rq2_root}/{bias}/{model}/acts/run.json                       provenance + summary stats
    results/rq2/{bias}/{model}/run.json                           the same, as step "extract"

Usage:
    make submit PURPOSE="rq2: extract lapa gender" \\
      CMD="python -m probing.extract --model lapa-12b-instruct --bias gender" \\
      JOB_TIME=00:30:00
    # dry run on the first 64 rows, written to {run_dir}/rq2-dry instead of rq2_root:
    python -m probing.extract --model lapa-12b-instruct --bias gender --limit 64
"""

import argparse
import datetime
import json
import os
import platform
import time

import numpy as np
import pandas as pd
import torch

from probing.io import (
    POSITION_COLUMNS,
    POSITIONS,
    git_commit,
    load_meta,
    record_run,
    run_paths,
    save_layer,
    save_meta,
)
from rmu.config import resolve_model, run_dir

FLOAT16_MAX = float(np.finfo(np.float16).max)  # 65504


# --------------------------------------------------------------------------- pure functions


def _absolute(idx, length: int) -> torch.Tensor:
    """Negative or non-negative indices into a dimension of size ``length`` -> non-negative."""
    idx = torch.as_tensor(np.array(idx), dtype=torch.long)
    if ((idx < -length) | (idx >= length)).any():
        raise IndexError(f"indices {idx.tolist()} out of range for length {length}")
    return torch.where(idx < 0, idx + length, idx)


def gather_positions(hidden: torch.Tensor, idx) -> torch.Tensor:
    """``hidden[b, idx[b]]`` for every row b: (B, T, H) -> (B, H).

    ``idx`` holds one index per row, typically negative (counted from the end), which points
    at the same token in a left-padded batch regardless of the padding.
    """
    if hidden.ndim != 3:
        raise ValueError(f"hidden must be (B, T, H), got shape {tuple(hidden.shape)}")
    b, t, _ = hidden.shape
    pos = _absolute(idx, t).to(hidden.device)
    if pos.shape != (b,):
        raise ValueError(f"need one index per row ({b}), got shape {tuple(pos.shape)}")
    return hidden[torch.arange(b, device=hidden.device), pos]


def word_logprob(logprobs_or_logits: torch.Tensor, input_ids: torch.Tensor, first, last):
    """Summed log-prob of tokens ``first[b] .. last[b]`` (inclusive) of each row: (B,) float32.

    The logits at position p predict the token at p + 1, so token x_p is scored with
    ``log_softmax(z[p - 1])``. ``log_softmax`` is taken in float32 and only at the gathered
    positions; it is idempotent, so log-probs can be passed instead of logits.
    """
    z = logprobs_or_logits
    if z.ndim != 3 or input_ids.shape != z.shape[:2]:
        raise ValueError(
            f"need logits (B, T, V) and input_ids (B, T), got {tuple(z.shape)} and "
            f"{tuple(input_ids.shape)}"
        )
    b, t, _ = z.shape
    first, last = _absolute(first, t), _absolute(last, t)
    if (last < first).any():
        raise ValueError("last must be >= first in every row")
    if (first < 1).any():
        raise ValueError("the first scored token needs a preceding position (first >= 1)")

    span = int((last - first).max()) + 1
    pos = first[:, None] + torch.arange(span)[None, :]  # (B, span)
    valid = pos <= last[:, None]
    pos = torch.where(valid, pos, last[:, None]).to(z.device)
    rows = torch.arange(b, device=z.device)[:, None]
    lp = torch.log_softmax(z[rows, pos - 1].float(), dim=-1)  # (B, span, V)
    tok = input_ids.to(z.device)[rows, pos]  # (B, span)
    per_token = lp.gather(-1, tok[..., None]).squeeze(-1)
    return (per_token * valid.to(z.device)).sum(dim=1)


# --------------------------------------------------------------------------- extraction


def _model_inputs(tokenizer, sentences, device):
    enc = tokenizer(list(sentences), padding=True, return_tensors="pt")
    ids, mask = enc["input_ids"], enc["attention_mask"]
    # Positions as in an unpadded run (left padding would otherwise shift them).
    pos_ids = (mask.cumsum(-1) - 1).clamp(min=0)
    return {
        "input_ids": ids.to(device),
        "attention_mask": mask.to(device),
        "position_ids": pos_ids.to(device),
    }


def _decode_span(tokenizer, ids_row, first, last):
    t = len(ids_row)
    a, b = int(_absolute(first, t)), int(_absolute(last, t))
    return tokenizer.decode(ids_row[a : b + 1].tolist(), skip_special_tokens=False)


@torch.no_grad()
def extract(model, tokenizer, meta: pd.DataFrame, batch_size: int = 32, log_every: int = 10):
    """Run ``meta`` through ``model`` in its own order.

    Returns ``acts`` ({position: [N x hidden float32 array per hidden state]}), ``ell`` (N,)
    and ``checks`` (decode mismatches, padding consistency).
    """
    from rmu.utils import get_decoder_layers

    model.eval()
    device = model.get_input_embeddings().weight.device
    text_config = model.config.get_text_config()
    # Embedding output + one per decoder layer; saved in run.json as n_hidden_states, which
    # every later step reads (never a hard-coded 0-48).
    n_states = len(get_decoder_layers(model)) + 1
    hidden = text_config.hidden_size

    acts = {p: [[] for _ in range(n_states)] for p in POSITIONS}
    ell, mismatches, examples = [], [], []
    n = len(meta)
    for start in range(0, n, batch_size):
        batch = meta.iloc[start : start + batch_size]
        inputs = _model_inputs(tokenizer, batch["sentence"], device)
        ids, mask = inputs["input_ids"], inputs["attention_mask"]
        ids_cpu = ids.cpu()

        got = mask.sum(1).cpu().numpy()
        want = batch["n_tokens"].to_numpy(copy=True)
        if not np.array_equal(got, want):
            bad = np.flatnonzero(got != want)
            raise AssertionError(
                f"token counts differ from the meta (tokenizer mismatch?) at rows "
                f"{(start + bad[:5]).tolist()}: got {got[bad[:5]].tolist()}, "
                f"meta {want[bad[:5]].tolist()}"
            )
        if tokenizer.bos_token_id is not None:
            bos = ids_cpu[torch.arange(len(batch)), ids.shape[1] - torch.as_tensor(want)]
            if not (bos == tokenizer.bos_token_id).all():
                raise AssertionError(f"sentence does not start with BOS in batch at {start}")

        first = batch["idx_first_diff"].to_numpy()
        last = batch["idx_last_subtok"].to_numpy()
        for j, (row, f, la) in enumerate(zip(batch.itertuples(), first, last)):
            text = _decode_span(tokenizer, ids_cpu[j], f, la).strip()
            if text != row.word:
                mismatches.append({"row": int(row.row), "word": row.word, "decoded": text})
            if len(examples) < 5:
                t = ids.shape[1]
                span = ids_cpu[j, int(_absolute(f, t)) : int(_absolute(la, t)) + 1].tolist()
                examples.append(
                    {
                        "row": int(row.row),
                        "word": row.word,
                        "tokens": tokenizer.convert_ids_to_tokens(span),
                        "decoded": text,
                    }
                )

        out = model(**inputs, output_hidden_states=True)
        hs = out.hidden_states
        if len(hs) != n_states:
            raise AssertionError(f"expected {n_states} hidden states, got {len(hs)}")
        for k, h in enumerate(hs):
            for p in POSITIONS:
                idx = batch[POSITION_COLUMNS[p]].to_numpy()
                acts[p][k].append(gather_positions(h, idx).float().cpu())
        ell.append(word_logprob(out.logits, ids, first, last).cpu())

        done = min(start + batch_size, n)
        if (start // batch_size) % log_every == 0 or done == n:
            print(f"  {done}/{n} sentences", flush=True)

    acts = {p: [torch.cat(a).numpy() for a in acts[p]] for p in POSITIONS}
    for p in POSITIONS:
        for k, a in enumerate(acts[p]):
            if a.shape != (n, hidden):
                raise AssertionError(f"{p} layer {k}: shape {a.shape}, expected {(n, hidden)}")
            if not np.isfinite(a).all():
                raise AssertionError(f"{p} layer {k}: non-finite activations")
    ell = torch.cat(ell).numpy().astype(np.float64)
    checks = {
        "decode_mismatches": mismatches,
        "examples": examples,
        "padding_max_rel_diff": padding_check(model, tokenizer, meta, acts),
    }
    return acts, ell, checks


@torch.no_grad()
def padding_check(model, tokenizer, meta, acts, n_rows: int = 4):
    """Max relative difference between batched and one-at-a-time activations (first rows).

    The rows chosen are the shortest of the first batch, i.e. the most-padded ones.
    """
    device = model.get_input_embeddings().weight.device
    n_tok = meta["n_tokens"].to_numpy()[:32]
    rows = np.argsort(n_tok, kind="stable")[:n_rows]
    worst = 0.0
    for r in rows:
        inputs = _model_inputs(tokenizer, [meta["sentence"].iloc[r]], device)
        hs = model(**inputs, output_hidden_states=True).hidden_states
        for p in POSITIONS:
            idx = [int(meta[POSITION_COLUMNS[p]].iloc[r])]
            for k, h in enumerate(hs):
                single = gather_positions(h, idx).float().cpu().numpy()[0]
                batched = acts[p][k][r]
                rel = np.linalg.norm(single - batched) / max(np.linalg.norm(single), 1e-12)
                worst = max(worst, float(rel))
    return worst


# --------------------------------------------------------------------------- summary


def summarize(meta: pd.DataFrame, acts, ell) -> dict:
    """Numbers for the WP3 card: max |activation| per layer, ell by label, attribute-word SS."""
    s = meta["S"].to_numpy()
    ell_s, ell_a = ell[0::2], ell[1::2]  # row-order rule: pairs are (S=1, S=0)
    assert (s[0::2] == 1).all() and (s[1::2] == 0).all()
    return {
        "max_abs": {p: [float(np.abs(a).max()) for a in acts[p]] for p in POSITIONS},
        "ell_mean_stereo": float(ell[s == 1].mean()),
        "ell_mean_anti": float(ell[s == 0].mean()),
        "ell_min": float(ell.min()),
        "ell_max": float(ell.max()),
        "ss_attribute_word": float((ell_s > ell_a).mean()),
        "n_pairs_tied": int((ell_s == ell_a).sum()),
    }


def choose_store_dtype(requested: str, max_abs) -> str:
    """float16 unless a value is beyond its range (``auto``) or float32 was asked for."""
    peak = max(max(v) for v in max_abs.values())
    if requested == "auto":
        return "float16" if peak <= FLOAT16_MAX else "float32"
    if requested == "float16" and peak > FLOAT16_MAX:
        raise ValueError(f"max |activation| {peak:.4g} overflows float16; use --store_dtype auto")
    return requested


# --------------------------------------------------------------------------- CLI


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="short name (config/default.yaml) or path")
    ap.add_argument("--bias", default="gender", help="StereoSet bias type (default: gender)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="first N rows only (dry run)")
    ap.add_argument("--root", default=None, help="where the meta is read (default: rq2_root)")
    ap.add_argument(
        "--out_root",
        default=None,
        help="root the outputs go to (default: --root; with --limit: {run_dir}/rq2-dry)",
    )
    ap.add_argument("--store_dtype", choices=["auto", "float16", "float32"], default="auto")
    ap.add_argument("--overwrite", action="store_true", help="replace existing layer files")
    args = ap.parse_args(argv)

    from rmu.utils import load_model  # heavy import; keeps the pure functions light

    t0 = time.time()
    src = run_paths(args.model, args.bias, root=args.root)
    if args.out_root:
        out = run_paths(args.model, args.bias, root=args.out_root)
    elif args.limit:
        out = run_paths(args.model, args.bias, root=run_dir() / "rq2-dry")
    else:
        out = src

    meta = load_meta(src)  # checks that its model and bias_type columns match
    if args.limit:
        if args.limit % 2:
            raise ValueError("--limit must be even (rows come in pairs)")
        meta = meta.iloc[: args.limit].copy()
    if not args.overwrite and out.layer(POSITIONS[0], 0).exists():
        raise FileExistsError(f"activations already exist under {out.acts}; pass --overwrite")
    print(f"{len(meta)} sentences from {src.meta}, writing to {out.data}", flush=True)

    model, tokenizer = load_model(args.model)
    print(f"loaded {args.model} ({model.dtype}) in {time.time() - t0:.0f}s", flush=True)
    t1 = time.time()
    acts, ell, checks = extract(model, tokenizer, meta, args.batch_size)
    t_forward = time.time() - t1

    if not np.isfinite(ell).all() or (ell > 0).any():
        raise AssertionError(f"ell must be finite and <= 0; got range {ell.min()}..{ell.max()}")
    stats = summarize(meta, acts, ell)
    store_dtype = choose_store_dtype(args.store_dtype, stats["max_abs"])

    for p in POSITIONS:
        for k, a in enumerate(acts[p]):
            save_layer(a, out, p, k, dtype=store_dtype)
    meta["ell"] = ell
    save_meta(meta, out)

    import transformers

    info = {
        "model": args.model,
        "model_path": str(resolve_model(args.model)),
        "model_dtype": str(model.dtype),
        "store_dtype": store_dtype,
        "n_rows": len(meta),
        "n_hidden_states": len(acts[POSITIONS[0]]),
        "hidden_size": int(acts[POSITIONS[0]][0].shape[1]),
        "bias": args.bias,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "meta_source": str(src.meta),
        "git_commit": git_commit(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "seconds_forward": round(t_forward, 1),
        "seconds_total": round(time.time() - t0, 1),
        "padding_max_rel_diff": checks["padding_max_rel_diff"],
        "n_decode_mismatches": len(checks["decode_mismatches"]),
        "decode_mismatches": checks["decode_mismatches"][:20],
        "examples": checks["examples"],
        **stats,
    }
    run_json = out.acts / "run.json"
    run_json.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    record_run(out, "extract", info, stamp=False)  # info has its own date, commit and job id

    print(report(meta, info))
    print(
        f"Wrote {len(POSITIONS) * info['n_hidden_states']} layer files, meta, {run_json} and "
        f"{out.run_json}"
    )


def report(meta, info) -> str:
    peak = {p: max(v) for p, v in info["max_abs"].items()}
    lines = [
        f"runtime: forward {info['seconds_forward']}s, total {info['seconds_total']}s",
        f"hidden states: {info['n_hidden_states']} x {info['hidden_size']}, "
        f"stored as {info['store_dtype']}",
        f"max |activation|: {peak} (float16 max {FLOAT16_MAX:.0f})",
        "max |activation| per layer (first_diff): "
        + " ".join(f"{v:.0f}" for v in info["max_abs"]["first_diff"]),
        f"ell: stereotype mean {info['ell_mean_stereo']:.3f}, anti mean "
        f"{info['ell_mean_anti']:.3f}, range {info['ell_min']:.2f}..{info['ell_max']:.2f}",
        f"attribute-word SS (share of pairs with ell_stereo > ell_anti): "
        f"{info['ss_attribute_word']:.3f} ({info['n_pairs_tied']} ties)",
        f"padding check (batched vs single, max rel diff): {info['padding_max_rel_diff']:.2e}",
        f"decode mismatches: {info['n_decode_mismatches']}",
    ]
    for m in info["decode_mismatches"][:5]:
        lines.append(f"  row {m['row']}: word {m['word']!r}, decoded {m['decoded']!r}")
    lines.append("examples (row: word | tokens at first_diff..last_subtok | ell):")
    for e in info["examples"]:
        ell = meta["ell"].iloc[e["row"]]
        lines.append(f"  {e['row']}: {e['word']!r} | {e['tokens']} | {ell:.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
