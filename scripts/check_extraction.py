"""WP4 (gate 1): checks on the RQ2 extraction output of ``probing.extract``.

A wrong activation file looks exactly like a right one, so before any probe is trained:

    1. single vs saved (GPU)  20 random rows (>= 5 multi-token words) plus the 4 known rows
                              whose token runs past the word, run alone: activations at both
                              positions and every layer match the saved rows (cos >= 0.999,
                              rel. error <= 1%), ``ell`` matches (<= 0.05 nats), and the tokens
                              at the saved positions decode to ``word``.
    2. shared prefix (GPU)    10 pairs, both sentences alone: every position before the first
                              differing token is identical at every layer (causal model).
                              Rerun in float32: bf16 rounding (different kernel shapes for
                              different lengths) shrinks there, a leak would not.
    3. layer 0 (CPU)          rows whose saved token is the same token id have identical
                              ``layer_00`` rows; different ids have different rows.
    4. positions (CPU)        single-token words: first_diff and last_subtok rows identical at
                              every layer; multi-token words: different.
    5. files (CPU)            2 x n_layers files, shape (N, hidden), one float dtype, finite,
                              no all-zero rows, no two consecutive layers identical;
                              validate_meta passes, ``ell`` has no NaN and is <= 0. Prints the
                              mean row norm and the largest |value| per layer.

Prints PASS / FAIL per check, writes ``{root}/acts/{model}/check_extraction.json`` and exits
with status 1 if any check fails.

Usage:
    # login node (CPU checks 3-5, a few minutes: reads all layer files)
    source slurm/env.sh && python scripts/check_extraction.py --model lapa-12b-instruct
    # GPU job (all five checks)
    make submit PURPOSE="rq2: WP4 extraction checks" \\
      CMD="python scripts/check_extraction.py --model lapa-12b-instruct --gpu" JOB_TIME=00:20:00
    # options: --root PATH (default rq2_root)  --tokenizer HUB_ID (if the model folder is not
    #          there, e.g. on a laptop)  --seed 0  --n_rows 20  --n_multi 5  --n_pairs 10
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from probing.extract import _decode_span, _model_inputs, gather_positions, word_logprob
from probing.io import POSITION_COLUMNS, POSITIONS, layer_path, load_layer, load_meta, rq2_root
from probing.io import validate_meta as _validate_meta
from rmu.config import resolve_model

# Sentences whose last token runs past the attribute word ("encourage" -> "▁encouraged"), WP3.
# By sentence_id, since row numbers shift when the meta changes.
KNOWN_OVERRUN = {
    "bbadaf677627088e7c3e2dd4631c013f": "encourage -> encouraged",
    "9afce77a97da5d1a2b03c956ddf58435": "demoralize -> demoralized",
    "451515cbb4bc6bcb3fa6b94c4cb160b8": "protect -> protected",
    "98740dca42c9c302dc2b8b1054883de4": "attack -> attacked",
}
COS_MIN, REL_MAX, ELL_TOL = 0.999, 0.01, 0.05
FP32_REL_MAX = 1e-4  # check 2 in float32: rounding only


class Results:
    def __init__(self):
        self.checks = {}

    def add(self, name, ok, summary, **details):
        self.checks[name] = {"status": "PASS" if ok else "FAIL", "summary": summary, **details}
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {summary}", flush=True)

    @property
    def ok(self):
        return all(c["status"] == "PASS" for c in self.checks.values())


def _rows(idx, limit=10):
    idx = list(map(int, idx))
    return idx[:limit] + (["..."] if len(idx) > limit else [])


# --------------------------------------------------------------------------- shapes


def expected_shape(model, info):
    """(n_hidden_states, hidden) from the model config if available, else from run.json."""
    path = Path(resolve_model(model))
    if (path / "config.json").exists():
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(path).get_text_config()
        return cfg.num_hidden_layers + 1, cfg.hidden_size, "model config"
    return info["n_hidden_states"], info["hidden_size"], "run.json (model config not found)"


# --------------------------------------------------------------------------- checks 4 + 5


def check_files_and_positions(res, meta, model, root, n_states, hidden, store_dtype):
    """One pass over all layer files: check 5 (files) and check 4 (positions agree)."""
    n = len(meta)
    missing = [
        str(layer_path(model, p, k, root))
        for p in POSITIONS
        for k in range(n_states)
        if not layer_path(model, p, k, root).exists()
    ]
    extra = [
        str(f)
        for p in POSITIONS
        for f in sorted((Path(root) / "acts" / model / p).glob("layer_*.npy"))
        if int(f.stem.split("_")[1]) >= n_states
    ]
    problems = []
    if missing:
        problems.append(f"{len(missing)} files missing, e.g. {missing[:3]}")
    if extra:
        problems.append(f"unexpected files beyond layer {n_states - 1}: {extra[:3]}")

    same_idx = (meta["idx_first_diff"] == meta["idx_last_subtok"]).to_numpy()
    single = (meta["n_word_tokens"] == 1).to_numpy()
    pos_problems = []
    if not np.array_equal(same_idx, single):
        pos_problems.append(
            f"idx_first_diff == idx_last_subtok disagrees with n_word_tokens == 1 at rows "
            f"{_rows(np.flatnonzero(same_idx != single))}"
        )

    dtypes, norms, maxabs = set(), {p: [] for p in POSITIONS}, {p: [] for p in POSITIONS}
    prev = {p: None for p in POSITIONS}
    for k in range(n_states):
        arrs = {}
        for p in POSITIONS:
            if not layer_path(model, p, k, root).exists():
                continue
            raw = load_layer(model, p, k, root=root)
            dtypes.add(str(raw.dtype))
            if raw.shape != (n, hidden):
                problems.append(f"{p} layer {k}: shape {raw.shape}, expected {(n, hidden)}")
                continue
            a = np.asarray(raw, dtype=np.float32)
            if not np.isfinite(a).all():
                problems.append(f"{p} layer {k}: {int((~np.isfinite(a)).sum())} non-finite")
            zero = np.flatnonzero((a == 0).all(axis=1))
            if len(zero):
                problems.append(f"{p} layer {k}: all-zero rows {_rows(zero)}")
            if prev[p] is not None and np.array_equal(prev[p], a):
                problems.append(f"{p} layers {k - 1} and {k} are identical")
            norms[p].append(float(np.linalg.norm(a, axis=1).mean()))
            maxabs[p].append(float(np.abs(a).max()))
            prev[p] = arrs[p] = a

        if len(arrs) == 2:  # check 4 at this layer
            equal = (arrs["first_diff"] == arrs["last_subtok"]).all(axis=1)
            bad_single = np.flatnonzero(single & ~equal)
            bad_multi = np.flatnonzero(~single & equal)
            if len(bad_single):
                pos_problems.append(f"layer {k}: single-token rows differ {_rows(bad_single)}")
            if len(bad_multi):
                pos_problems.append(f"layer {k}: multi-token rows identical {_rows(bad_multi)}")

    if len(dtypes) > 1 or not dtypes <= {"float16", "float32"}:
        problems.append(f"dtypes {sorted(dtypes)}; expected one of float16/float32")
    if store_dtype and dtypes and dtypes != {store_dtype}:
        problems.append(f"files are {sorted(dtypes)} but run.json says {store_dtype}")

    try:
        _validate_meta(meta)
    except ValueError as e:
        problems.append(f"validate_meta: {e}")
    ell = meta["ell"].to_numpy()
    if np.isnan(ell).any():
        problems.append(f"ell is NaN at rows {_rows(np.flatnonzero(np.isnan(ell)))}")
    if (ell > 0).any():
        problems.append(f"ell > 0 at rows {_rows(np.flatnonzero(ell > 0))}")
    n_zero = int((ell == 0).sum())

    print("\nlayer  mean norm (first_diff / last_subtok)   max |x| (first_diff / last_subtok)")
    for k in range(len(norms["first_diff"])):
        ratio = norms["first_diff"][k] / norms["first_diff"][k - 1] if k else float("nan")
        print(
            f"  {k:2d}   {norms['first_diff'][k]:10.1f} / {norms['last_subtok'][k]:10.1f}"
            f"   x{ratio:5.2f}   {maxabs['first_diff'][k]:9.4g} / {maxabs['last_subtok'][k]:9.4g}"
        )
    print()

    res.add(
        "5 files",
        not problems,
        f"{2 * n_states} files, ({n}, {hidden}), dtype {sorted(dtypes)}; ell <= 0 "
        f"({n_zero} exactly 0)"
        if not problems
        else "; ".join(problems[:5]),
        problems=problems,
        mean_norm=norms,
        max_abs=maxabs,
        dtypes=sorted(dtypes),
        ell_exact_zero=n_zero,
    )
    res.add(
        "4 positions agree",
        not pos_problems,
        f"{int(single.sum())} single-token rows identical and {int((~single).sum())} "
        f"multi-token rows different at all {n_states} layers"
        if not pos_problems
        else "; ".join(pos_problems[:5]),
        problems=pos_problems,
    )


# --------------------------------------------------------------------------- check 3


def check_layer0(res, meta, model, root, tokenizer):
    problems, groups_info = [], {}
    ids = [tokenizer(s)["input_ids"] for s in meta["sentence"]]
    lens = np.array([len(x) for x in ids])
    if not np.array_equal(lens, meta["n_tokens"].to_numpy()):
        bad = np.flatnonzero(lens != meta["n_tokens"].to_numpy())
        res.add("3 layer 0", False, f"token counts differ from the meta at rows {_rows(bad)}")
        return
    for p in POSITIONS:
        idx = meta[POSITION_COLUMNS[p]].to_numpy()
        tok = np.array([x[len(x) + i] for x, i in zip(ids, idx)])
        a = np.asarray(load_layer(model, p, 0, root=root), dtype=np.float32)
        uniq, first_row, inverse = np.unique(tok, return_index=True, return_inverse=True)
        unequal = [
            int(t) for g, t in enumerate(uniq) if not (a[inverse == g] == a[first_row[g]]).all()
        ]
        if unequal:
            problems.append(f"{p}: rows differ within token ids {unequal[:5]}")
        n_distinct = len(np.unique(a[first_row], axis=0))
        if n_distinct != len(uniq):
            problems.append(f"{p}: {len(uniq)} token ids but {n_distinct} distinct layer-0 rows")
        sizes = np.bincount(inverse)
        top = np.argsort(-sizes)[:3]
        groups_info[p] = {
            "n_token_ids": int(len(uniq)),
            "largest": [
                [tokenizer.convert_ids_to_tokens(int(uniq[g])), int(sizes[g])] for g in top
            ],
        }
    summary = "; ".join(
        f"{p}: {g['n_token_ids']} token ids, largest groups {g['largest']}"
        for p, g in groups_info.items()
    )
    res.add(
        "3 layer 0",
        not problems,
        summary if not problems else "; ".join(problems),
        **{
            "problems": problems,
            "groups": groups_info,
        },
    )


# --------------------------------------------------------------------------- GPU checks


def _compare(single, saved):
    cos = float(single @ saved / max(np.linalg.norm(single) * np.linalg.norm(saved), 1e-30))
    rel = float(np.linalg.norm(single - saved) / max(np.linalg.norm(saved), 1e-30))
    return cos, rel


def pick_rows(meta, n_rows, n_multi, seed):
    rng = np.random.default_rng(seed)
    known = np.flatnonzero(meta["sentence_id"].isin(list(KNOWN_OVERRUN)).to_numpy())
    pool = np.setdiff1d(np.arange(len(meta)), known)
    multi = pool[(meta["n_word_tokens"].to_numpy()[pool] > 1)]
    m = rng.choice(multi, size=min(n_multi, len(multi)), replace=False)
    rest = rng.choice(np.setdiff1d(pool, m), size=n_rows - len(m), replace=False)
    return sorted(map(int, np.concatenate([m, rest]))), [int(r) for r in known]


@torch.no_grad()
def check_single(res, meta, model_name, root, model, tokenizer, rows, known, n_states):
    device = model.get_input_embeddings().weight.device
    saved = {
        p: [load_layer(model_name, p, k, root=root) for k in range(n_states)] for p in POSITIONS
    }
    worst_cos, worst_rel, worst_ell, table, problems = 1.0, 0.0, 0.0, [], []
    for r in rows + known:
        row = meta.iloc[r]
        inputs = _model_inputs(tokenizer, [row["sentence"]], device)
        out = model(**inputs, output_hidden_states=True)
        hs = out.hidden_states
        if len(hs) != n_states:
            problems.append(f"row {r}: {len(hs)} hidden states")
            continue
        ids = inputs["input_ids"][0].cpu()
        if len(ids) != row["n_tokens"]:
            problems.append(f"row {r}: {len(ids)} tokens, meta says {row['n_tokens']}")
            continue
        r_cos, r_rel, worst_at = 1.0, 0.0, None
        for p in POSITIONS:
            idx = int(row[POSITION_COLUMNS[p]])
            for k, h in enumerate(hs):
                a = gather_positions(h, [idx]).float().cpu().numpy()[0]
                cos, rel = _compare(a, np.asarray(saved[p][k][r], dtype=np.float32))
                if worst_at is None or rel > r_rel:
                    worst_at = f"{p} layer {k}"
                r_cos, r_rel = min(r_cos, cos), max(r_rel, rel)
        f, la = int(row["idx_first_diff"]), int(row["idx_last_subtok"])
        ell = float(word_logprob(out.logits, inputs["input_ids"], [f], [la])[0])
        d_ell = abs(ell - float(row["ell"]))
        decoded = _decode_span(tokenizer, ids, f, la).strip()
        t_first = tokenizer.convert_ids_to_tokens(int(ids[f]))
        t_last = tokenizer.convert_ids_to_tokens(int(ids[la]))
        is_known = r in known
        word_ok = decoded.startswith(row["word"]) if is_known else decoded == row["word"]
        ok = r_cos >= COS_MIN and r_rel <= REL_MAX and d_ell <= ELL_TOL and word_ok
        if not ok:
            problems.append(f"row {r}")
        if not is_known:
            worst_cos, worst_rel, worst_ell = (
                min(worst_cos, r_cos),
                max(worst_rel, r_rel),
                max(worst_ell, d_ell),
            )
        table.append(
            {
                "row": r,
                "known_overrun": is_known,
                "word": row["word"],
                "n_word_tokens": int(row["n_word_tokens"]),
                "token_first": t_first,
                "token_last": t_last,
                "decoded": decoded,
                "min_cos": r_cos,
                "max_rel": r_rel,
                "d_ell": d_ell,
                "worst_at": worst_at,
                "ok": ok,
            }
        )

    print("\n  row   word              n  first / last token          min cos    max rel   d ell")
    for t in table:
        mark = " (known overrun)" if t["known_overrun"] else ""
        print(
            f"  {t['row']:4d}  {t['word'][:16]:16s}  {t['n_word_tokens']}  "
            f"{t['token_first'] + ' / ' + t['token_last']:26s}  {t['min_cos']:.6f}  "
            f"{t['max_rel']:.2e}  {t['d_ell']:.3f}"
            f"{'' if t['ok'] else '  <-- FAIL (worst at ' + t['worst_at'] + ')'}{mark}"
        )
    print()
    res.add(
        "1 single vs saved",
        not problems,
        f"{len(rows)} rows ({sum(t['n_word_tokens'] > 1 for t in table if not t['known_overrun'])}"
        f" multi-token) + {len(known)} known overruns, all {n_states} layers, both positions: "
        f"worst cos {worst_cos:.6f}, worst rel {worst_rel:.2e}, worst |d ell| {worst_ell:.4f}"
        if not problems
        else f"failed: {problems[:10]}",
        worst_cos=worst_cos,
        worst_rel=worst_rel,
        worst_d_ell=worst_ell,
        rows=table,
    )


@torch.no_grad()
def prefix_diffs(model, tokenizer, meta, pairs):
    """Per pair: both sentences alone, compare every prefix position at every layer."""
    device = model.get_input_embeddings().weight.device
    details, problems = [], []
    for i in pairs:
        s, a = meta.iloc[2 * i], meta.iloc[2 * i + 1]
        n_prefix = int(s["n_tokens"] + s["idx_first_diff"])  # tokens before first diff, incl. BOS
        if n_prefix != int(a["n_tokens"] + a["idx_first_diff"]):
            problems.append(f"pair {i}: prefix lengths differ")
            continue
        out = {}
        for key, row in (("s", s), ("a", a)):
            inputs = _model_inputs(tokenizer, [row["sentence"]], device)
            out[key] = (inputs["input_ids"][0], model(**inputs, output_hidden_states=True))
        if not torch.equal(out["s"][0][:n_prefix], out["a"][0][:n_prefix]):
            problems.append(f"pair {i}: prefix token ids differ")
            continue
        d = {
            "pair": i,
            "rows": [2 * i, 2 * i + 1],
            "n_prefix": n_prefix,
            "bit_identical": True,
            "min_cos": 1.0,
            "max_rel": 0.0,
            "worst_at": None,
            "max_rel_by_layer": [],
        }
        for k, (hs, ha) in enumerate(zip(out["s"][1].hidden_states, out["a"][1].hidden_states)):
            x = hs[0, :n_prefix].float().cpu().numpy()
            y = ha[0, :n_prefix].float().cpu().numpy()
            d["bit_identical"] &= bool(np.array_equal(x, y))
            layer_rel = 0.0
            for j, (u, v) in enumerate(zip(x, y)):
                cos, rel = _compare(u, v)
                d["min_cos"] = min(d["min_cos"], cos)
                layer_rel = max(layer_rel, rel)
                if rel > d["max_rel"]:
                    d["max_rel"], d["worst_at"] = rel, f"layer {k}, token {j}"
            d["max_rel_by_layer"].append(layer_rel)
        details.append(d)
    return details, problems


def _print_prefix(details, label):
    print(f"\n  shared prefix, {label}")
    print("  pair  rows        prefix  bit-identical  min cos     max rel   worst at")
    for d in details:
        print(
            f"  {d['pair']:4d}  {d['rows'][0]:4d}/{d['rows'][1]:<4d}  {d['n_prefix']:6d}  "
            f"{str(d['bit_identical']):13s}  {d['min_cos']:.6f}  {d['max_rel']:.2e}  "
            f"{d['worst_at'] or '-'}"
        )


def check_prefix(res, meta, model, tokenizer, n_pairs, seed, n_states, fp32=True):
    """bf16 first; then the same pairs in float32, which separates rounding from a leak.

    Two sentences of different lengths go through kernels of different shapes, so bf16 sums
    can differ slightly even with identical prefixes. Rounding shrinks by orders of magnitude in
    float32; a leak between positions would not.
    """
    rng = np.random.default_rng(seed + 1)
    pairs = sorted(map(int, rng.choice(len(meta) // 2, size=n_pairs, replace=False)))
    dtype = str(model.dtype).replace("torch.", "")
    native, problems = prefix_diffs(model, tokenizer, meta, pairs)
    _print_prefix(native, dtype)

    def within(ds, cos_min, rel_max):
        return all(d["min_cos"] >= cos_min and d["max_rel"] <= rel_max for d in ds)

    def worst(ds):
        return max((d["max_rel"] for d in ds), default=0.0)

    fp32_details = None
    if fp32 and model.dtype != torch.float32:
        print("\n  converting the model to float32 for the same pairs ...", flush=True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        model.float()
        fp32_details, more = prefix_diffs(model, tokenizer, meta, pairs)
        problems += more
        _print_prefix(fp32_details, "float32")
    print()

    bit = all(d["bit_identical"] for d in native)
    if problems:
        ok, summary = False, f"failed: {problems}"
    elif within(native, COS_MIN, REL_MAX):
        ok = True
        summary = "bit-identical" if bit else f"within tolerance (worst rel {worst(native):.2e})"
        if fp32_details is not None:
            summary += f"; float32 worst rel {worst(fp32_details):.2e}"
    elif fp32_details is not None and within(fp32_details, 1.0 - FP32_REL_MAX, FP32_REL_MAX):
        ok = True
        summary = (
            f"{dtype} differs by up to rel {worst(native):.2e} (above {REL_MAX:g}), float32 "
            f"by {worst(fp32_details):.2e}: rounding from different kernel shapes, no leak"
        )
    else:
        ok = False
        bad = [d for d in (fp32_details or native) if d["max_rel"] > FP32_REL_MAX]
        summary = (
            f"prefix differs beyond rounding ({'float32' if fp32_details else dtype}): "
            + ", ".join(f"pair {d['pair']} rel {d['max_rel']:.2e} at {d['worst_at']}" for d in bad)
        )
    res.add(
        "2 shared prefix",
        ok,
        f"{len(pairs)} pairs, all {n_states} layers, every prefix position: {summary}",
        bit_identical=bit,
        dtype=dtype,
        pairs=native,
        pairs_float32=fp32_details,
    )


# --------------------------------------------------------------------------- main


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--root", default=None, help="default: paths.rq2_root")
    ap.add_argument("--gpu", action="store_true", help="also run checks 1-2 (loads the model)")
    ap.add_argument("--tokenizer", default=None, help="tokenizer source if not the model's")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_rows", type=int, default=20)
    ap.add_argument("--n_multi", type=int, default=5)
    ap.add_argument("--n_pairs", type=int, default=10)
    ap.add_argument(
        "--no_fp32", action="store_true", help="skip the float32 rerun of check 2 (saves ~1 min)"
    )
    args = ap.parse_args(argv)

    from rmu import utils

    root = Path(args.root) if args.root else rq2_root()
    meta = load_meta(args.model, root=root)
    run_json = root / "acts" / args.model / "run.json"
    info = json.loads(run_json.read_text()) if run_json.exists() else {}
    n_states, hidden, source = expected_shape(args.model, info)
    print(
        f"{args.model}: {len(meta)} rows, expecting {n_states} hidden states x {hidden} "
        f"(from {source}); files in {root}\n",
        flush=True,
    )

    res = Results()
    check_files_and_positions(
        res, meta, args.model, root, n_states, hidden, info.get("store_dtype")
    )

    if args.gpu:
        model, tokenizer = utils.load_model(args.model)
        model.eval()
    else:
        model, tokenizer = None, utils.load_tokenizer(args.tokenizer or args.model)
    check_layer0(res, meta, args.model, root, tokenizer)

    if args.gpu:
        rows, known = pick_rows(meta, args.n_rows, args.n_multi, args.seed)
        check_single(res, meta, args.model, root, model, tokenizer, rows, known, n_states)
        check_prefix(
            res, meta, model, tokenizer, args.n_pairs, args.seed, n_states, not args.no_fp32
        )

    out = root / "acts" / args.model / "check_extraction.json"
    payload = {"model": args.model, "gpu": args.gpu, "seed": args.seed, "checks": res.checks}
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\n" + "\n".join(f"{c['status']}  {name}" for name, c in res.checks.items()))
    print(f"{'ALL PASS' if res.ok else 'FAILED'}; wrote {out}")
    sys.exit(0 if res.ok else 1)


if __name__ == "__main__":
    main()
