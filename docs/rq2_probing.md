# RQ2: linear probing of stereotype vs anti-stereotype

Question: is the stereotype label `S` linearly readable from a frozen model's activations at the
attribute word, beyond what the word's own predictability explains? First run: Lapa-12B-Instruct,
StereoSet intrasentence gender (dev + test), 1,025 pairs = 2,050 sentences (1,026 minus one
misspelled pair, see `EXCLUDE_ITEMS` in `src/probing/meta.py`). Code: `src/probing/`.

## The four steps

1. **Metadata** (CPU): one row per sentence with its label and token positions, built from the
   tokenization check (`results/tokenization-check-nov17/pairs.csv`, nothing is re-tokenized):
   `python -m probing.meta --model lapa-12b-instruct --bias_types gender`
2. **Extraction** (GPU): one frozen forward pass per sentence. Save `hidden_states[k]` at every
   layer and at each token position, and fill `ell` (the attribute word's log-prob) in the meta.
   `src/probing/extract.py`; a few minutes for 2,050 sentences:
   ```bash
   # dry run, first 64 rows, written to {runs_root}/<job>/rq2-dry (rq2_root is not touched)
   make submit PURPOSE="rq2: extract dry run" \
     CMD="python -m probing.extract --model lapa-12b-instruct --limit 64" JOB_TIME=00:20:00
   # full run
   make submit PURPOSE="rq2: extract lapa gender" \
     CMD="python -m probing.extract --model lapa-12b-instruct --bias_types gender" \
     JOB_TIME=00:30:00
   ```
   `ell = Σ_{p = first_diff..last_subtok} log softmax(z_{p-1})[x_p]` (summed, float32). For Lapa
   gender `first_diff` is always the word's first token, so this is the whole word's log-prob.
   Checks that run every time: `attention_mask.sum() == n_tokens` and BOS for every row, the
   number of hidden states, finite activations, `ell` finite and ≤ 0, the decoded tokens at the
   saved positions compared with `word`, and batched vs one-at-a-time activations (left
   padding). Results and provenance go to `acts/{model}/run.json` (git commit read from `.git`,
   since compute nodes have no git; versions; job id; max |activation| per layer; mean `ell`
   by label; attribute-word SS = share of pairs with `ell_stereo > ell_anti`). Existing layer
   files are only replaced with `--overwrite`.
3. **Probes** (CPU): per layer, logistic regression `P(S | a_k)` with outer CV grouped by
   `target` and an inner grouped CV that picks C. References: predictability (logistic
   regression on `ell`), lexical (the same probe at layer 0), null (labels swapped within pairs).
4. **Comparison**: bootstrap CIs over targets, p-values from the null. "Signal beyond
   predictability" = the probe is significantly above both baselines and the null is at ≈50%.

To test steps 3–4 without a GPU, `scripts/make_fake_acts.py` writes the real meta plus Gaussian
activations with a signal planted at chosen layers (`--signal 20,21,22 --delta 2 --root ...`).
`scripts/check_fake_probe.py --root ...` then runs a grouped-CV probe per layer with real and
within-pair-flipped labels: planted layers should be well above 50%, the rest and the flipped
labels at ≈50%.

## Files

Everything lives under `paths.rq2_root` (`config/default.yaml`, override in `config/local.yaml`):

```
{rq2_root}/meta/{model}.csv                              metadata: y, groups, bookkeeping
{rq2_root}/acts/{model}/{position}/layer_{kk:02d}.npy    activations: X, float16 (N, hidden)
```

- **Activations = X.** `kk` indexes `hidden_states` (`00` = embeddings, `00..48` for Lapa,
  hidden size 3840). Stored as float16; if any activation of a model exceeds the float16 range
  (65,504), all its layers are stored as float32 instead (`--store_dtype auto`, recorded as
  `store_dtype` in `run.json`). `load_layer` returns whatever was stored. `position` is `first_diff` or `last_subtok` (`end` comes later).
- **Metadata = y + groups + bookkeeping.** `S` is the label, `target` the CV group, the rest is
  for positions, filters and sanity checks. Positions depend on the tokenizer, so there is one
  meta file per model.

Read and write them only through `probing.io` (`load_meta`, `load_layer`, `save_meta`,
`save_layer`). `validate_meta` runs on every write.

### Row-order rule

Row `i` of the meta is row `i` of **every** layer file. Rows come pair by pair in `pairs.csv`
order, stereotype first, so rows `2i` (S=1) and `2i+1` (S=0) are one StereoSet item. The
within-pair null swaps `y[2i]` and `y[2i+1]` for a random half of the pairs.

### Meta schema

| Column | Meaning |
|---|---|
| `row` | 0..N-1, the row in every layer file |
| `sentence_id`, `item_id` | StereoSet ids (each `item_id` has exactly two rows) |
| `split` | `dev` or `test` (StereoSet file) |
| `bias_type`, `target` | StereoSet fields; `target` is the CV group (40 for gender) |
| `S` | 1 = stereotype, 0 = anti-stereotype |
| `sentence`, `word` | text byte for byte (some have double spaces; positions were computed on them) and the attribute word |
| `n_tokens` | tokens in the sentence, including BOS |
| `idx_first_diff` | first token that differs from the paired sentence, negative from the end (`first_diff - n_tokens`) |
| `idx_last_subtok` | last subtoken of the attribute word, negative from the end |
| `n_word_tokens` | subtokens of the attribute word |
| `target_after_word` | the target (case-insensitive) first appears after the attribute word |
| `ends_with_punct` | `sentence.strip()` ends with `.`, `!` or `?` |
| `ell` | attribute word log-prob; NaN until the extraction step fills it |
| `model` | model short name |

Indices are negative so `x[:, idx]` works on left-padded batches unchanged. They are never BOS
(`idx >= -n_tokens + 1`), and `idx_last_subtok >= idx_first_diff` (equal for single-token words).

## Layer numbering (joining with RQ1)

`hidden_states[j + 1]` is the output of decoder layer `j`, the module RMU hooks with
`--layer_id j`. So the row `layer = j` of `results/layer-sweep-adap-rmu-oct18/06/sweep.csv`
corresponds to `layer_{j+1:02d}.npy`: layer 32 in the sweep is `layer_33.npy`.

One exception: Hugging Face replaces the last entry, `hidden_states[48]`, with the output of
the final norm. `layer_48.npy` is therefore `norm(output of layer 47)`, not the raw output that
the sweep's layer 47 hook sees.
