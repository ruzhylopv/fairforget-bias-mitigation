# RQ2: linear probing of stereotype vs anti-stereotype

Question: is the stereotype label `S` linearly readable from a frozen model's activations at the
attribute word, beyond what the word's own predictability explains? First run: Lapa-12B-Instruct,
StereoSet intrasentence gender (dev + test), 1,025 pairs = 2,050 sentences (1,026 minus one
misspelled pair, see `config/rq2/gender.yaml`). Code: `src/probing/`. Every step takes
`--model` and `--bias` (default `gender`) and finds its files through
`probing.io.run_paths(model, bias)` (see [Files](#files)).

## One command

```bash
make rq2 MODEL=qwen3-8b BIAS=race              # on the login node: two Slurm jobs
make rq2 MODEL=qwen3-8b BIAS=race DRY_RUN=1    # print the steps and what is already done
```

`slurm/rq2.sh` first dry-runs `python -m probing.pipeline` on the login node (a model missing
from `pairs.csv`, card [7], or a missing `config/rq2/{bias}.yaml`, card [6], fails here), then
submits:

- **job A** (`--stage quick`, GPU node, ~1 h for gender): meta → extraction → gate 1
  (`scripts/check_extraction.py --gpu` → `gate1.txt`) → probes → null with
  `quick_null_seeds` (5) → C grid for the probes whose C hit the bottom of the grid → stats
  (`gate2.txt`);
- **job B** (`--stage null`, `--dependency=afterok:A`): the null with `null_seeds` (100,
  `config/rq2/pipeline.yaml`) → stats. It refuses to start unless `gate2.txt` is all PASS.
  `NULL_PIECES=n` splits it into n jobs in a row (`afterany`), each continuing the last. If A
  fails, B stays pending forever: `scancel` it.

Both jobs use the whole node (288 CPUs). Each step checks its outputs first and logs
`done, skipping (...)` or `running (...)`: complete activations, `gate1.txt` (or a passing
`acts/check_extraction.json`) all PASS, probes for every layer `0..n_layers` (from
`acts/run.json`, which extraction fills with `len(get_decoder_layers(model)) + 1` hidden
states), null seeds complete. `stats` always runs (minutes). So after a failure or a time
limit, **submit the same line again**. The driver stops after a gate that is not all PASS,
and also when a step has to run while results of later steps exist (they would be stale):
then move those away by hand. The outcome of each stage goes to `run.json`
(`pipeline quick` / `pipeline null`).

## Report over all runs

`uv run python scripts/rq2_report.py` (laptop) reads every finished run in
`results/rq2/{bias}/{model}/` and writes to `results/rq2/`:

- `all_runs.csv`: every (position, layer, metric) row of every run (`ci.csv` joined with
  `pvalues.csv` and `summary.csv`) with `model`, `bias` and run-level columns (layers, pairs,
  targets, fixed layer, gate status). Everything below is made from it.
- `report_table.csv` / `.tex` (booktabs): one row per (model, bias): `ell` and layer 0 with
  CIs; at the fixed depth the probe, its gaps vs `ell` and vs layer 0, and the gap split by
  target before / after the word; the best layer as relative depth with `p_corr` (post hoc);
  gates; a flag below `few_targets_below` targets (`config/rq2/report.yaml`).
- `fig_gap_by_depth` (small multiples, bias type × model, gap vs `ell` against relative
  depth) and `fig_forest` (both gaps at the fixed depth, one row per run), PNG + PDF.

Runs are compared by their gaps, not raw accuracy: each has its own baseline. `--metric acc`
for per-sentence accuracy (default: pairwise).

## The four steps

1. **Metadata** (CPU): one row per sentence with its label and token positions, built from the
   tokenization check (`results/tokenization-check-nov17/pairs.csv`, nothing is re-tokenized):
   `python -m probing.meta --model lapa-12b-instruct --bias gender`. Items left out are listed,
   with a reason, under `exclude_items` in `config/rq2/{bias}.yaml`.
2. **Extraction** (GPU): one frozen forward pass per sentence. Save `hidden_states[k]` at every
   layer and at each token position, and fill `ell` (the attribute word's log-prob) in the meta.
   `src/probing/extract.py`; a few minutes for 2,050 sentences:
   ```bash
   # dry run, first 64 rows, written to {runs_root}/<job>/rq2-dry (rq2_root is not touched)
   make submit PURPOSE="rq2: extract dry run" JOB_TIME=00:20:00 \
     CMD="python -m probing.extract --model lapa-12b-instruct --bias gender --limit 64"
   # full run
   make submit PURPOSE="rq2: extract lapa gender" JOB_TIME=00:30:00 \
     CMD="python -m probing.extract --model lapa-12b-instruct --bias gender"
   ```
   `ell = Σ_{p = first_diff..last_subtok} log softmax(z_{p-1})[x_p]` (summed, float32). For Lapa
   gender `first_diff` is always the word's first token, so this is the whole word's log-prob.
   Checks that run every time: `attention_mask.sum() == n_tokens` and BOS for every row, the
   number of hidden states, finite activations, `ell` finite and ≤ 0, the decoded tokens at the
   saved positions compared with `word`, and batched vs one-at-a-time activations. Sentences
   run one at a time (`--batch_size 1`, the default): in bf16 Qwen3's activations depend on
   the batch shape (1-2%, `ell` up to 0.16 nats; with left padding and with equal-length
   batches alike), lapa's do not (bit-identical). So every model's activations are those of
   the sentence run alone. Results and provenance go to `acts/run.json` and to the committed `run.json` as
   step `extract` (git commit read from `.git`,
   since compute nodes have no git; versions; job id; max |activation| per layer; mean `ell`
   by label; attribute-word SS = share of pairs with `ell_stereo > ell_anti`). Existing layer
   files are only replaced with `--overwrite`.
3. **Probes** (CPU): per layer, logistic regression `P(S | a_k)` with outer CV grouped by
   `target` and an inner grouped CV that picks C. References: predictability (logistic
   regression on `ell`), lexical (the same probe at layer 0), null (labels swapped within pairs).
   `src/probing/probe.py`:
   ```bash
   make submit PURPOSE="rq2: probes lapa gender" JOB_CPUS=72 JOB_TIME=01:00:00 \
     CMD="python -m probing.probe --model lapa-12b-instruct --bias gender --layers 0-48"
   ```
   - Pipeline `StandardScaler + LogisticRegression(max_iter=5000)`; the scaler matters because
     a few of lapa's dimensions reach ~1e5.
   - Outer `GroupKFold(5)` by target, saved once as `probes/folds.csv` and reused by
     the baseline and WP6. Inner `GridSearchCV` over C in {1e-4 … 1}, `GroupKFold(4)` on the
     training targets, selected by log-loss. The outer loop is explicit so the groups reach
     the inner split. Baseline: the same pipeline on `ell` alone, C = 1e4, no inner search.
   - Scores from the held-out predictions: per-sentence accuracy (z > 0), pairwise accuracy
     (z of the stereotype row > z of the anti row, ties ½; computed from z, not p, because p
     saturates) and log-loss, each overall, `_tfirst` (target before the word) and `_tafter`
     (target after the word in either row of the pair).
   - `C_edge` in the summary: `low` (flattest probe fits best, normal at chance layers) or
     `high` (wants less regularisation: widen `--grid` upwards).
   - Weights for RQ3 (`probe_layer_kk.npz`): refit on all rows with the most frequent C and
     mapped back to raw activations (`w_raw = w'/s`, `b_raw = b' − Σ w'μ/s`), so
     `X @ w_raw + b_raw` is the decision function on raw X. The scaled `coef_` is not a
     direction in activation space.
   - Outputs: `probes/{position}/preds_layer_kk.csv` (row, fold, z, p, C),
     `probe_layer_kk.npz`, `probes/baseline_ell.csv`, `probes/run.json` (settings; WP6 reads
     the grid from it), and the committed `summary.csv`.
   - To test on fake data: `scripts/make_fake_acts.py --layers 19-23 --signal 20,21,22
     --delta 2 --root R` then `python -m probing.probe --model lapa-12b-instruct --layers 19-23
     --root R` gives ~0.92 per sentence on 20–22, ~0.50 elsewhere. With `--root`, the
     results go to `R/results/...`, not into the repo.
4. **Comparison**: bootstrap CIs over targets, p-values from the null. "Signal beyond
   predictability" = the probe is significantly above both baselines and the null is at ≈50%.
   `src/probing/controls.py` (WP6, gate 2), CPU, on the WP5 folds and predictions:
   ```bash
   # null: labels swapped in a random half of the pairs, full nested CV per probe and seed
   make submit PURPOSE="rq2 wp6: null 5 seeds" JOB_CPUS=288 JOB_TIME=01:00:00 \
     CMD="python -m probing.controls null --model lapa-12b-instruct --bias gender --seeds 0-4"
   make submit PURPOSE="rq2 wp6: null 100 seeds" JOB_CPUS=288 JOB_TIME=12:00:00 \
     CMD="python -m probing.controls null --model lapa-12b-instruct --bias gender --seeds 0-99"
   # layers 0-7 with C down to 1e-6 (scores only; WP5 predictions untouched)
   make submit PURPOSE="rq2 wp6: C grid 0-7" JOB_CPUS=72 JOB_TIME=00:30:00 \
     CMD="python -m probing.controls cgrid --model lapa-12b-instruct --bias gender --layers 0-7"
   # p-values, bootstrap CIs, misspellings, gate 2 checklist (minutes)
   make submit PURPOSE="rq2 wp6: stats" JOB_CPUS=8 JOB_TIME=00:20:00 \
     CMD="python -m probing.controls stats --model lapa-12b-instruct --bias gender"
   uv run python scripts/plot_controls.py --model lapa-12b-instruct --bias gender   # laptop
   uv run python scripts/plot_probes.py --model lapa-12b-instruct --bias gender     # laptop
   ```
   - `null` appends one row per (seed, probe) to `null_scores.csv` as fits finish and skips
     rows already there, so rerunning the same command after a time limit resumes it. Seeds
     0-4 are the first five of 0-99. Pairwise accuracy follows the flipped labels.
   - `pvalues.csv`: `p` against the same probe's null, `p_corr` against the max over all
     probes of each shuffle (seeds with every probe only). Smallest p is 1/(B+1).
   - `ci.csv`: 2,000 resamples of the 40 targets (all rows of each drawn target), the same
     draws for every probe and the baseline, so the gaps are paired: `gap*` = probe − `ell`
     (predictability), `gap_layer0*` = probe − layer 0 of the same position (lexical
     baseline: the word's identity, no context; NaN on the `ell` row), for every metric and
     subset. RQ2 needs the probe above both. The fitted probes are treated as fixed (no
     refitting variance). `rel_depth` = layer / n_layers (also in `summary.csv`), with
     n_layers the decoder layers (48 lapa/gemma3, 36 qwen3, 32 Llama), read from
     `acts/run.json`.
   - `robustness_no_misspellings.csv`: the summary without the pairs whose word is listed
     under `misspelled` in `config/rq2/{bias}.yaml` (gender: `mainicure`, `uncertanties`,
     `forebearance`), from the stored predictions (no refit).
   - `gate2.txt`: the checklist as far as it can be computed (null: every probe's mean over
     seeds within 0.50 ± 0.03, since a leak shifts the mean; single runs may leave the band,
     their spread is ~1.3× binomial for lapa/gender because folds are grouped by target; fold
     balance, folds
     shared by probe and baseline and reproducing `summary.csv`, C grid), then the probe at
     the **fixed depth** and the peak layer.
   - **Fixed depth** (`config/rq2/report.yaml`): chosen once, 2026-11-21, on lapa/gender
     before any other run was looked at: layer 20 of 48 (`rel_depth` 0.417, `last_subtok`),
     the middle of the WP5 plateau (layers 15–26), not its peak. Every new run is reported at
     its nearest layer (qwen3 15, Llama 13, gemma3 20), with the per-probe p. The best layer
     may still be shown, but only with `p_corr`. lapa/gender is marked as the run the depth
     was chosen on (not an independent test).

To test steps 3–4 without a GPU, `scripts/make_fake_acts.py` writes the real meta plus Gaussian
activations with a signal planted at chosen layers (`--signal 20,21,22 --delta 2 --root ...`).
`scripts/check_fake_probe.py --root ...` then runs a grouped-CV probe per layer with real and
within-pair-flipped labels: planted layers should be well above 50%, the rest and the flipped
labels at ≈50%.

## Files

One run = one model on one StereoSet bias type, with one folder for the big files (under
`paths.rq2_root`, `config/default.yaml`, override in `config/local.yaml`) and one committed folder
for the small results. `probing.io.run_paths(model, bias)` returns all of them, and every step
uses it, so a race run on lapa never touches the gender files:

```
{rq2_root}/{bias}/{model}/meta.csv                            metadata: y, groups, bookkeeping
{rq2_root}/{bias}/{model}/acts/{position}/layer_{kk:02d}.npy  activations: X, float16 (N, hidden)
{rq2_root}/{bias}/{model}/acts/run.json                       extraction provenance + stats
{rq2_root}/{bias}/{model}/probes/                             folds, predictions, probe weights
results/rq2/{bias}/{model}/                                   summary.csv, WP6 tables, plots
results/rq2/{bias}/{model}/run.json                           date, git commit, job id, settings
                                                              of every step (meta, extract,
                                                              probe, controls ...)
config/rq2/{bias}.yaml                                        exclude_items (id: reason),
                                                              misspelled (words)
```

Dates and commits go into `run.json`, never into folder names; a rerun of a step replaces its
entry. `--root R` (every step) replaces `rq2_root` and sends the results to `R/results/...`, so
fake data and dry runs never overwrite committed results. Runs from before this layout were
moved once with `scripts/migrate_rq2_layout.py --model M --bias B [--apply]` (prints the plan
without `--apply`, refuses while project jobs are queued). For lapa/gender,
`results/rq2/gender/lapa-12b-instruct/labels.csv` is the committed meta without `ell`; a test
checks that `probing.meta` still rebuilds it exactly.

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
