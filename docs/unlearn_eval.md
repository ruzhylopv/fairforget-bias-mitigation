# Unlearn + evaluate in one job

`python -m evaluation.unlearn_eval` runs **adaptive RMU** on StereoSet gender sentences and then
evaluates the unlearned model on **BBQ** (Gender_identity) without reloading it. `make unlearn-eval`
submits it as a Slurm job.

## Defaults

The settings of the oct18 adaptive RMU runs. Any `rmu.unlearn` option overrides them.

| Option | Default | Meaning |
|---|---|---|
| `--model_name_or_path` | **required** | short name from `config/default.yaml`, local path or hub id |
| `--beta` | 1 | control vector = β · ‖frozen activation of the token‖ · u (adaptive RMU) |
| `--lr` | 5e-4 | AdamW learning rate |
| `--alpha` | 5 | retain loss weight |
| `--layer_id` / `--layer_ids` | 32 / 30,31,32 | hooked layer / layers whose `mlp.down_proj.weight` is trained |
| `--split` | on | forget = stereotypes of half the items, retain = anti-stereotypes of the other half |
| `--loss_positions` | words | losses over word tokens only (BOS/EOS/padding masked) |
| `--max_num_batches` | 257 | clamped to all batches: 129 steps with `--split` |
| `--save_model` | on | `--no-save_model` keeps only losses and eval results |
| `--eval_batch_size`, `--eval_limit`, `--gender_types` | 8, all, all | BBQ options |

Why layer 32: at deeper layers the forget loss drops by rotating the activations towards the
control direction instead of shrinking them. See `results/adaptive-rmu-oct18/05/05.ipynb` and the
layer sweep in `results/layer-sweep-adap-rmu-oct18/06/`.

## Running it on JUPITER

From your clone on the login node (`make env` done once):

```bash
make unlearn-eval MODEL=lapa-12b-instruct
make unlearn-eval MODEL=gemma3-12b-it
make unlearn-eval MODEL=qwen3-8b EXTRA_ARGS="--layer_id 24 --layer_ids 22,23,24"
```

`MODEL` defaults to `lapa-12b-instruct`, `JOB_TIME` to `02:00:00`. `EXTRA_ARGS` takes any option
from the table, e.g. `EXTRA_ARGS="--lr 1e-3 --no-save_model"`. The same job without Make:

```bash
make submit PURPOSE="unlearn+eval: gemma, alpha 10" JOB_TIME=02:00:00 \
    CMD="python -m evaluation.unlearn_eval --model_name_or_path gemma3-12b-it --alpha 10"
```

Model notes:

- **Lapa 12B** and **Gemma 3 12B** have 48 layers, so layer 32 is ≈ 67% depth.
- **Qwen3 8B** has 36 layers: layer 32 is ≈ 89% depth. Use `--layer_id 24 --layer_ids 22,23,24`
  for the same relative depth as on Lapa/Gemma.
- The trained parameter is looked up by name (`mlp.down_proj.weight`), so all three work as is.

First time, or after code changes, do a quick check:

```bash
make unlearn-eval MODEL=qwen3-8b JOB_TIME=00:30:00 EXTRA_ARGS="--eval_limit 50 --no-save_model"
```

Follow the job with `make jobs` and `make watch JOB=<id> LOSSES=1`.

## Outputs

One folder: `runs_root/<JOB_ID>/<model>_alpha-5_batches-129_layer-32_beta-1_positions-words/`

| File | Content |
|---|---|
| model files | the unlearned model (≈ 24 GB for 12B; skipped with `--no-save_model`) |
| `losses.csv`, `losses.png` | per-step RMU losses and shrink/rotation diagnostics |
| `rmu_config.json` | all arguments of the run |
| `bbq/eval_summary.csv` | accuracy, S_DIS, S_AMB |
| `bbq/eval_results.json`, `bbq/predictions.jsonl` | full metrics and per-item predictions |

Compare `bbq/eval_summary.csv` with the original model's baseline in
`results/bbq-eval/baseline-<model>/eval_summary.csv` (lapa-12b, gemma3-12b, qwen3-8b). The job log
prints both paths at the end. Lower |S_AMB| / |S_DIS| at similar accuracy means less bias.

Scratch deletes files after 90 days: copy `bbq/`, `losses.csv` and `rmu_config.json` to
`/e/project1/e-dev-2026d09-075/fairforget-bias-mitigation/results/` and fill in the job's row in
`usage_log.tsv`.

## Laptop check

Runs the whole pipeline with a tiny random model (downloaded from the Hub). The losses and scores
are meaningless, and the losses may go NaN in fp16 on CPU:

```bash
uv run python -m evaluation.unlearn_eval \
    --model_name_or_path hf-internal-testing/tiny-random-LlamaForCausalLM \
    --layer_id 1 --layer_ids 0,1 --max_num_batches 2 --eval_limit 8 --no-save_model \
    --output_dir runs/tiny-eval
```
