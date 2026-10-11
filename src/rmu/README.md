# rmu: pilot

RMU (Representation Misdirection for Unlearning, adapted from
[WMDP](https://github.com/centerforaisafety/wmdp)) applied to gender stereotypes from StereoSet.

## Pilot experiment

Unlearn gender stereotypes from **Lapa** (`lapa-12b-instruct`) on JUPITER and compare the
model before and after.

- **Forget set:** every gender `stereotype` sentence from StereoSet (intrasentence, test + dev).
- **Retain set:** every gender `anti-stereotype` sentence.
- That is `--no-split`: 1026 sentences on each side, 257 batches of 4.

On the JUPITER login node, from your clone:

```bash
make env      # once
make smoke    # check GPU + Lapa loading first
make submit PURPOSE="pilot: RMU on Lapa, no-split" JOB_TIME=02:00:00 \
    CMD="python -m rmu.unlearn --model_name_or_path lapa-12b-instruct --no-split --max_num_batches 257"
```

`--max_num_batches` defaults to 80, so without it only 320 of the 1026 sentences are used.

Evaluation of the before/after models is not implemented yet.

## Open issues

- Lapa has 48 layers; the defaults (`--layer_id 7`, `--layer_ids 5,6,7`) come from WMDP's
  32-layer Zephyr and haven't been tuned for Lapa.
- The 2-hour `JOB_TIME` is a guess; check the smoke test's load time and adjust.

## Where results go

Without `--output_dir`, the unlearned model is saved to
`/e/scratch/e-dev-2026d09-075/fairforget-bias-mitigation/runs/<SLURM_JOB_ID>/`. Scratch deletes
files after 90 days: copy anything worth keeping to
`/e/project1/e-dev-2026d09-075/fairforget-bias-mitigation/results/`. Paths are set in
`config/default.yaml`.

## Checking the code on a laptop

```bash
make setup && make test
```

To check the whole pipeline without a GPU, use a tiny random model from the Hugging Face Hub
(downloaded automatically; its losses are meaningless):

```bash
uv run python -m rmu.unlearn --model_name_or_path hf-internal-testing/tiny-random-LlamaForCausalLM \
    --layer_id 1 --layer_ids 0,1 --max_num_batches 3 --output_dir runs/tiny
```

Put laptop path overrides in `config/local.yaml` (copy `config/local.example.yaml`).
