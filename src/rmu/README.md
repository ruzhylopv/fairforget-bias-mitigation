# rmu: pilot

RMU (Representation Misdirection for Unlearning, adapted from
[WMDP](https://github.com/centerforaisafety/wmdp)) applied to gender stereotypes from StereoSet.

## Pilot experiment

Unlearn gender stereotypes from **Lapa** (`lapa-12b-instruct`) on JUPITER and compare the
model before and after.

- **Forget set:** every gender `stereotype` sentence from StereoSet (intrasentence, test + dev).
- **Retain set:** every gender `anti-stereotype` sentence.
- That is `--no-split`: 1026 sentences on each side, 257 batches of 4.

```bash
python -m rmu.unlearn \
    --model_name_or_path lapa-12b-instruct \
    --no-split \
    --max_num_batches 257
```

`--max_num_batches` defaults to 80, so without it only 320 of the 1026 sentences are used.

Evaluation of the before/after models is not implemented yet.

## Open issues

- Lapa is Gemma-3-based and multimodal. The default `--module_str`
  (`{model_name}.model.layers[{layer_id}]`) and `get_params` assume a plain decoder layout, so
  the layer path may need changing before the pilot can run. Check it on the cluster first.
- Running on JUPITER needs the Slurm job scripts (next setup step). GPU work must run inside a
  Slurm job, not on the login node.

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
    --layer_id 1 --layer_ids 0,1 --param_ids 0 --max_num_batches 3 --output_dir runs/tiny
```

Put laptop path overrides in `config/local.yaml` (copy `config/local.example.yaml`).
