"""RQ2 pipeline: every step for one (model, bias type), in order. Started again, it continues.

Stages::

    quick   meta -> extract -> gate 1 (check_extraction --gpu) -> probe -> null (quick seeds)
            -> cgrid -> stats (gate 2)                  GPU job, about 1 h for gender
    null    [gate2.txt all PASS] -> null (all seeds) -> stats       CPU job, long, resumes
    extract meta -> extract -> gate 1 only (the start of quick)      GPU job, minutes; a short
            time limit gets through a busy queue sooner, and quick later skips these steps

Before each step the driver checks whether its outputs are complete and says so ("done,
skipping: ..." or "running: ..."). ``stats`` always runs: it takes minutes and rewrites
``gate2.txt`` from whatever is there. After gate 1 and after ``stats`` the gate file must be all
PASS, or the driver stops. ``--stage null`` refuses to start unless ``gate2.txt`` is all PASS.

If a step has to run while results of later steps already exist (e.g. the activations are
incomplete but probes were fitted), those results would be stale: the driver stops and names
them instead of mixing old and new.

Seed counts come from ``config/rq2/pipeline.yaml``. Each step is its own process (the module
or script it always was), so its output and errors appear in the log as usual.

Usage::

    python -m probing.pipeline --model qwen3-8b --bias race --stage quick [--dry_run]
    python -m probing.pipeline --model qwen3-8b --bias race --stage null
    python -m probing.pipeline --model qwen3-8b --bias race --stage extract
    make rq2 MODEL=qwen3-8b BIAS=race                 # both, as two Slurm jobs
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from probing.controls import BASELINE, complete_seeds, low_edge_probes
from probing.gates import gate1_items, gate_passed, write_gate
from probing.io import (
    POSITIONS,
    bias_config,
    n_decoder_layers,
    read_meta,
    record_run,
    run_paths,
)
from probing.meta import PAIRS_CSV
from rmu.config import PROJECT_ROOT, repo_path

PIPELINE_CONFIG = "config/rq2/pipeline.yaml"
CHECK_SCRIPT = PROJECT_ROOT / "scripts" / "check_extraction.py"


class Stop(Exception):
    """The pipeline cannot continue; the message says why and what to do."""


@dataclass
class Run:
    paths: object  # probing.io.RunPaths
    root: str | None
    n_jobs: int | None
    quick_seeds: int
    null_seeds: int

    def args(self, *, jobs=False) -> list:
        out = ["--model", self.paths.model, "--bias", self.paths.bias]
        out += ["--root", str(self.root)] if self.root else []
        out += ["--n_jobs", str(self.n_jobs)] if jobs and self.n_jobs else []
        return out

    @property
    def null_csv(self) -> Path:
        return self.paths.results / "null_scores.csv"

    @property
    def cgrid_csv(self) -> Path:
        return self.paths.results / "c_grid_low.csv"


@dataclass
class Step:
    name: str
    status: Callable  # Run -> (done: bool, why: str)
    command: Callable  # Run -> [argv]
    stale: Callable = field(default=lambda run: [])  # later outputs a rerun would make stale
    gate: str | None = None  # gate file (in results) that must be all PASS afterwards
    on_skip: Callable | None = None  # Run -> None, when skipped (not in a dry run)


# --------------------------------------------------------------------------- helpers


def _json(path) -> dict:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _summary(run) -> pd.DataFrame | None:
    return pd.read_csv(run.paths.summary) if run.paths.summary.exists() else None


def _existing(*paths) -> list:
    return [p for p in paths if Path(p).exists()]


def _py(*argv) -> list:
    return [sys.executable, *map(str, argv)]


def pipeline_config() -> dict:
    with open(repo_path(PIPELINE_CONFIG), encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def check_inputs(model, bias, paths, pairs_csv=PAIRS_CSV) -> None:
    """Stop early if the run cannot start: no pairs for the model or bias (card [7]), or no
    config/rq2/{bias}.yaml (card [6]). Skipped once the meta table exists."""
    try:
        bias_config(bias)
    except FileNotFoundError as e:
        raise Stop(f"{e} (card [6])") from None
    if paths.meta.exists():
        return
    pairs = pd.read_csv(repo_path(pairs_csv), usecols=["model", "bias_type"], dtype=str)
    if model not in set(pairs["model"]):
        raise Stop(
            f"{model} is not in {pairs_csv}: the tokenization check has not been run for it, so "
            "there are no token positions to build the meta table from. Run "
            "scripts/tokenization_check.py for this model first (card [7])."
        )
    have = sorted(set(pairs.loc[pairs["model"] == model, "bias_type"]))
    if bias not in have:
        raise Stop(f"{pairs_csv} has no {bias} pairs for {model} (only {have}); see card [7].")


# --------------------------------------------------------------------------- step status


def meta_status(run):
    if run.paths.meta.exists():
        return True, f"{run.paths.meta.name} exists"
    return False, "no meta.csv"


def extract_status(run):
    p = run.paths
    info = _json(p.acts / "run.json")
    if not info:
        return False, "no acts/run.json"
    if info.get("limit"):
        return False, f"acts/run.json is a dry run (--limit {info['limit']})"
    n = int(info["n_hidden_states"])
    missing = [(pos, k) for pos in POSITIONS for k in range(n) if not p.layer(pos, k).exists()]
    if missing:
        return False, f"{len(missing)} of {len(POSITIONS) * n} layer files missing"
    if not p.meta.exists() or read_meta(p.meta)["ell"].isna().any():
        return False, "ell is not filled in meta.csv"
    return True, f"{len(POSITIONS) * n} layer files ({n} hidden states), ell filled"


def extract_command(run):
    partial = run.paths.acts.exists() and any(run.paths.acts.rglob("layer_*.npy"))
    return _py("-m", "probing.extract", *run.args()) + (["--overwrite"] if partial else [])


def gate1_status(run):
    gate = run.paths.results / "gate1.txt"
    ok, bad = gate_passed(gate)
    if ok:
        return True, "gate1.txt all PASS"
    payload = _json(run.paths.acts / "check_extraction.json")
    if payload and all(ok is True for _, ok, _ in gate1_items(payload)):
        return True, "all PASS in acts/check_extraction.json (gate1.txt written from it)"
    return False, "; ".join(bad)


def gate1_backfill(run):
    gate = run.paths.results / "gate1.txt"
    if not gate.exists():
        write_gate(gate, gate1_items(_json(run.paths.acts / "check_extraction.json")))


def probe_status(run):
    p = run.paths
    info = _json(p.probes / "run.json")
    if not info:
        return False, "no probes/run.json"
    want = list(range(n_decoder_layers(p) + 1))
    if info.get("layers") != want or sorted(info.get("positions", [])) != sorted(POSITIONS):
        return False, f"probes/run.json covers layers {info.get('layers')}, not 0..{want[-1]}"
    missing = [
        (pos, k)
        for pos in POSITIONS
        for k in want
        if not (p.probes / pos / f"preds_layer_{k:02d}.csv").exists()
    ]
    if missing or not (p.probes / "baseline_ell.csv").exists():
        return False, f"{len(missing)} prediction files missing"
    if not p.summary.exists():
        return False, "no summary.csv"
    return True, f"{len(POSITIONS) * len(want)} probes and the ell baseline, summary.csv"


def null_status(n_seeds):
    def status(run):
        summary = _summary(run)
        if summary is None or not run.null_csv.exists():
            return False, f"no null_scores.csv for seeds 0-{n_seeds - 1}"
        probes = [
            (pos, int(k))
            for pos, k in zip(summary["position"], summary["layer"])
            if pos != BASELINE[0]
        ]
        have = set(complete_seeds(pd.read_csv(run.null_csv), probes))
        missing = sorted(set(range(n_seeds)) - have)
        if missing:
            return False, f"{n_seeds - len(missing)} of {n_seeds} seeds complete"
        return True, f"seeds 0-{n_seeds - 1} complete for all {len(probes)} probes"

    return status


def null_command(n_seeds):
    return lambda run: _py(
        "-m", "probing.controls", "null", *run.args(jobs=True), "--seeds", f"0-{n_seeds - 1}"
    )


def cgrid_status(run):
    summary = _summary(run)
    if summary is None:
        return False, "no summary.csv yet (decides which probes to refit)"
    need = low_edge_probes(summary)
    if not need:
        return True, "no probe chose the smallest C"
    have = set()
    if run.cgrid_csv.exists():
        c = pd.read_csv(run.cgrid_csv)
        have = set(zip(c["position"], c["layer"].astype(int)))
    missing = sorted(set(need) - have)
    if missing:
        return False, f"{len(missing)} of {len(need)} probes with C at the low edge not refitted"
    return True, f"all {len(need)} probes with C at the low edge refitted"


def always(run):
    return False, "always runs (writes gate2.txt)"


# --------------------------------------------------------------------------- stages


def steps(stage) -> list:
    stats = Step(
        "stats",
        always,
        lambda run: _py("-m", "probing.controls", "stats", *run.args()),
        gate="gate2.txt",
    )
    later_than_probe = lambda run: _existing(run.null_csv, run.cgrid_csv)  # noqa: E731
    later_than_extract = lambda run: (  # noqa: E731
        _existing(run.paths.probes / "run.json", run.paths.summary) + later_than_probe(run)
    )
    if stage == "extract":
        return steps("quick")[:3]  # meta, extract, gate 1
    if stage == "quick":
        return [
            Step(
                "meta",
                meta_status,
                lambda run: _py("-m", "probing.meta", *run.args()),
                stale=lambda run: _existing(run.paths.acts / "run.json") + later_than_extract(run),
            ),
            Step("extract", extract_status, extract_command, stale=later_than_extract),
            Step(
                "gate 1",
                gate1_status,
                lambda run: _py(CHECK_SCRIPT, *run.args(), "--gpu"),
                gate="gate1.txt",
                on_skip=gate1_backfill,
            ),
            Step(
                "probe",
                probe_status,
                lambda run: _py("-m", "probing.probe", *run.args(jobs=True)),
                stale=later_than_probe,
            ),
            Step(
                "null (quick)",
                lambda run: null_status(run.quick_seeds)(run),
                lambda run: null_command(run.quick_seeds)(run),
            ),
            Step(
                "cgrid",
                cgrid_status,
                lambda run: _py("-m", "probing.controls", "cgrid", *run.args(jobs=True)),
            ),
            stats,
        ]
    if stage == "null":
        return [
            Step(
                "null",
                lambda run: null_status(run.null_seeds)(run),
                lambda run: null_command(run.null_seeds)(run),
            ),
            stats,
        ]
    raise ValueError(f"unknown stage {stage!r}; expected quick, null or extract")


def _status(step, run):
    try:
        return step.status(run)
    except Exception as e:  # e.g. the layer count is unknown before extraction
        return False, f"cannot tell yet ({type(e).__name__}: {e})"


def run_stage(run, stage, dry_run=False, log=print, outcome=None) -> dict:
    """Run (or, with ``dry_run``, describe) the steps of ``stage``. Returns {step: outcome};
    pass ``outcome`` to keep what happened when it raises.

    Raises Stop when the pipeline cannot continue (failed step, gate not passed, stale
    results).
    """
    p = run.paths
    tag = f"[rq2 {p.model}/{p.bias} {stage}]"
    outcome = {} if outcome is None else outcome
    if stage == "null":
        ok, bad = gate_passed(p.results / "gate2.txt")
        if not ok:
            raise Stop(
                "--stage null needs gate2.txt all PASS (run --stage quick first): " + "; ".join(bad)
            )
    for step in steps(stage):
        done, why = _status(step, run)
        if done:
            log(f"{tag} {step.name}: done, skipping ({why})")
            outcome[step.name] = f"skipped: {why}"
            if step.on_skip and not dry_run:
                step.on_skip(run)
            continue
        stale = step.stale(run)
        if stale:
            raise Stop(
                f"{step.name} has to run ({why}), but results of later steps exist and would "
                f"be stale: {', '.join(map(str, stale))}. Move them away and start again."
            )
        cmd = step.command(run)
        shown = " ".join(str(c) for c in [Path(cmd[0]).name, *cmd[1:]])
        if dry_run:
            log(f"{tag} {step.name}: would run ({why}): {shown}")
            outcome[step.name] = f"would run: {why}"
            continue
        log(f"{tag} {step.name}: running ({why}): {shown}")
        t0 = time.time()
        env = os.environ | {"PYTHONUNBUFFERED": "1"}
        code = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env).returncode
        minutes = (time.time() - t0) / 60
        if code:
            outcome[step.name] = f"failed (exit {code}) after {minutes:.1f} min"
            raise Stop(f"{step.name} failed with exit code {code}; see the log above")
        outcome[step.name] = f"ran ({minutes:.1f} min)"
        log(f"{tag} {step.name}: finished in {minutes:.1f} min")
        if step.gate:
            ok, bad = gate_passed(p.results / step.gate)
            if not ok:
                outcome[step.name] += f"; {step.gate} not all PASS"
                raise Stop(f"{step.gate} is not all PASS: " + "; ".join(bad))
    return outcome


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--bias", required=True, help="StereoSet bias type")
    ap.add_argument("--stage", required=True, choices=["quick", "null", "extract"])
    ap.add_argument("--dry_run", action="store_true", help="print the steps, run nothing")
    ap.add_argument("--root", default=None, help="data root (fake data); default rq2_root")
    ap.add_argument("--n_jobs", type=int, default=None, help="passed to probe / controls")
    cfg = pipeline_config()
    ap.add_argument("--quick_seeds", type=int, default=cfg["quick_null_seeds"])
    ap.add_argument("--null_seeds", type=int, default=cfg["null_seeds"])
    args = ap.parse_args(argv)

    paths = run_paths(args.model, args.bias, root=args.root)
    run = Run(paths, args.root, args.n_jobs, args.quick_seeds, args.null_seeds)
    print(
        f"[rq2 {args.model}/{args.bias} {args.stage}] data {paths.data}, results {paths.results}"
        + (" (dry run)" if args.dry_run else ""),
        flush=True,
    )
    outcome, error = {}, None
    try:
        check_inputs(args.model, args.bias, paths)
        run_stage(run, args.stage, args.dry_run, lambda s: print(s, flush=True), outcome)
    except Stop as e:
        error = str(e)
    if outcome and not args.dry_run:  # nothing to record if it could not even start
        record_run(
            paths,
            f"pipeline {args.stage}",
            {"steps": outcome, "result": "stopped: " + error if error else "ok"},
        )
    if error:
        print(f"[rq2 {args.model}/{args.bias} {args.stage}] STOPPED: {error}", flush=True)
        sys.exit(1)
    done = "dry run, nothing ran" if args.dry_run else "all steps done"
    print(f"[rq2 {args.model}/{args.bias} {args.stage}] {done}", flush=True)


if __name__ == "__main__":
    main()
