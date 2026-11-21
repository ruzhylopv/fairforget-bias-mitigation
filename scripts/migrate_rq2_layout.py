"""Once per run: move the big RQ2 files from the old per-model layout to ``run_paths``.

    old                                   new
    {rq2_root}/meta/{model}.csv       ->  {rq2_root}/{bias}/{model}/meta.csv
    {rq2_root}/acts/{model}/          ->  {rq2_root}/{bias}/{model}/acts/
    {rq2_root}/probes/{model}/        ->  {rq2_root}/{bias}/{model}/probes/

It also fills ``results/rq2/{bias}/{model}/run.json`` (in this clone) with the steps "extract"
and "probe" copied from the old ``acts/run.json`` and ``probes/run.json`` (their own dates; the
old probe run recorded no commit), plus a "migration" step. Commit that file afterwards.

Without ``--apply`` it only prints the plan. It refuses to run if a target already exists, if
the meta's bias type is not ``--bias``, and (unless ``--force``) while the project has jobs in
the queue: a running job may be reading the files, e.g. the 100-seed null of probing.controls.

Usage (login node, from a clone):
    source slurm/env.sh
    python scripts/migrate_rq2_layout.py --model lapa-12b-instruct --bias gender           # plan
    python scripts/migrate_rq2_layout.py --model lapa-12b-instruct --bias gender --apply
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from probing.io import read_meta, record_run, rq2_root, run_paths

ACCOUNT = "e-dev-2026d09-075"  # as in slurm/run.sbatch


def old_paths(model, root) -> dict:
    root = Path(root)
    return {
        "meta": root / "meta" / f"{model}.csv",
        "acts": root / "acts" / model,
        "probes": root / "probes" / model,
    }


def plan(model, bias, root=None, results_root=None):
    """[(old, new)] for the files that exist, plus the run paths and a list of problems."""
    new = run_paths(model, bias, root=root, results_root=results_root)
    old = old_paths(model, rq2_root() if root is None else root)
    moves, problems = [], []
    for key, src in old.items():
        dst = getattr(new, key)
        if dst.exists():
            problems.append(f"target exists: {dst}")
        if src.exists():
            moves.append((src, dst))
        elif key != "probes":  # probes may not have run yet
            problems.append(f"missing: {src}")
    if old["meta"].exists():
        found = set(read_meta(old["meta"])["bias_type"])
        if found != {bias}:
            problems.append(f"{old['meta']} has bias types {sorted(found)}, not {bias!r}")
    return moves, new, problems


def queued_jobs() -> list:
    """Jobs of the project account in the queue (empty if squeue is not available)."""
    try:
        out = subprocess.run(
            ["squeue", "-h", "-A", ACCOUNT, "-o", "%i %u %T %j"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line for line in out.splitlines() if line.strip()]


def migrate(moves, new, old_root) -> None:
    """Move the files, then record the old run.json files in ``new.run_json``."""
    new.data.mkdir(parents=True, exist_ok=True)
    for src, dst in moves:
        shutil.move(str(src), str(dst))
        print(f"moved {src} -> {dst}")
    old = old_paths(new.model, old_root)
    for step, key in (("extract", "acts"), ("probe", "probes")):
        run_json = getattr(new, key) / "run.json"
        if run_json.exists():
            info = json.loads(run_json.read_text(encoding="utf-8"))
            info.setdefault("git_commit", None)
            info["migrated_from"] = str(old[key] / "run.json")
            record_run(new, step, info, stamp=False)
    record_run(new, "migration", {"moved": {str(s): str(d) for s, d in moves}})
    print(f"wrote {new.run_json}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--bias", required=True, help="the bias type the old files hold")
    ap.add_argument("--root", default=None, help="default: paths.rq2_root")
    ap.add_argument("--apply", action="store_true", help="move (default: print the plan only)")
    ap.add_argument("--force", action="store_true", help="move even with jobs in the queue")
    args = ap.parse_args(argv)

    moves, new, problems = plan(args.model, args.bias, root=args.root)
    old_root = rq2_root() if args.root is None else Path(args.root)
    for src, dst in moves:
        print(f"{'move' if args.apply else 'would move'} {src} -> {dst}")
    print(f"run.json: {new.run_json}")
    jobs = queued_jobs()
    if jobs and not args.force:
        problems.append(
            "jobs in the queue (they may read these files; wait, or pass --force):\n    "
            + "\n    ".join(jobs)
        )
    if problems:
        print("NOT moving:\n  " + "\n  ".join(problems))
        sys.exit(1)
    if not moves:
        print("nothing to move")
        return
    if not args.apply:
        print("dry run; pass --apply to move")
        return
    migrate(moves, new, old_root)


if __name__ == "__main__":
    main()
