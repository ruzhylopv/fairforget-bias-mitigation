"""probing.pipeline on fake data (scripts/make_fake_acts.py): skipping, resuming, gates."""

import json
import subprocess
import sys

import pandas as pd
import pytest

from probing import io, pipeline
from probing.gates import gate_passed
from rmu.config import PROJECT_ROOT

MODEL = "lapa-12b-instruct"


def fake_run(root, layers="0-2"):
    """Fake activations + meta for a 3-hidden-state model, and a passing gate-1 json."""
    subprocess.run(
        [sys.executable, "scripts/make_fake_acts.py", "--root", str(root), "--layers", layers]
        + ["--signal", layers, "--hidden", "8", "--delta", "1.5", "--ell_delta", "0.5"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    )
    paths = io.run_paths(MODEL, "gender", root=root)
    checks = {f"{i}. check": {"status": "PASS", "summary": "fake"} for i in range(1, 6)}
    (paths.acts / "check_extraction.json").write_text(json.dumps({"gpu": True, "checks": checks}))
    return pipeline.Run(paths, str(root), 2, quick_seeds=2, null_seeds=3)


def test_check_inputs_stops_for_models_without_pairs_and_bias_without_config(tmp_path):
    with pytest.raises(pipeline.Stop, match=r"not in .*pairs.csv.*card \[7\]"):
        pipeline.check_inputs(
            "llama31-8b-instruct", "gender", io.run_paths("x", "gender", tmp_path)
        )
    with pytest.raises(pipeline.Stop, match=r"race.yaml is missing.*card \[6\]"):
        pipeline.check_inputs("qwen3-8b", "race", io.run_paths("qwen3-8b", "race", tmp_path))
    pipeline.check_inputs("qwen3-8b", "gender", io.run_paths("qwen3-8b", "gender", tmp_path))


def test_dry_run_on_an_empty_root_would_run_everything_and_writes_nothing(tmp_path):
    run = pipeline.Run(io.run_paths("qwen3-8b", "gender", tmp_path), str(tmp_path), None, 5, 100)
    out = pipeline.run_stage(run, "quick", dry_run=True, log=lambda s: None)
    assert list(out) == ["meta", "extract", "gate 1", "probe", "null (quick)", "cgrid", "stats"]
    assert all(v.startswith("would run") for v in out.values())
    assert not any(tmp_path.iterdir())
    with pytest.raises(pipeline.Stop, match="needs gate2.txt all PASS"):
        pipeline.run_stage(run, "null", dry_run=True, log=lambda s: None)


def test_pipeline_end_to_end_skips_what_is_done_and_resumes(tmp_path):
    run = fake_run(tmp_path)
    p = run.paths
    quiet = lambda s: None  # noqa: E731

    first = pipeline.run_stage(run, "quick", log=quiet)
    assert first["meta"].startswith("skipped") and first["extract"].startswith("skipped")
    assert "gate1.txt written" in first["gate 1"] and gate_passed(p.results / "gate1.txt")[0]
    assert all(first[s].startswith("ran") for s in ("probe", "null (quick)", "stats"))
    assert gate_passed(p.results / "gate2.txt") == (True, [])
    summary = pd.read_csv(p.summary)
    assert sorted(summary["layer"].unique()) == [-1, 0, 1, 2]  # layers from acts/run.json
    ci = (p.results / "ci.csv").read_bytes()

    again = pipeline.run_stage(run, "quick", log=quiet)
    assert [k for k, v in again.items() if not v.startswith("skipped")] == ["stats"]
    assert (p.results / "ci.csv").read_bytes() == ci  # same numbers

    # Long null: resumes the quick seeds, adds the rest; then nothing left to do.
    null = pipeline.run_stage(run, "null", log=quiet)
    assert null["null"].startswith("ran")
    scores = pd.read_csv(p.results / "null_scores.csv")
    assert sorted(scores["seed"].unique()) == [0, 1, 2]
    assert not scores.duplicated(["seed", "position", "layer"]).any()
    assert pipeline.run_stage(run, "null", log=quiet)["null"].startswith("skipped")

    # A lost layer file after the probes were fitted: stop instead of mixing old and new.
    p.layer("first_diff", 1).unlink()
    with pytest.raises(pipeline.Stop, match="extract has to run.*would be stale"):
        pipeline.run_stage(run, "quick", log=quiet)


def test_failed_gates_are_not_skipped_and_block_the_null(tmp_path):
    run = fake_run(tmp_path)
    checks = {"5. files": {"status": "FAIL", "summary": "fake failure"}}
    (run.paths.acts / "check_extraction.json").write_text(
        json.dumps({"gpu": True, "checks": checks})
    )
    gate1 = next(s for s in pipeline.steps("quick") if s.name == "gate 1")
    assert gate1.gate == "gate1.txt" and not gate1.status(run)[0]  # reruns the checks

    run.paths.results.mkdir(parents=True)
    (run.paths.results / "gate2.txt").write_text("[x] a\n      ok\n[ ] FAIL b\n      bad\n")
    with pytest.raises(pipeline.Stop, match=r"needs gate2.txt all PASS.*FAIL b"):
        pipeline.run_stage(run, "null", log=lambda s: None)
