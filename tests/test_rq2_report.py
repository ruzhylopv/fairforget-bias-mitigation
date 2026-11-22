"""scripts/rq2_report.py on two small fake run folders."""

import importlib.util
import json
import sys

import numpy as np
import pandas as pd
import pytest

from probing.controls import ACC_METRICS, CI_METRICS, SUBSETS
from probing.gates import write_gate
from probing.io import POSITIONS, report_config
from rmu.config import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))  # rq2_report imports plot_probes
_spec = importlib.util.spec_from_file_location("rq2_report", PROJECT_ROOT / "scripts/rq2_report.py")
rq2_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rq2_report)


def fake_run(root, bias, model, n_layers, n_targets, peak, gate2_ok=True):
    """ci.csv / pvalues.csv / summary.csv / run.json / gates shaped like probing.controls'."""
    folder = root / bias / model
    folder.mkdir(parents=True)
    keys = [("baseline_ell", -1)] + [(p, k) for p in POSITIONS for k in range(n_layers + 1)]
    ci, pv, summary = [], [], []
    for pos, k in keys:
        base = pos == "baseline_ell"
        est = 0.6 if base else 0.55 + peak * np.exp(-(((k / n_layers) - 0.45) ** 2) / 0.05)
        s = {"position": pos, "layer": k}
        for sub in SUBSETS:
            s |= {f"acc{sub}": est, f"acc_pair{sub}": est + 0.01, f"logloss{sub}": 1 - est}
            s[f"n_pairs{sub}"] = {"": 100, "_tfirst": 80, "_tafter": 20}[sub]
        s |= {f"C_f{f}": 0.01 for f in range(5)} | {"C_edge": "", "n_conv_warnings": 0}
        summary.append(s)
        for m in CI_METRICS:
            e = s[m]
            gap = e - (0.6 if "logloss" not in m else 0.4) - (0.01 if "acc_pair" in m else 0)
            row = {
                "position": pos,
                "layer": k,
                "rel_depth": np.nan if base else k / n_layers,
                "metric": m,
                "estimate": e,
                "low": e - 0.03,
                "high": e + 0.03,
                "gap": gap,
                "gap_low": gap - 0.04,
                "gap_high": gap + 0.04,
                "share_gap_le0": 0.0,
                "gap_layer0": np.nan if base else gap + 0.05,
                "gap_layer0_low": np.nan if base else gap + 0.01,
                "gap_layer0_high": np.nan if base else gap + 0.09,
                "share_gap_layer0_le0": np.nan if base else 0.0,
                "R": 2000,
            }
            ci.append(row)
            if m in ACC_METRICS:
                pv.append(
                    {"position": pos, "layer": k, "metric": m, "real": e, "null_mean": 0.5}
                    | {"null_max": 0.55, "p": 0.0099, "p_corr": np.nan if base else 0.0099}
                    | {"B": 100}
                )
    pd.DataFrame(ci).to_csv(folder / "ci.csv", index=False)
    pd.DataFrame(pv).to_csv(folder / "pvalues.csv", index=False)
    pd.DataFrame(summary).to_csv(folder / "summary.csv", index=False)
    stats = {"n_layers": n_layers, "n_targets": n_targets, "n_pairs": 100}
    (folder / "run.json").write_text(
        json.dumps({"model": model, "bias": bias, "steps": {"controls stats": stats}})
    )
    write_gate(folder / "gate1.txt", [("checks", True, "")])
    write_gate(folder / "gate2.txt", [("null", True, ""), ("folds", gate2_ok, "")])
    return folder


@pytest.fixture
def two_runs(tmp_path):
    fake_run(tmp_path, "gender", "lapa-12b-instruct", 48, 40, peak=0.3)
    fake_run(tmp_path, "religion", "qwen3-8b", 36, 12, peak=0.1, gate2_ok=False)
    (tmp_path / "race" / "gemma3-12b-it").mkdir(parents=True)  # unfinished: no ci.csv
    return tmp_path


def test_collect_joins_every_run_with_its_run_level_columns(two_runs):
    all_runs = rq2_report.collect(two_runs)

    assert set(zip(all_runs["bias"], all_runs["model"])) == {
        ("gender", "lapa-12b-instruct"),
        ("religion", "qwen3-8b"),
    }
    n_rows = (1 + 2 * 49 + 1 + 2 * 37) * len(CI_METRICS)
    assert len(all_runs) == n_rows
    lapa = all_runs[all_runs["model"] == "lapa-12b-instruct"]
    assert (lapa["fixed_layer"] == 20).all() and (lapa["n_targets"] == 40).all()
    qwen = all_runs[all_runs["model"] == "qwen3-8b"]
    assert (qwen["fixed_layer"] == 15).all() and (qwen["gate2"] == "FAIL (1 of 2)").all()
    assert lapa["depth_chosen_here"].all() and not qwen["depth_chosen_here"].any()
    # p-values only for accuracies; n_pairs of the metric's own subset.
    assert lapa.loc[lapa["metric"] == "logloss", "p"].isna().all()
    tafter = lapa[(lapa["metric"] == "acc_pair_tafter") & (lapa["layer"] == 3)]
    assert (tafter["n_pairs_subset"] == 20).all()


def test_collect_refuses_summary_and_ci_that_disagree(two_runs):
    path = two_runs / "gender" / "lapa-12b-instruct" / "summary.csv"
    s = pd.read_csv(path)
    s.loc[5, "acc_pair"] += 0.01
    s.to_csv(path, index=False)
    with pytest.raises(ValueError, match="disagree on acc_pair"):
        rq2_report.collect(two_runs)


def test_report_table_reads_the_fixed_depth_and_flags(two_runs):
    both = rq2_report.report_table(rq2_report.collect(two_runs))
    assert both["metric"].tolist() == ["acc_pair"] * 2 + ["acc"] * 2
    # the fixture's pairwise accuracy is the per-sentence one + 0.01 everywhere
    pair, sent = (both[both["metric"] == m].set_index("model") for m in ("acc_pair", "acc"))
    assert (pair["fixed_estimate"] - sent["fixed_estimate"]).round(6).eq(0.01).all()
    table = pair
    cfg = report_config()

    lapa, qwen = table.loc["lapa-12b-instruct"], table.loc["qwen3-8b"]
    assert (lapa["fixed_layer"], qwen["fixed_layer"]) == (20, 15)
    assert lapa["fixed_rel_depth"] == pytest.approx(cfg["fixed_depth"]["rel_depth"])
    assert qwen["few_targets"] and not lapa["few_targets"]
    assert lapa["ell"] == pytest.approx(0.61) and lapa["fixed_gap_layer0"] > lapa["fixed_gap"]
    assert lapa["n_pairs_tafter"] == 20 and lapa["best_p_corr_post_hoc"] == pytest.approx(0.0099)
    assert (lapa["gate1"], lapa["gate2"]) == ("PASS", "PASS")
    tex = rq2_report.to_latex(both)
    assert r"\toprule" in tex and "lapa-12b-instruct$^\\dagger$" in tex and "12$^\\ast$" in tex
    assert tex.index("{Pairwise accuracy}") < tex.index("{Per-sentence accuracy}")
    assert tex.count("lapa-12b-instruct$^\\dagger$") == 2 and tex.count(r"\midrule") == 2


def test_main_writes_the_tables_and_both_figures(two_runs, capsys):
    rq2_report.main(["--results_root", str(two_runs)])
    for name in (
        "all_runs.csv",
        "report_table.csv",
        "report_table.tex",
        "fig_gap_by_depth.png",
        "fig_gap_by_depth.pdf",
        "fig_acc_by_depth.png",
        "fig_acc_by_depth.pdf",
        "fig_forest.png",
        "fig_forest.pdf",
    ):
        assert (two_runs / name).stat().st_size > 0, name
    assert "2 runs" in capsys.readouterr().out


def test_the_caption_states_the_null_seeds_per_bias_type():
    t = pd.DataFrame({"bias": ["gender", "profession", "race", "race"], "B": [100, 100, 20, 20.0]})
    note = rq2_report.null_note(t)
    assert "100 for gender, profession (smallest $p$ 0.010)" in note
    assert "20 for race (smallest $p$ 0.048)" in note
    assert rq2_report.null_note(t[t["bias"] == "gender"]).endswith(
        "100 seeds for gender (smallest $p$ 0.010)."
    )
