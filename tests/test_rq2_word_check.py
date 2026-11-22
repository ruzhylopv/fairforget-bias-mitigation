"""scripts/rq2_word_check.py on a small synthetic run."""

import importlib.util

import numpy as np
import pandas as pd

from rmu.config import PROJECT_ROOT

_spec = importlib.util.spec_from_file_location(
    "rq2_word_check", PROJECT_ROOT / "scripts/rq2_word_check.py"
)
wc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wc)


def fake_run(root):
    """6 targets x 2 pairs. Pair 0 of every target: "poor"/"rich" (recurring words), layer 0
    right. Pair 1: words unique to the target, layer 0 wrong, layer 13 right."""
    rows = []
    for t in range(6):
        for j, (ws, wa) in enumerate((("poor", "rich"), (f"bad{t}", f"good{t}"))):
            for S, w in ((1, ws), (0, wa)):
                rows.append(
                    {
                        "item_id": f"t{t}p{j}",
                        "target": f"T{t}",
                        "S": S,
                        "word": w,
                        "sentence": f"T{t} people are {w}.",
                        "target_after_word": 0,
                    }
                )
    meta = pd.DataFrame(rows)
    meta.insert(0, "row", np.arange(len(meta)))
    unique = meta["word"].str.match(r"(bad|good)\d")
    sign = np.where(meta["S"] == 1, 1.0, -1.0)
    z = {"ell": 0.1 * sign, "0": np.where(unique, -sign, sign), "13": sign}
    (root / "probes/last_subtok").mkdir(parents=True)
    meta.to_csv(root / "meta.csv", index=False)
    pd.DataFrame({"row": meta["row"], "z": z["ell"]}).to_csv(
        root / "probes/baseline_ell.csv", index=False
    )
    for k in ("0", "13"):
        pd.DataFrame({"row": meta["row"], "z": z[k]}).to_csv(
            root / f"probes/last_subtok/preds_layer_{int(k):02d}.csv", index=False
        )


def test_word_check_separates_recurring_and_unique_words(tmp_path):
    fake_run(tmp_path / "run")
    wc.main(
        [
            "--model",
            "m",
            "--bias",
            "race",
            "--layer",
            "13",
            "--data",
            str(tmp_path / "run"),
            "--out",
            str(tmp_path / "out"),
        ]
    )
    md = (tmp_path / "out/race_m.md").read_text()
    assert "| layer 0 (word identity) | 0.500 (n=12) |" in md
    assert "| layer 13 (fixed depth) | 1.000 (n=12) |" in md
    assert "Depth fixes 6 pairs and breaks 0." in md
    assert "| both seen under other targets | 6 | 1.000 | 1.000 | 1.000 |" in md
    assert "| neither: unique to this target | 6 | 0.000 | 1.000 | 1.000 |" in md
    words = pd.read_csv(tmp_path / "out/race_m_words.csv", index_col=0)
    assert words.loc["poor", "n_stereo"] == 6 and words.loc["poor", "n_targets"] == 6
