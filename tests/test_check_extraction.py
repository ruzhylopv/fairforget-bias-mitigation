"""scripts/check_extraction.py check 4 on fake activations (no model)."""

import importlib.util
import sys

import numpy as np
import pandas as pd

from probing.io import run_paths, save_layer
from rmu.config import PROJECT_ROOT

sys.modules.setdefault("check_extraction", None)
_spec = importlib.util.spec_from_file_location(
    "check_extraction", PROJECT_ROOT / "scripts/check_extraction.py"
)
ce = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ce)


def fake(tmp_path, first, last, n_word_tokens, same_layer0=()):
    """Activations where rows with first == last share one vector; rows in ``same_layer0``
    share it at layer 0 only (a repeated token, as in "doctor as a doctor")."""
    paths = run_paths("m", "race", root=tmp_path)
    n = len(first)
    rng = np.random.default_rng(0)
    for k in range(3):
        a = rng.standard_normal((n, 4)).astype(np.float32)
        b = rng.standard_normal((n, 4)).astype(np.float32)
        same = np.asarray(first) == np.asarray(last)
        b[same] = a[same]
        if k == 0:
            b[list(same_layer0)] = a[list(same_layer0)]
        save_layer(a, paths, "first_diff", k, dtype="float32")
        save_layer(b, paths, "last_subtok", k, dtype="float32")
    meta = pd.DataFrame(
        {
            "idx_first_diff": first,
            "idx_last_subtok": last,
            "n_word_tokens": n_word_tokens,
            "ell": -1.0,
        }
    )
    return meta, paths


def check4(meta, paths):
    res = ce.Results()
    ce.check_files_and_positions(res, meta, paths, 3, 4, "float32")
    return res.checks["4 positions agree"]


def test_mid_word_first_difference_passes(tmp_path):
    # rows: single-token; multi-token; multi-token with the first difference mid-word
    meta, paths = fake(tmp_path, [-2, -3, -2], [-2, -2, -2], [1, 2, 2])
    c = check4(meta, paths)
    assert c["status"] == "PASS" and "1 multi-token, first difference mid-word" in c["summary"]


def test_a_repeated_token_at_layer_0_fails(tmp_path):
    meta, paths = fake(tmp_path, [-2, -5], [-2, -2], [1, 4], same_layer0=[1])
    c = check4(meta, paths)
    assert c["status"] == "FAIL" and "layer 0: two-position rows identical [1]" in c["summary"]


def test_a_single_token_word_with_two_positions_fails(tmp_path):
    meta, paths = fake(tmp_path, [-3, -2], [-2, -2], [1, 1])
    assert (
        "single-token words with idx_first_diff != idx_last_subtok"
        in check4(meta, paths)["summary"]
    )
