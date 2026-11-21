import json
import sys

import numpy as np
import pandas as pd
import pytest

from probing import io, meta
from probing.io import META_COLUMNS
from rmu.config import repo_path

MODEL = "lapa-12b-instruct"


@pytest.fixture
def pairs(tmp_path):
    """The first 5 lapa gender pairs of the real tokenization check, as their own pairs.csv."""
    real = pd.read_csv(repo_path(meta.PAIRS_CSV), keep_default_na=False, dtype=str)
    five = real[(real["model"] == MODEL) & (real["bias_type"] == "gender")].head(5)
    path = tmp_path / "pairs.csv"
    five.to_csv(path, index=False)
    return five.reset_index(drop=True), path


def test_build_meta_interleaves_pairs_stereotype_first(pairs):
    five, path = pairs

    df = meta.build_meta(MODEL, "gender", pairs_csv=path)

    assert list(df.columns) == META_COLUMNS
    assert len(df) == 10
    assert df["row"].tolist() == list(range(10))
    assert df["S"].tolist() == [1, 0] * 5
    assert df["item_id"].tolist() == [i for i in five["item_id"] for _ in (0, 1)]
    assert df["sentence_id"][0::2].tolist() == five["sentence_id_stereo"].tolist()
    assert df["sentence_id"][1::2].tolist() == five["sentence_id_anti"].tolist()
    assert df["sentence"][0::2].tolist() == five["sentence_stereo"].tolist()
    assert df["word"][1::2].tolist() == five["word_anti"].tolist()
    assert (df["model"] == MODEL).all() and (df["bias_type"] == "gender").all()


def test_build_meta_drops_excluded_items_as_whole_pairs(pairs):
    five, path = pairs
    gone = five["item_id"][1]

    df = meta.build_meta(MODEL, "gender", pairs_csv=path, exclude={gone: "test"})

    assert len(df) == 8 and gone not in set(df["item_id"])
    assert list(df["row"]) == list(range(8))


@pytest.mark.parametrize("bias", ["gender", "profession", "race"])
def test_real_exclusions_are_items_of_their_bias_type(bias):
    real = pd.read_csv(repo_path(meta.PAIRS_CSV), keep_default_na=False, dtype=str)
    ids = set(real[(real["model"] == MODEL) & (real["bias_type"] == bias)]["item_id"])
    exclude = io.bias_config(bias)["exclude_items"]
    assert exclude and set(exclude) <= ids


@pytest.mark.parametrize("bias", ["profession", "race"])
def test_meta_builds_for_every_model_with_the_reviewed_config(bias):
    real = pd.read_csv(repo_path(meta.PAIRS_CSV), usecols=["model", "bias_type"], dtype=str)
    for model in sorted(set(real.loc[real["bias_type"] == bias, "model"])):
        df = meta.build_meta(model, bias)
        assert not set(io.bias_config(bias)["exclude_items"]) & set(df["item_id"])


def test_default_exclusions_come_from_the_bias_config(pairs, monkeypatch):
    five, path = pairs
    gone = five["item_id"][2]
    monkeypatch.setattr(
        meta, "bias_config", lambda bias: {"exclude_items": {gone: "why"}, "misspelled": []}
    )
    assert gone not in set(meta.build_meta(MODEL, "gender", pairs_csv=path)["item_id"])


def test_build_meta_reproduces_the_committed_lapa_gender_labels():
    """The meta rebuilt with config/rq2/gender.yaml is the committed label table (no ell)."""
    committed = io.read_meta(repo_path(f"{io.RESULTS_ROOT}/gender/{MODEL}/labels.csv"))
    pd.testing.assert_frame_equal(meta.build_meta(MODEL, "gender"), committed)


def test_build_meta_index_formula(pairs):
    five, path = pairs
    df = meta.build_meta(MODEL, "gender", pairs_csv=path)

    for side, rows in (("stereo", df.iloc[0::2]), ("anti", df.iloc[1::2])):
        n = five[f"len_{side}"].astype(int).to_numpy()
        first = five[f"first_diff_{side}"].astype(int).to_numpy()
        assert rows["n_tokens"].tolist() == n.tolist()
        assert rows["idx_first_diff"].tolist() == (first - n).tolist()
        assert rows["idx_last_subtok"].tolist() == (
            five[f"last_subtok_from_end_{side}"].astype(int).tolist()
        )
        assert rows["n_word_tokens"].tolist() == five[f"n_tok_{side}"].astype(int).tolist()


def test_build_meta_new_columns(pairs):
    _, path = pairs
    df = meta.build_meta(MODEL, "gender", pairs_csv=path)

    for r in df.itertuples():
        start = r.sentence.index(r.word)
        assert r.target_after_word == (r.sentence.lower().find(r.target.lower()) > start)
        assert r.ends_with_punct == r.sentence.strip().endswith((".", "!", "?"))
    assert df["ell"].isna().all()


def test_build_meta_unknown_model_raises(pairs):
    _, path = pairs
    with pytest.raises(ValueError, match="no pairs"):
        meta.build_meta("no-such-model", "gender", pairs_csv=path)


def test_word_start_picks_the_occurrence_that_differs():
    # "mother" appears twice; the attribute word is the one where the pair differs.
    s = "The mother said the mother was caring."
    other = "The mother said the father was caring."
    assert meta.word_start(s, "mother", other) == s.index("mother", 5)
    assert meta.word_start("The uncaring mom", "uncaring", "The unkind mom") == 4


def test_target_inside_the_word_is_not_after_it():
    s = "They called her his bridezilla."
    assert not meta.target_after_word(s, "bride", s.index("bridezilla"))
    assert meta.target_after_word("A caring Mother.", "mother", 2)


def test_cli_writes_validated_meta_and_records_the_step(pairs, tmp_path, monkeypatch, capsys):
    _, path = pairs
    root = tmp_path / "rq2"
    argv = ["meta", "--model", MODEL, "--bias", "gender", "--pairs_csv", str(path)]
    monkeypatch.setattr(sys, "argv", [*argv, "--root", str(root)])

    meta.main()

    paths = io.run_paths(MODEL, "gender", root=root)
    written = io.load_meta(paths)
    assert paths.meta == root / "gender" / MODEL / "meta.csv" and paths.meta.exists()
    assert len(written) == 10 and np.isnan(written["ell"]).all()
    assert "rows: 10" in capsys.readouterr().out
    run = json.loads(paths.run_json.read_text())
    assert run["bias"] == "gender" and run["steps"]["meta"]["n_rows"] == 10
    assert run["steps"]["meta"]["excluded_items"] == io.bias_config("gender")["exclude_items"]


def test_pairs_without_context_are_dropped_for_tokenizers_without_bos(pairs, tmp_path):
    """No BOS and the attribute word first: first_diff 0, no context and no ell -> dropped."""
    five, _ = pairs
    no_bos = five.copy()
    gone = no_bos["item_id"][3]
    no_bos.loc[3, "first_diff_anti"] = "0"
    path = tmp_path / "no_bos.csv"
    no_bos.to_csv(path, index=False)

    df = meta.build_meta(MODEL, "gender", pairs_csv=path, exclude={})

    assert len(df) == 8 and gone not in set(df["item_id"])
    assert df.attrs["dropped_no_context"] == [gone]
    assert (
        meta.build_meta(MODEL, "gender", pairs_csv=path.parent / "pairs.csv").attrs[
            "dropped_no_context"
        ]
        == []
    )
    assert "no BOS: 1" in meta.describe(df)


def test_pairs_whose_first_difference_lies_outside_the_word_are_dropped(pairs, tmp_path):
    """qwen3 "Romanians live in rome": the space before "rome" sits in the previous token."""
    five, _ = pairs
    odd = five.copy()
    gone = odd["item_id"][1]
    odd.loc[1, "first_diff_class_anti"] = "before_word"
    keep = odd["item_id"][2]
    odd.loc[2, "first_diff_class_stereo"] = "mid_word"  # inside the word: kept
    path = tmp_path / "odd.csv"
    odd.to_csv(path, index=False)

    df = meta.build_meta(MODEL, "gender", pairs_csv=path, exclude={})

    assert gone not in set(df["item_id"]) and keep in set(df["item_id"])
    assert df.attrs["dropped_diff_outside_word"] == [gone]
