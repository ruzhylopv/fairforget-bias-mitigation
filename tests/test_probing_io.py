import numpy as np
import pandas as pd
import pytest

from probing import io
from probing.io import META_COLUMNS, validate_meta

MODEL = "tiny-model"


@pytest.fixture(autouse=True)
def tmp_root(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "rq2_root", lambda: tmp_path)
    return tmp_path


def make_meta(n_pairs=3):
    """A valid meta table: pairs in order, stereotype first."""
    rows = []
    for i in range(n_pairs):
        for s, word in ((1, "caring"), (0, "uncaring")):
            rows.append(
                {
                    "row": len(rows),
                    "sentence_id": f"sent{i}-{s}",
                    "item_id": f"item{i}",
                    "split": "test",
                    "bias_type": "gender",
                    "target": ["mother", "gentlemen", "schoolgirl"][i % 3],
                    "S": s,
                    # Double space and a comma: text must survive the CSV byte for byte.
                    "sentence": f'The {word} mother,  says "hi" {i}.',
                    "word": word,
                    "n_tokens": 9,
                    "idx_first_diff": -7,
                    "idx_last_subtok": -7 if s else -6,
                    "n_word_tokens": 1 if s else 2,
                    "target_after_word": True,
                    "ends_with_punct": True,
                    "ell": np.nan,
                    "model": MODEL,
                }
            )
    return pd.DataFrame(rows, columns=META_COLUMNS)


def test_paths_follow_the_layout(tmp_root):
    assert io.meta_path(MODEL) == tmp_root / "meta" / "tiny-model.csv"
    assert io.layer_path(MODEL, "first_diff", 7) == (
        tmp_root / "acts" / "tiny-model" / "first_diff" / "layer_07.npy"
    )
    assert io.layer_path(MODEL, "last_subtok", 48, root="/x").as_posix() == (
        "/x/acts/tiny-model/last_subtok/layer_48.npy"
    )


def test_layer_path_rejects_unknown_position():
    with pytest.raises(ValueError, match="unknown position"):
        io.layer_path(MODEL, "end_of_sentence", 0)


def test_meta_round_trip_keeps_text_and_types():
    df = make_meta()

    io.save_meta(df, MODEL)
    back = io.load_meta(MODEL)

    pd.testing.assert_frame_equal(back, df, check_dtype=False)
    assert back["sentence"][0] == 'The caring mother,  says "hi" 0.'
    assert back["S"].dtype == np.int64
    assert back["target_after_word"].dtype == bool
    assert back["ell"].isna().all()


def test_save_meta_refuses_other_model():
    with pytest.raises(ValueError, match="model column"):
        io.save_meta(make_meta(), "other-model")


def test_layer_round_trip_is_float16(tmp_root):
    x = np.random.default_rng(0).standard_normal((6, 4))

    path = io.save_layer(x, MODEL, "first_diff", 3)
    back = io.load_layer(MODEL, "first_diff", 3)

    assert path.exists() and not path.with_name(path.name + ".tmp").exists()
    assert isinstance(back, np.memmap)
    assert back.dtype == np.float16 and back.shape == (6, 4)
    np.testing.assert_allclose(back, x, atol=1e-2)
    assert not isinstance(io.load_layer(MODEL, "first_diff", 3, mmap=False), np.memmap)


def test_save_layer_checks_shape_and_overflow():
    with pytest.raises(ValueError, match="2-D"):
        io.save_layer(np.zeros(5), MODEL, "first_diff", 0)
    with pytest.raises(ValueError, match="overflow float16"):
        io.save_layer(np.full((2, 2), 1e5), MODEL, "first_diff", 0)


def test_valid_meta_passes():
    validate_meta(make_meta())


def test_validate_catches_wrong_pair_order():
    df = make_meta()
    df.iloc[[0, 1]] = df.iloc[[1, 0]].to_numpy()  # anti-stereotype first in pair 0
    df["row"] = range(len(df))

    with pytest.raises(ValueError, match="S=1 then S=0"):
        validate_meta(df)


def test_validate_catches_split_pairs():
    df = make_meta()
    df.iloc[[1, 2]] = df.iloc[[2, 1]].to_numpy()  # rows 0 and 1 now from different items
    df["row"] = range(len(df))

    with pytest.raises(ValueError, match="share item_id"):
        validate_meta(df)


def test_validate_catches_row_column_out_of_order():
    df = make_meta()
    df.loc[0, "row"], df.loc[1, "row"] = 1, 0

    with pytest.raises(ValueError, match="row must be"):
        validate_meta(df)


def test_validate_catches_unbalanced_labels():
    df = make_meta()
    df.loc[3, "S"] = 1

    with pytest.raises(ValueError, match="unbalanced"):
        validate_meta(df)


def test_validate_catches_positive_index():
    df = make_meta()
    df.loc[4, "idx_last_subtok"] = 2

    with pytest.raises(ValueError, match="idx_last_subtok must be negative"):
        validate_meta(df)


def test_validate_catches_index_on_bos():
    df = make_meta()
    df.loc[2, "idx_first_diff"] = -9  # n_tokens = 9, so -9 is BOS

    with pytest.raises(ValueError, match="idx_first_diff points at BOS"):
        validate_meta(df)


def test_validate_catches_last_subtok_before_first_diff():
    df = make_meta()
    df.loc[0, "idx_last_subtok"] = -8

    with pytest.raises(ValueError, match="idx_last_subtok must be >= idx_first_diff"):
        validate_meta(df)


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda df: df.drop(columns="ell"), "columns must be exactly"),
        (lambda df: df[list(reversed(META_COLUMNS))], "columns must be exactly"),
        (lambda df: df.assign(target=df["target"].where(df["row"] != 2)), "NaN in columns"),
        (lambda df: df.assign(sentence_id="same"), "sentence_id must be unique"),
        (lambda df: df.iloc[:0], "empty"),
    ],
)
def test_validate_catches_schema_problems(change, message):
    with pytest.raises(ValueError, match=message):
        validate_meta(change(make_meta()))
