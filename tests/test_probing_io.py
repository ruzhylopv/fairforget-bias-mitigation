import datetime
import json

import numpy as np
import pandas as pd
import pytest

from probing import io
from probing.io import META_COLUMNS, validate_meta
from rmu.config import PROJECT_ROOT

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


@pytest.fixture
def paths(tmp_root):
    return io.run_paths(MODEL, "gender")


# --------------------------------------------------------------------------- run_paths


def test_run_paths_layout(tmp_root):
    p = io.run_paths(MODEL, "gender")
    run = tmp_root / "gender" / "tiny-model"
    results = PROJECT_ROOT / "results" / "rq2" / "gender" / "tiny-model"

    assert (p.model, p.bias, p.data, p.results) == (MODEL, "gender", run, results)
    assert p.meta == run / "meta.csv"
    assert p.acts == run / "acts" and p.probes == run / "probes"
    assert p.layer("first_diff", 7) == run / "acts" / "first_diff" / "layer_07.npy"
    assert p.layer("last_subtok", 48).name == "layer_48.npy"
    assert p.summary == results / "summary.csv" and p.run_json == results / "run.json"


def test_run_paths_keys_by_bias_and_model(tmp_root):
    """Race on lapa must not touch gender on lapa (nor gender on another model)."""
    runs = [io.run_paths(m, b) for m in ("lapa", "qwen") for b in ("gender", "race")]
    for attr in ("data", "results", "meta", "acts", "probes", "summary", "run_json"):
        assert len({getattr(r, attr) for r in runs}) == 4, attr


def test_run_paths_with_root_keeps_results_out_of_the_repo(tmp_path):
    p = io.run_paths(MODEL, "race", root=tmp_path / "fake")
    assert p.data == tmp_path / "fake" / "race" / "tiny-model"
    assert p.results == tmp_path / "fake" / "results" / "race" / "tiny-model"

    q = io.run_paths(MODEL, "race", root=tmp_path / "fake", results_root=tmp_path / "res")
    assert q.results == tmp_path / "res" / "race" / "tiny-model"


@pytest.mark.parametrize("model", ["", "org/model", "../up", ".hidden", "a\\b"])
def test_run_paths_rejects_paths_as_model(model):
    with pytest.raises(ValueError, match="short name"):
        io.run_paths(model, "gender")


def test_run_paths_rejects_unknown_bias():
    with pytest.raises(ValueError, match="unknown bias type"):
        io.run_paths(MODEL, "genders")


def test_layer_path_rejects_unknown_position_and_negative_layer(paths):
    with pytest.raises(ValueError, match="unknown position"):
        paths.layer("end_of_sentence", 0)
    with pytest.raises(ValueError, match="non-negative"):
        paths.layer("first_diff", -1)


# --------------------------------------------------------------------------- bias config


def test_gender_config_holds_the_exclusion_and_the_misspellings():
    cfg = io.bias_config("gender")
    assert list(cfg["exclude_items"]) == ["bdccea984366464e4a58c7a257a39abf"]
    assert "testoserone" in cfg["exclude_items"]["bdccea984366464e4a58c7a257a39abf"]
    assert {"mainicure", "uncertanties", "forebearance", "igorant"} <= set(cfg["misspelled"])
    assert len(cfg["misspelled"]) == 17 and "testoserone" not in cfg["misspelled"]


def test_profession_and_race_null_covers_last_subtok_only_gender_both():
    assert io.null_positions("gender") == list(io.POSITIONS)
    assert io.null_positions("profession") == io.null_positions("race") == ["last_subtok"]


def test_bias_config_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "repo_path", lambda rel: tmp_path / rel)
    with pytest.raises(FileNotFoundError, match="race.yaml is missing"):
        io.bias_config("race")
    (tmp_path / io.BIAS_CONFIG_DIR).mkdir(parents=True)
    cfg = tmp_path / io.BIAS_CONFIG_DIR / "race.yaml"

    cfg.write_text("exclude_items:\nmisspelled: []\n")
    assert io.bias_config("race") == {
        "exclude_items": {},
        "misspelled": [],
        "null_seeds": None,
        "null_positions": None,
    }
    assert io.null_positions("race") == list(io.POSITIONS)
    cfg.write_text("null_positions: [last_token]\n")
    with pytest.raises(ValueError, match="null_positions"):
        io.bias_config("race")
    cfg.write_text("null_seeds: 0\n")
    with pytest.raises(ValueError, match="null_seeds"):
        io.bias_config("race")
    cfg.write_text("exclude_items:\n  abc:\n")  # no reason
    with pytest.raises(ValueError, match="item_id -> reason"):
        io.bias_config("race")
    cfg.write_text("misspeled: [x]\n")
    with pytest.raises(ValueError, match="unknown keys"):
        io.bias_config("race")


# --------------------------------------------------------------------------- depth


def test_n_decoder_layers_from_the_extraction_run_json(paths):
    paths.acts.mkdir(parents=True)
    (paths.acts / "run.json").write_text(json.dumps({"n_hidden_states": 49}))
    assert io.n_decoder_layers(paths) == 48


def test_n_decoder_layers_from_the_model_config(paths, tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    monkeypatch.setattr(io, "resolve_model", lambda name: str(model_dir))
    with pytest.raises(FileNotFoundError, match="unknown"):
        io.n_decoder_layers(paths)
    (model_dir / "config.json").write_text(json.dumps({"num_hidden_layers": 36}))
    assert io.n_decoder_layers(paths) == 36
    gemma3 = {"text_config": {"num_hidden_layers": 48}, "vision_config": {}}
    (model_dir / "config.json").write_text(json.dumps(gemma3))
    assert io.n_decoder_layers(paths) == 48


def test_rel_depth_is_layer_over_n_layers_and_nan_for_the_baseline():
    np.testing.assert_allclose(io.rel_depth([0, 24, 48, -1], 48), [0, 0.5, 1, np.nan])
    assert io.rel_depth(18, 36) == 0.5


def test_fixed_depth_is_layer_20_of_48_and_maps_to_the_nearest_layer():
    cfg = io.report_config()
    fd = cfg["fixed_depth"]
    assert (fd["layer"], fd["n_layers"], fd["position"]) == (20, 48, "last_subtok")
    assert fd["rel_depth"] == pytest.approx(0.4167, abs=1e-4) and fd["date"] == "2026-11-21"
    assert fd["chosen_on"] == {"model": "lapa-12b-instruct", "bias": "gender"}
    # lapa / gemma3, qwen3, Llama-3.1: 20.0, 15.0, 13.33
    assert [io.fixed_layer(n, cfg) for n in (48, 36, 32)] == [20, 15, 13]
    half = {"fixed_depth": {"layer": 1, "n_layers": 4}}  # 1/4 of 2 layers = 0.5 -> 1
    assert io.fixed_layer(2, half) == 1
    assert io.chosen_on(io.run_paths("lapa-12b-instruct", "gender"), cfg)
    assert not io.chosen_on(io.run_paths("lapa-12b-instruct", "race"), cfg)


def test_report_config_checks_fixed_depth(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "repo_path", lambda rel: tmp_path / "report.yaml")
    (tmp_path / "report.yaml").write_text("fixed_depth:\n  layer: 20\n")
    with pytest.raises(ValueError, match="needs"):
        io.report_config()


# --------------------------------------------------------------------------- run.json


def test_record_run_keeps_every_step(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    p = io.run_paths(MODEL, "gender", root=tmp_path)

    io.record_run(p, "probe", {"grid": [0.1, 1.0]})
    io.record_run(p, "controls null", {"seeds": [0, 1]})
    io.record_run(p, "probe", {"grid": [1.0]})  # a rerun replaces its own step only
    io.record_run(p, "extract", {"date": "2026-11-17T10:00:00"}, stamp=False)

    run = json.loads(p.run_json.read_text())
    assert (run["model"], run["bias"]) == (MODEL, "gender")
    assert list(run["steps"]) == ["probe", "controls null", "extract"]
    probe = run["steps"]["probe"]
    assert probe["grid"] == [1.0] and probe["slurm_job_id"] == "123"
    assert datetime.datetime.fromisoformat(probe["date"])
    assert probe["git_commit"] == io.git_commit()
    assert run["steps"]["extract"] == {"date": "2026-11-17T10:00:00"}


def test_record_run_refuses_another_runs_file(tmp_path):
    p = io.run_paths(MODEL, "gender", root=tmp_path)
    p.run_json.parent.mkdir(parents=True)
    p.run_json.write_text(json.dumps({"model": MODEL, "bias": "race", "steps": {}}))
    with pytest.raises(ValueError, match="belongs to"):
        io.record_run(p, "probe", {})


def test_git_commit_reads_dot_git_without_the_git_binary(tmp_path, monkeypatch):
    sha = "a" * 40
    (tmp_path / ".git" / "refs" / "heads").mkdir(parents=True)
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (tmp_path / ".git" / "refs" / "heads" / "main").write_text(sha + "\n")
    monkeypatch.setenv("PATH", "")  # no git on compute nodes
    assert io.git_commit(tmp_path) == sha

    (tmp_path / ".git" / "refs" / "heads" / "main").unlink()
    (tmp_path / ".git" / "packed-refs").write_text(f"# pack-refs\n{'b' * 40} refs/heads/main\n")
    assert io.git_commit(tmp_path) == "b" * 40


# --------------------------------------------------------------------------- meta + layers


def test_meta_round_trip_keeps_text_and_types(paths):
    df = make_meta()

    io.save_meta(df, paths)
    back = io.load_meta(paths)

    assert paths.meta.exists()
    pd.testing.assert_frame_equal(back, df, check_dtype=False)
    assert back["sentence"][0] == 'The caring mother,  says "hi" 0.'
    assert back["S"].dtype == np.int64
    assert back["target_after_word"].dtype == bool
    assert back["ell"].isna().all()


def test_meta_must_belong_to_the_run(paths):
    with pytest.raises(ValueError, match="has model"):
        io.save_meta(make_meta(), io.run_paths("other-model", "gender"))
    with pytest.raises(ValueError, match="has bias_type"):
        io.save_meta(make_meta(), io.run_paths(MODEL, "race"))

    io.save_meta(make_meta(), paths)
    io.write_meta(make_meta().assign(bias_type="race"), paths.meta)  # wrong file in place
    with pytest.raises(ValueError, match="has bias_type"):
        io.load_meta(paths)


def test_layer_round_trip_is_float16(paths):
    x = np.random.default_rng(0).standard_normal((6, 4))

    path = io.save_layer(x, paths, "first_diff", 3)
    back = io.load_layer(paths, "first_diff", 3)

    assert path == paths.layer("first_diff", 3)
    assert path.exists() and not path.with_name(path.name + ".tmp").exists()
    assert isinstance(back, np.memmap)
    assert back.dtype == np.float16 and back.shape == (6, 4)
    np.testing.assert_allclose(back, x, atol=1e-2)
    assert not isinstance(io.load_layer(paths, "first_diff", 3, mmap=False), np.memmap)


def test_save_layer_checks_shape_and_overflow(paths):
    with pytest.raises(ValueError, match="2-D"):
        io.save_layer(np.zeros(5), paths, "first_diff", 0)
    with pytest.raises(ValueError, match="overflow float16"):
        io.save_layer(np.full((2, 2), 1e5), paths, "first_diff", 0)
    with pytest.raises(ValueError, match="float16 or float32"):
        io.save_layer(np.zeros((2, 2)), paths, "first_diff", 0, dtype="int8")


def test_float32_storage_keeps_values_beyond_float16(paths):
    x = np.array([[1e5, -3.0], [0.5, 7e4]])

    io.save_layer(x, paths, "last_subtok", 0, dtype="float32")
    back = io.load_layer(paths, "last_subtok", 0)

    assert back.dtype == np.float32
    np.testing.assert_array_equal(back, x.astype(np.float32))


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
