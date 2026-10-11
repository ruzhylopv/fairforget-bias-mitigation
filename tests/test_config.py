from pathlib import Path

import pytest

from rmu import config


@pytest.fixture
def cfg():
    return {
        "paths": {"models_root": "/shared/models", "runs_root": "/scratch/runs"},
        "models": {"lapa-12b-instruct": "lapa-v0.1.2-instruct"},
    }


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_default_config_lists_models_and_paths():
    loaded = config.load_config()

    assert loaded["models"]["lapa-12b-instruct"] == "lapa-v0.1.2-instruct"
    assert {"models_root", "runs_root", "results_root", "usage_log"} <= loaded["paths"].keys()
    for path in loaded["data"]["stereoset_files"]:
        assert config.repo_path(path).exists()


def test_local_yaml_overrides_only_listed_keys(tmp_path):
    write(tmp_path / "default.yaml", "seed: 42\npaths:\n  models_root: /a\n  runs_root: /b\n")
    write(tmp_path / "local.yaml", "paths:\n  runs_root: runs\n")

    loaded = config.load_config.__wrapped__(tmp_path)

    assert loaded == {"seed": 42, "paths": {"models_root": "/a", "runs_root": "runs"}}


def test_missing_local_yaml_uses_defaults(tmp_path):
    write(tmp_path / "default.yaml", "seed: 1\n")

    assert config.load_config.__wrapped__(tmp_path) == {"seed": 1}


def test_short_name_resolves_to_shared_models_folder(cfg):
    assert config.resolve_model("lapa-12b-instruct", cfg) == "/shared/models/lapa-v0.1.2-instruct"


@pytest.mark.parametrize("name", ["HuggingFaceH4/zephyr-7b-beta", "/some/local/model"])
def test_unknown_names_are_passed_through(cfg, name):
    assert config.resolve_model(name, cfg) == name


def test_relative_paths_resolve_from_repo_root():
    assert config.repo_path("data/x.json") == config.PROJECT_ROOT / "data/x.json"
    assert config.repo_path("/abs/x.json") == Path("/abs/x.json")


def test_run_dir_uses_slurm_job_id(cfg, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123456")

    assert config.run_dir(cfg) == Path("/scratch/runs/123456")


def test_run_dir_outside_slurm_is_timestamped(cfg, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    run = config.run_dir(cfg)

    assert run.parent == Path("/scratch/runs")
    assert run.name.startswith("local-")
