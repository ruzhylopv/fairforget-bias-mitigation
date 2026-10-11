import csv
from argparse import Namespace

from rmu import unlearn


def test_loss_log_writes_header_and_rows_immediately(tmp_path):
    path = tmp_path / "losses.csv"
    log = unlearn.LossLog(path)

    log.write({"step": 0, "loss": 2.5})
    # Readable before close(): the job can be watched while it runs.
    assert path.read_text().splitlines() == ["step,loss", "0,2.5"]

    log.write({"step": 1, "loss": 1.5})
    log.close()
    assert list(csv.DictReader(path.open())) == [
        {"step": "0", "loss": "2.5"},
        {"step": "1", "loss": "1.5"},
    ]


def args(**overrides):
    defaults = dict(
        output_dir=None, model_name_or_path="lapa-12b-instruct", alpha=[100.0], layer_id=7
    )
    return Namespace(**{**defaults, **overrides})


def test_output_path_has_no_brackets_in_alpha(monkeypatch, tmp_path):
    monkeypatch.setattr(unlearn, "run_dir", lambda: tmp_path)

    assert (
        unlearn.output_path(args(), 257).name == "lapa-12b-instruct_alpha-100_batches-257_layer-7"
    )
    assert unlearn.output_path(args(alpha=[100.0, 1.5]), 3).name.startswith(
        "lapa-12b-instruct_alpha-100,1.5_"
    )


def test_output_dir_overrides_default(tmp_path):
    assert unlearn.output_path(args(output_dir=str(tmp_path / "x")), 1) == tmp_path / "x"
