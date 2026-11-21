"""scripts/tokenization_check.py --merge: other models' rows stay byte for byte."""

import importlib.util

import pytest

from rmu.config import PROJECT_ROOT

_spec = importlib.util.spec_from_file_location(
    "tokenization_check", PROJECT_ROOT / "scripts/tokenization_check.py"
)
tc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tc)


@pytest.mark.parametrize("end", ["\n", "\r\n"])
def test_merge_replaces_or_appends_one_model_and_keeps_the_rest_byte_for_byte(tmp_path, end):
    path = tmp_path / "pairs.csv"
    text = end.join(["model,item_id,n", 'a,1,"x, y"', "b,2,3", "a,3,True"]) + end
    path.write_bytes(text.encode())
    assert tc.line_end(path) == end

    new = [{"model": "c", "item_id": "9", "n": False}]
    tc.write_csv(path, tc.merge_csv(path, new, ["c"]), lineterminator=end)
    assert path.read_bytes() == (text + "c,9,False" + end).encode()

    redo = [{"model": "b", "item_id": "2", "n": 4}]
    tc.write_csv(path, tc.merge_csv(path, redo, ["b"]), lineterminator=end)
    lines = path.read_bytes().decode().split(end)
    assert lines[:4] == ["model,item_id,n", 'a,1,"x, y"', "a,3,True", "c,9,False"]
    assert lines[4] == "b,2,4"


def test_merge_refuses_rows_with_other_columns(tmp_path):
    path = tmp_path / "pairs.csv"
    path.write_text("model,n\na,1\n")
    with pytest.raises(ValueError, match="columns"):
        tc.merge_csv(path, [{"model": "b", "x": 1}], ["b"])
