"""probing.gates: the gate file format the pipeline reads."""

from probing import gates


def test_gate_round_trip_ignores_the_free_text_after_the_checks(tmp_path):
    path = tmp_path / "gate2.txt"
    items = [("null at chance", True, "mean 0.50"), ("folds", False, "bad"), ("C grid", None, "")]
    text = gates.write_gate(path, items, ["fixed depth ...", "[x] not a check, after the blank"])

    assert text.splitlines()[0] == "[x] null at chance"
    assert gates.read_gate(path)[:3] == [
        (True, "null at chance"),
        (False, "folds"),
        (None, "C grid"),
    ]
    ok, bad = gates.gate_passed(path)
    assert not ok and bad == ["[ ] FAIL folds", "[?] C grid"]


def test_gate_passed_needs_an_existing_file_with_every_check_passed(tmp_path):
    path = tmp_path / "gate1.txt"
    assert gates.gate_passed(path) == (False, ["gate1.txt does not exist"])
    path.write_text("nothing here\n")
    assert gates.gate_passed(path) == (False, ["gate1.txt has no checks"])
    gates.write_gate(path, [("a", True, ""), ("b", True, "")])
    assert gates.gate_passed(path) == (True, [])


def test_gate1_stays_open_without_the_gpu_checks():
    checks = {"3. layer 0": {"status": "PASS", "summary": "ok"}}
    cpu = gates.gate1_items({"gpu": False, "checks": checks})
    gpu = gates.gate1_items({"gpu": True, "checks": checks | {"5. files": {"status": "FAIL"}}})
    assert [ok for _, ok, _ in cpu] == [None, True]
    assert [ok for _, ok, _ in gpu] == [True, True, False]
