"""Gate files of an RQ2 run: ``gate1.txt`` (extraction checks) and ``gate2.txt`` (controls).

One line per check, then its detail indented below::

    [x] name            passed
    [ ] FAIL name       failed
    [?] name            not run yet / not decided

Anything after the checks (a blank line, then free text such as the peak report) is ignored by
``read_gate``. ``probing.pipeline`` continues past a gate only if every check is ``[x]``.
"""

import re
from pathlib import Path

MARK = {True: "[x]", False: "[ ] FAIL", None: "[?]"}
_LINE = re.compile(r"^\[(x| |\?)\](?: FAIL)? (.+)$")


def format_gate(items) -> str:
    """``items``: [(name, ok True/False/None, detail)]."""
    return "\n".join(f"{MARK[ok]} {name}\n      {detail}" for name, ok, detail in items)


def write_gate(path, items, extra_lines=()) -> str:
    """Write the checks (and free text after a blank line) to ``path``; returns the text."""
    text = format_gate(items)
    if extra_lines:
        text += "\n\n" + "\n".join(extra_lines)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    return text


def read_gate(path) -> list:
    """[(ok True/False/None, name)] for every check line of a gate file."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = _LINE.match(line)
        if m:
            out.append(({"x": True, " ": False, "?": None}[m.group(1)], m.group(2)))
    return out


def gate_passed(path) -> tuple:
    """(True, []) if ``path`` exists and every check passed, else (False, [reasons])."""
    path = Path(path)
    if not path.exists():
        return False, [f"{path.name} does not exist"]
    checks = read_gate(path)
    if not checks:
        return False, [f"{path.name} has no checks"]
    bad = [f"{MARK[ok]} {name}" for ok, name in checks if ok is not True]
    return not bad, bad


def gate1_items(payload: dict) -> list:
    """Gate 1 items from ``check_extraction.json`` (written by scripts/check_extraction.py).

    Checks 1-2 need the GPU; without ``--gpu`` they did not run, which leaves the gate open.
    """
    gpu = bool(payload.get("gpu"))
    items = [
        (
            "checks 1-2 run (GPU: single vs saved, shared prefix)",
            True if gpu else None,
            "ran with --gpu" if gpu else "only the CPU checks 3-5 ran; rerun with --gpu",
        )
    ]
    for name, c in payload.get("checks", {}).items():
        items.append((name, c["status"] == "PASS", c.get("summary", "")))
    return items
