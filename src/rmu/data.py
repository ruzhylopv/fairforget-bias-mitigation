"""Forget / retain corpora for RMU.

A corpus is given as a spec string (``--forget_corpora`` / ``--retain_corpora``):

* ``stereoset:<label>[:<bias_types>]`` - StereoSet sentences with gold label ``stereotype``,
  ``anti-stereotype`` or ``unrelated``, from items of the comma-separated bias types
  (``gender`` if omitted, ``all`` for every type), e.g. ``stereoset:anti-stereotype:race``.
* anything else is a path to a sentence file (relative to the working directory or the repo
  root):

  - ``.txt``: one sentence per line (blank lines skipped),
  - ``.jsonl``: one string or object with ``sentence`` / ``text`` per line,
  - ``.json``: a list of such strings / objects,
  - ``.csv`` / ``.tsv``: a ``sentence`` or ``text`` column.

  Objects may carry an ``id``; otherwise ids are ``<file stem>-<index>``. File sentences keep
  their order (no shuffling).

Defaults are the original setup: forget = gender stereotypes, retain = gender anti-stereotypes.
"""

import csv
import json
import random
from pathlib import Path

from rmu.config import load_config, repo_path

random.seed(42)
STEREO_PATHS = [repo_path(path) for path in load_config()["data"]["stereoset_files"]]

STEREOSET_LABELS = ("stereotype", "anti-stereotype", "unrelated")
DEFAULT_FORGET = "stereoset:stereotype:gender"
DEFAULT_RETAIN = "stereoset:anti-stereotype:gender"
TEXT_KEYS = ("sentence", "text")


def get_json_dict(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- specs


def parse_corpus(spec):
    """Spec string -> ``{"kind": "stereoset", "label", "bias_types"}`` or ``{"kind": "file"}``.

    ``bias_types`` is a frozenset, or None for all bias types.
    """
    if spec.startswith("stereoset:"):
        parts = spec.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(
                f"bad StereoSet corpus {spec!r}: use stereoset:<label>[:<bias_types>]"
            )
        label = parts[1]
        if label not in STEREOSET_LABELS:
            raise ValueError(
                f"bad StereoSet label {label!r} in {spec!r}: one of {STEREOSET_LABELS}"
            )
        types = parts[2] if len(parts) == 3 else "gender"
        bias_types = None if types == "all" else frozenset(t for t in types.split(",") if t)
        if bias_types is not None and not bias_types:
            raise ValueError(f"no bias types in {spec!r}")
        return {"kind": "stereoset", "label": label, "bias_types": bias_types}

    path = Path(spec)
    if not path.exists() and repo_path(spec).exists():
        path = repo_path(spec)
    return {"kind": "file", "path": path}


def is_paired(forget, retain):
    """True if forget sentence i and retain sentence i come from the same StereoSet item.

    That holds without --split when both are StereoSet corpora over the same bias types with
    different labels (every intrasentence item has all three labels). --loss_positions diff
    relies on it.
    """
    f, r = parse_corpus(forget), parse_corpus(retain)
    return (
        f["kind"] == r["kind"] == "stereoset"
        and f["bias_types"] == r["bias_types"]
        and f["label"] != r["label"]
    )


def corpus_name(spec):
    """Short, filesystem-safe name of a corpus, for output folder names."""
    corpus = parse_corpus(spec)
    if corpus["kind"] == "file":
        name = corpus["path"].stem
    else:
        types = "all" if corpus["bias_types"] is None else "+".join(sorted(corpus["bias_types"]))
        name = f"{corpus['label']}-{types}"
    return "".join(c if c.isalnum() or c in "-+." else "-" for c in name)


# --------------------------------------------------------------------------- loading


def _matches(item, bias_types):
    return bias_types is None or item["bias_type"] in bias_types


def _stereoset_sentences(items, label):
    return [s for item in items for s in item["sentences"] if s["gold_label"] == label]


def _record(entry, default_id):
    if isinstance(entry, str):
        return {"id": default_id, "sentence": entry}
    if isinstance(entry, dict):
        for key in TEXT_KEYS:
            if isinstance(entry.get(key), str):
                return {**entry, "id": str(entry.get("id", default_id)), "sentence": entry[key]}
    raise ValueError(f"expected a string or an object with one of {TEXT_KEYS}, got {entry!r}")


def load_sentence_file(path):
    """Sentence dicts (``id``, ``sentence``, ...) from a .txt/.jsonl/.json/.csv/.tsv file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"corpus file not found: {path}")
    suffix = path.suffix.lower()
    with open(path, encoding="utf-8", newline="") as f:
        if suffix == ".txt":
            entries = [line.strip() for line in f if line.strip()]
        elif suffix == ".jsonl":
            entries = [json.loads(line) for line in f if line.strip()]
        elif suffix == ".json":
            entries = json.load(f)
            if not isinstance(entries, list):
                raise ValueError(f"{path}: expected a JSON list of sentences")
        elif suffix in (".csv", ".tsv"):
            entries = list(csv.DictReader(f, delimiter="\t" if suffix == ".tsv" else ","))
        else:
            raise ValueError(f"{path}: unsupported corpus format {suffix!r}")
    records = [_record(entry, f"{path.stem}-{i}") for i, entry in enumerate(entries)]
    records = [r for r in records if r["sentence"].strip()]
    if not records:
        raise ValueError(f"{path}: no sentences")
    return records


def load_corpora(
    forget=DEFAULT_FORGET,
    retain=DEFAULT_RETAIN,
    batch_size: int = 4,
    split: bool = True,
    cat_type: str = "intrasentence",
    paths=STEREO_PATHS,
):
    """Forget and retain sentences for RMU, in batches.

    StereoSet items are shuffled once (module-level seed). With ``split``, an item that both
    corpora could draw from goes to one side only: the first half of those items to forget,
    the second half to retain. Items only one corpus can use, and file corpora, are used in
    full. Without ``split`` both StereoSet corpora use all their items.

    Returns ``(forget_batches, retain_batches)``: lists of batches of sentence dicts, each with
    at least ``id`` and ``sentence``.
    """
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    specs = [parse_corpus(forget), parse_corpus(retain)]
    stereo = [c for c in specs if c["kind"] == "stereoset"]

    if stereo:
        items = []
        for path in paths:
            items.extend(get_json_dict(path)["data"][cat_type])
        pool = [item for item in items if any(_matches(item, c["bias_types"]) for c in stereo)]
        random.shuffle(pool)

        if len(stereo) == 2 and split:
            shared = [
                item for item in pool if all(_matches(item, c["bias_types"]) for c in stereo)
            ]
            half = len(shared) // 2
            excluded = [
                {id(item) for item in shared[half:]},  # forget side: not the retain half
                {id(item) for item in shared[:half]},  # retain side: not the forget half
            ]
        else:
            excluded = [set(), set()]

    data = []
    for side, corpus in enumerate(specs):
        if corpus["kind"] == "file":
            data.append(load_sentence_file(corpus["path"]))
            continue
        items = [
            item
            for item in pool
            if _matches(item, corpus["bias_types"]) and id(item) not in excluded[side]
        ]
        data.append(_stereoset_sentences(items, corpus["label"]))

    def batchify(sentences):
        return [sentences[i : i + batch_size] for i in range(0, len(sentences), batch_size)]

    return batchify(data[0]), batchify(data[1])


def get_stereoset_data(
    cat_type: str = "intrasentence",
    paths=STEREO_PATHS,
    batch_size: int = 4,
    split: bool = True,
):
    """Load StereoSet gender examples and return them in batches.

    Items are shuffled before they are divided into two groups. The forget
    batches contain stereotype sentences from the first group; the retain
    batches contain anti-stereotype sentences from the second group. When
    ``split`` is false, both sentence sets are taken from all gender items.

    Same as ``load_corpora(DEFAULT_FORGET, DEFAULT_RETAIN, ...)``.

    Args:
        cat_type: StereoSet category to load (for example, ``intrasentence``).
        paths: JSON files whose ``data[cat_type]`` entries should be combined.
        batch_size: Maximum number of sentences in each returned batch. The
            final batch may contain fewer sentences.
        split: Whether to use separate groups for forgetting and retaining.

    Returns:
        A pair ``(forget_batches, retain_batches)``. Each value is a list of
        sentence batches, and each batch is a list of sentence dictionaries.

    Raises:
        ValueError: If ``batch_size`` is not a positive integer.
    """
    return load_corpora(
        DEFAULT_FORGET,
        DEFAULT_RETAIN,
        batch_size=batch_size,
        split=split,
        cat_type=cat_type,
        paths=paths,
    )


if __name__ == "__main__":
    forget_batches, retain_batches = get_stereoset_data()

    forget_count = sum(len(batch) for batch in forget_batches)
    retain_count = sum(len(batch) for batch in retain_batches)
    print(f"Forget: {forget_count} sentences in {len(forget_batches)} batches")
    print(f"Retain: {retain_count} sentences in {len(retain_batches)} batches")

    if forget_batches:
        print("First forget example:")
        print(forget_batches[0][0])
    if retain_batches:
        print("First retain example:")
        print(retain_batches[0][0])
