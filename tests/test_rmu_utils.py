import pytest

from rmu import data


@pytest.fixture
def stereo_data(monkeypatch):
    entries = [
        {
            "bias_type": "gender",
            "sentences": [
                {"id": f"{i}-stereo", "gold_label": "stereotype"},
                {"id": f"{i}-anti", "gold_label": "anti-stereotype"},
                {"id": f"{i}-unrelated", "gold_label": "unrelated"},
            ],
        }
        for i in range(4)
    ]
    entries.append(
        {
            "bias_type": "profession",
            "sentences": [{"id": "other-bias", "gold_label": "stereotype"}],
        }
    )
    monkeypatch.setattr(data, "get_json_dict", lambda path: {"data": {"intrasentence": entries}})
    monkeypatch.setattr(data.random, "shuffle", lambda items: None)
    return entries


def test_returns_filtered_forget_and_retain_batches(stereo_data):
    forget_batches, retain_batches = data.get_stereoset_data(paths=["fixture.json"], batch_size=3)

    assert [[sentence["id"] for sentence in batch] for batch in forget_batches] == [
        ["0-stereo", "1-stereo"]
    ]
    assert [[sentence["id"] for sentence in batch] for batch in retain_batches] == [
        ["2-anti", "3-anti"]
    ]


def test_batches_are_limited_to_batch_size_and_keep_remainder(stereo_data):
    forget_batches, retain_batches = data.get_stereoset_data(
        paths=["fixture.json"], batch_size=1
    )

    assert [len(batch) for batch in forget_batches] == [1, 1]
    assert [len(batch) for batch in retain_batches] == [1, 1]


def test_unsplit_mode_uses_all_items_for_both_sets(stereo_data):
    forget_batches, retain_batches = data.get_stereoset_data(
        paths=["fixture.json"], batch_size=4, split=False
    )

    assert [sentence["id"] for batch in forget_batches for sentence in batch] == [
        "0-stereo",
        "1-stereo",
        "2-stereo",
        "3-stereo",
    ]
    assert [sentence["id"] for batch in retain_batches for sentence in batch] == [
        "0-anti",
        "1-anti",
        "2-anti",
        "3-anti",
    ]


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_rejects_invalid_batch_size(batch_size):
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        data.get_stereoset_data(paths=[], batch_size=batch_size)


def ids(batches):
    return [sentence["id"] for batch in batches for sentence in batch]


def test_default_corpora_match_get_stereoset_data(stereo_data):
    assert data.load_corpora(paths=["fixture.json"], batch_size=3) == data.get_stereoset_data(
        paths=["fixture.json"], batch_size=3
    )


def test_split_keeps_shared_items_apart_for_any_labels(stereo_data):
    forget, retain = data.load_corpora(
        "stereoset:anti-stereotype", "stereoset:unrelated", paths=["fixture.json"]
    )

    assert ids(forget) == ["0-anti", "1-anti"]
    assert ids(retain) == ["2-unrelated", "3-unrelated"]


def test_split_uses_items_only_one_corpus_can_draw_from_in_full(stereo_data):
    forget, retain = data.load_corpora(
        "stereoset:stereotype:gender",
        "stereoset:stereotype:profession",
        paths=["fixture.json"],
    )

    assert ids(forget) == ["0-stereo", "1-stereo", "2-stereo", "3-stereo"]
    assert ids(retain) == ["other-bias"]


def test_all_bias_types(stereo_data):
    forget, _ = data.load_corpora(
        "stereoset:stereotype:all", "stereoset:anti-stereotype", split=False, paths=["fixture.json"]
    )

    assert ids(forget) == ["0-stereo", "1-stereo", "2-stereo", "3-stereo", "other-bias"]


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("retain.txt", "The nurse is tall.\n\nThe pilot is late.\n"),
        ("retain.jsonl", '{"sentence": "The nurse is tall."}\n"The pilot is late."\n'),
        ("retain.json", '[{"text": "The nurse is tall."}, "The pilot is late."]'),
        ("retain.csv", "id,sentence\nretain-0,The nurse is tall.\nretain-1,The pilot is late.\n"),
        ("retain.tsv", "text\nThe nurse is tall.\nThe pilot is late.\n"),
    ],
)
def test_file_corpus_formats(stereo_data, tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")

    forget, retain = data.load_corpora(
        data.DEFAULT_FORGET, str(path), batch_size=4, paths=["fixture.json"]
    )

    # A file retain corpus doesn't compete for StereoSet items: forget gets all of them.
    assert ids(forget) == ["0-stereo", "1-stereo", "2-stereo", "3-stereo"]
    assert [s["sentence"] for batch in retain for s in batch] == [
        "The nurse is tall.",
        "The pilot is late.",
    ]
    assert ids(retain) == ["retain-0", "retain-1"]


def test_file_corpora_on_both_sides_skip_stereoset(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "get_json_dict", lambda path: pytest.fail("StereoSet was loaded"))
    (tmp_path / "f.txt").write_text("a\nb\nc\n", encoding="utf-8")
    (tmp_path / "r.txt").write_text("d\n", encoding="utf-8")

    forget, retain = data.load_corpora(
        str(tmp_path / "f.txt"), str(tmp_path / "r.txt"), batch_size=2
    )

    assert [[s["sentence"] for s in batch] for batch in forget] == [["a", "b"], ["c"]]
    assert [[s["sentence"] for s in batch] for batch in retain] == [["d"]]


@pytest.mark.parametrize(
    "spec", ["stereoset:biased", "stereoset:stereotype:gender:extra", "stereoset:stereotype:"]
)
def test_bad_stereoset_spec(spec):
    with pytest.raises(ValueError):
        data.parse_corpus(spec)


def test_missing_corpus_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_corpora(data.DEFAULT_FORGET, str(tmp_path / "nope.txt"), paths=[])


def test_is_paired():
    assert data.is_paired(data.DEFAULT_FORGET, data.DEFAULT_RETAIN)
    assert data.is_paired("stereoset:stereotype:all", "stereoset:unrelated:all")
    assert not data.is_paired(data.DEFAULT_FORGET, "stereoset:anti-stereotype:race")
    assert not data.is_paired(data.DEFAULT_FORGET, data.DEFAULT_FORGET)
    assert not data.is_paired(data.DEFAULT_FORGET, "retain.txt")
