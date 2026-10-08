import pytest

from rmu import utils


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
    monkeypatch.setattr(utils, "get_json_dict", lambda path: {"data": {"intrasentence": entries}})
    monkeypatch.setattr(utils.random, "shuffle", lambda items: None)
    return entries


def test_returns_filtered_forget_and_retain_batches(stereo_data):
    forget_batches, retain_batches = utils.get_stereoset_data(paths=["fixture.json"], batch_size=3)

    assert [[sentence["id"] for sentence in batch] for batch in forget_batches] == [
        ["0-stereo", "1-stereo"]
    ]
    assert [[sentence["id"] for sentence in batch] for batch in retain_batches] == [
        ["2-anti", "3-anti"]
    ]


def test_batches_are_limited_to_batch_size_and_keep_remainder(stereo_data):
    forget_batches, retain_batches = utils.get_stereoset_data(
        paths=["fixture.json"], batch_size=1
    )

    assert [len(batch) for batch in forget_batches] == [1, 1]
    assert [len(batch) for batch in retain_batches] == [1, 1]


def test_unsplit_mode_uses_same_group_for_both_sets(stereo_data):
    forget_batches, retain_batches = utils.get_stereoset_data(
        paths=["fixture.json"], batch_size=4, split=False
    )

    assert [sentence["id"] for batch in forget_batches for sentence in batch] == [
        "2-stereo",
        "3-stereo",
    ]
    assert [sentence["id"] for batch in retain_batches for sentence in batch] == [
        "2-anti",
        "3-anti",
    ]


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_rejects_invalid_batch_size(batch_size):
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        utils.get_stereoset_data(paths=[], batch_size=batch_size)
