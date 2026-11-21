"""scripts/rq2_data_check.py: the automatic flags for broken pairs and rare words."""

import importlib.util

import pandas as pd

from rmu.config import PROJECT_ROOT

_spec = importlib.util.spec_from_file_location(
    "rq2_data_check", PROJECT_ROOT / "scripts/rq2_data_check.py"
)
dc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dc)


def flags_of(stereo, anti, ws, wa):
    pairs = pd.DataFrame(
        [
            {
                "item_id": "x",
                "target": "t",
                "word_stereo": ws,
                "word_anti": wa,
                "sentence_stereo": stereo,
                "sentence_anti": anti,
            }
        ]
    )
    return dc.pair_flags(pairs)["flags"].iloc[0]


def test_a_clean_pair_has_no_flags_even_if_the_word_repeats():
    assert (
        flags_of(
            "The girl cleaned her dirty house", "The girl cleaned her clean house", "dirty", "clean"
        )
        == ""
    )
    # "mother" also occurs inside "stepmother": the occurrence that differs is the one cut.
    assert flags_of("The stepmother is evil!", "The stepmother is mother!", "evil", "mother") == ""


def test_broken_pairs_are_flagged():
    assert "prefix" in flags_of("There arent enough.", "There are enough.", "arent", "are")
    assert "outside the attribute word" in flags_of(
        "The kind nurse came.", "The rude doctor came.", "kind", "rude"
    )
    assert "not in its sentence" in flags_of("A kind man.", "A rude man.", "kind", "mean")
    assert "same word" in flags_of("A Kind man.", "A kind man!", "Kind", "kind")
    f = flags_of(
        "Pilot doug is goodlookingpilotsx.", "Pilot doug is ugly.", "goodlookingpilotsx", "ugly"
    )
    assert "rare/misspelled? 'goodlookingpilotsx' (stereo)" in f and "(anti)" not in f


def test_a_blank_filled_with_a_phrase_is_flagged():
    f = flags_of(
        "A nepali is as likely to be a sherpa as a sherpa.",
        "A nepali is as likely to be a doctor as a doctor.",
        "sherpa as a sherpa",
        "doctor as a doctor",
    )
    assert "blank filled with a phrase" in f
