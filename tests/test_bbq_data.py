import json

from evaluation.bbq.data import (
    find_target_loc,
    find_unknown_loc,
    format_bbq_prompt,
    get_bbq_data,
    load_additional_metadata,
)


def test_load_additional_metadata(tmp_path):
    csv_file = tmp_path / "metadata.csv"
    csv_file.write_text(
        '"category","question_index","example_id","target_loc"\n'
        '"Gender_identity","1",0,2\n'
        '"Gender_identity","1",1,NA\n'
        '"Gender_identity","2",5,0\n',
        encoding="utf-8",
    )
    metadata = load_additional_metadata(csv_file)
    assert metadata[("Gender_identity", "1", 0)] == 2
    assert ("Gender_identity", "1", 1) not in metadata
    assert metadata[("Gender_identity", "2", 5)] == 0


def test_find_unknown_loc():
    item_with_unknown_tag = {
        "context_condition": "disambig",
        "answer_info": {
            "ans0": ["man", "man"],
            "ans1": ["Cannot determine", "unknown"],
            "ans2": ["woman", "woman"],
        },
    }
    assert find_unknown_loc(item_with_unknown_tag) == 1

    item_with_ambig_label = {
        "context_condition": "ambig",
        "label": 2,
        "answer_info": {
            "ans0": ["man", "man"],
            "ans1": ["woman", "woman"],
            "ans2": ["Undetermined", "other"],
        },
    }
    assert find_unknown_loc(item_with_ambig_label) == 2


def test_find_target_loc():
    metadata = {("Gender_identity", "1", 42): 0}
    item = {"category": "Gender_identity", "question_index": "1", "example_id": 42}
    assert find_target_loc(item, metadata) == 0

    item_fallback = {
        "category": "Gender_identity",
        "question_index": "2",
        "example_id": 99,
        "question_polarity": "neg",
        "additional_metadata": {"stereotyped_groups": ["woman"]},
        "answer_info": {
            "ans0": ["man", "man"],
            "ans1": ["Not enough information", "unknown"],
            "ans2": ["woman", "woman"],
        },
    }
    assert find_target_loc(item_fallback, {}) == 2


def test_format_bbq_prompt():
    item = {
        "context": "Alice and Bob went to the store.",
        "question": "Who bought the milk?",
        "ans0": "Alice",
        "ans1": "Bob",
        "ans2": "Cannot be determined",
    }
    prompt = format_bbq_prompt(item)
    assert "Context: Alice and Bob went to the store." in prompt
    assert "Question: Who bought the milk?" in prompt
    assert "A. Alice" in prompt
    assert "B. Bob" in prompt
    assert "C. Cannot be determined" in prompt
    assert prompt.endswith("Answer:")


def test_get_bbq_data_filter_by_gender_types(tmp_path):
    data_file = tmp_path / "Gender_identity.jsonl"

    items = [
        {
            "example_id": 0,
            "category": "Gender_identity",
            "additional_metadata": {"stereotyped_groups": ["F"]},
            "answer_info": {
                "ans0": ["m", "man"],
                "ans1": ["w", "woman"],
                "ans2": ["u", "unknown"],
            },
        },
        {
            "example_id": 1,
            "category": "Gender_identity",
            "additional_metadata": {"stereotyped_groups": ["M"]},
            "answer_info": {
                "ans0": ["m", "man"],
                "ans1": ["w", "woman"],
                "ans2": ["u", "unknown"],
            },
        },
        {
            "example_id": 2,
            "category": "Gender_identity",
            "additional_metadata": {"stereotyped_groups": ["trans"]},
            "answer_info": {
                "ans0": ["t", "trans"],
                "ans1": ["w", "woman"],
                "ans2": ["u", "unknown"],
            },
        },
    ]

    with open(data_file, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    # Load all
    all_loaded = get_bbq_data(paths=[data_file])
    assert len(all_loaded) == 3

    # Filter for 'F'
    f_loaded = get_bbq_data(paths=[data_file], gender_types=["F"])
    assert len(f_loaded) == 1
    assert f_loaded[0]["example_id"] == 0

    # Filter for ['M', 'trans']
    m_trans_loaded = get_bbq_data(paths=[data_file], gender_types=["M", "trans"])
    assert len(m_trans_loaded) == 2
    assert {x["example_id"] for x in m_trans_loaded} == {1, 2}
