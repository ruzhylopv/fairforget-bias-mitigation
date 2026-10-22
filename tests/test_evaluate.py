import json

from evaluation.bbq.evaluate import Evaluator, run_evaluation


def test_evaluator_mock_mode():
    evaluator = Evaluator(model_name_or_path="lapa-12b-instruct", mock=True)
    assert evaluator.mock is True

    items = [
        {"example_id": 0, "label": 1},
        {"example_id": 1, "label": 2},
    ]
    preds = evaluator.predict_batch(items)
    assert len(preds) == 2
    assert all(p in (0, 1, 2) for p in preds)


def test_run_evaluation_pipeline(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    sample_items = [
        {
            "example_id": 0,
            "question_index": "1",
            "category": "Gender_identity",
            "context_condition": "ambig",
            "question_polarity": "neg",
            "context": "Context 1",
            "question": "Question 1",
            "ans0": "man",
            "ans1": "Cannot answer",
            "ans2": "woman",
            "label": 1,
            "additional_metadata": {"stereotyped_groups": ["F"]},
            "answer_info": {
                "ans0": ["man", "man"],
                "ans1": ["Cannot answer", "unknown"],
                "ans2": ["woman", "woman"],
            },
        },
        {
            "example_id": 1,
            "question_index": "1",
            "category": "Gender_identity",
            "context_condition": "disambig",
            "question_polarity": "neg",
            "context": "Context 2",
            "question": "Question 2",
            "ans0": "man",
            "ans1": "Cannot answer",
            "ans2": "woman",
            "label": 0,
            "additional_metadata": {"stereotyped_groups": ["F"]},
            "answer_info": {
                "ans0": ["man", "man"],
                "ans1": ["Cannot answer", "unknown"],
                "ans2": ["woman", "woman"],
            },
        },
    ]

    jsonl_path = data_dir / "Gender_identity.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in sample_items:
            f.write(json.dumps(item) + "\n")

    meta_path = data_dir / "additional_metadata.csv"
    meta_path.write_text(
        '"category","question_index","example_id","target_loc"\n'
        '"Gender_identity","1",0,2\n'
        '"Gender_identity","1",1,2\n',
        encoding="utf-8",
    )

    out_dir = tmp_path / "outputs"
    metrics = run_evaluation(
        model_name_or_path="lapa-12b-instruct",
        paths=[jsonl_path],
        metadata_path=meta_path,
        gender_types=["F"],
        output_dir=out_dir,
        batch_size=2,
        mock=True,
    )

    assert metrics["total_examples"] == 2
    assert "overall_accuracy" in metrics
    assert "s_dis" in metrics
    assert "s_amb" in metrics
    assert metrics["gender_types"] == ["F"]

    assert (out_dir / "eval_results.json").is_file()
    assert (out_dir / "eval_summary.csv").is_file()
    assert (out_dir / "predictions.jsonl").is_file()

    loaded_results = json.loads((out_dir / "eval_results.json").read_text(encoding="utf-8"))
    assert loaded_results["model_name_or_path"] == "lapa-12b-instruct"
