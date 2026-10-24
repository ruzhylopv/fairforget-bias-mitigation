import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

from evaluation.bbq.evaluate import Evaluator
from evaluation.unlearn_eval import get_pipeline_args
from rmu.unlearn import get_args


def test_model_is_required():
    with pytest.raises(SystemExit):
        get_pipeline_args([])


def test_adaptive_rmu_defaults():
    args = get_pipeline_args(["--model_name_or_path", "lapa-12b-instruct"])

    assert args.lr == 5e-4
    assert args.alpha == [5.0]
    assert args.layer_id == 32
    assert args.layer_ids == [30, 31, 32]
    assert args.beta == 1.0
    assert args.split is True
    assert args.loss_positions == "words"
    assert args.save_model is True


def test_defaults_can_be_overridden():
    args = get_pipeline_args(
        [
            "--model_name_or_path",
            "qwen3-8b",
            "--layer_id",
            "24",
            "--layer_ids",
            "22,23,24",
            "--no-save_model",
        ]
    )

    assert args.layer_id == 24
    assert args.layer_ids == [22, 23, 24]
    assert args.save_model is False


def test_plain_unlearn_cli_keeps_its_defaults():
    args = get_args(argv=[])

    assert args.model_name_or_path == "HuggingFaceH4/zephyr-7b-beta"
    assert args.lr == 5e-5
    assert args.beta is None


def test_evaluator_uses_a_loaded_model():
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    model = AutoModelForCausalLM.from_config(config).train()
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # The label path does not exist: without model= this would fall back to mock mode.
    evaluator = Evaluator("runs/does-not-exist", model=model, tokenizer=tokenizer)

    assert evaluator.mock is False
    assert evaluator.model is model
    assert not model.training
    item = {
        "context": "A man and a woman talked.",
        "question": "Who was rude?",
        "ans0": "The man",
        "ans1": "Cannot be determined",
        "ans2": "The woman",
    }
    assert all(p in (0, 1, 2) for p in evaluator.predict_batch([item, item]))
