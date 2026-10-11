"""Layer lookup must work for text-only models (Llama, Qwen) and multimodal Gemma 3 (Lapa)."""

import pytest
import torch
from transformers import AutoModelForCausalLM, Gemma3Config, LlamaConfig

from rmu.utils import get_decoder_layers, get_hidden_size, get_params

HIDDEN = 16


def tiny_llama():
    config = LlamaConfig(
        hidden_size=HIDDEN,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=64,
    )
    with torch.device("meta"):
        return AutoModelForCausalLM.from_config(config)


def tiny_gemma3():
    # Same architecture as Lapa (Gemma3ForConditionalGeneration), with tiny sizes.
    config = Gemma3Config(
        text_config={
            "hidden_size": HIDDEN,
            "intermediate_size": 32,
            "num_hidden_layers": 3,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": 64,
        },
        vision_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 28,
            "patch_size": 14,
        },
        mm_tokens_per_image=4,
    )
    with torch.device("meta"):
        return AutoModelForCausalLM.from_config(config)


@pytest.fixture(params=[tiny_llama, tiny_gemma3], ids=["llama", "gemma3"])
def model(request):
    return request.param()


def test_finds_all_decoder_layers(model):
    assert len(get_decoder_layers(model)) == 3


def test_hidden_size_comes_from_text_config(model):
    assert get_hidden_size(model) == HIDDEN


def test_selects_down_proj_by_name(model):
    layers = get_decoder_layers(model)

    params = get_params(model, [0, 2], ["mlp.down_proj.weight"])

    assert params[0] is layers[0].mlp.down_proj.weight
    assert params[1] is layers[2].mlp.down_proj.weight


def test_unknown_parameter_name_lists_available_ones(model):
    with pytest.raises(ValueError, match="mlp.down_proj.weight"):
        get_params(model, [0], ["mlp.no_such.weight"])
