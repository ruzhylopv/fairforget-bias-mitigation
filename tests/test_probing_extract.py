"""Pure functions of probing.extract on small fake tensors (CPU, no model)."""

import numpy as np
import pandas as pd
import pytest
import torch

from probing.extract import (
    FLOAT16_MAX,
    choose_store_dtype,
    gather_positions,
    length_batches,
    word_logprob,
)
from probing.io import POSITION_COLUMNS, POSITIONS

PAD = 0


def left_padded(lengths, width=None):
    """input_ids where row b is [pad.., 100b + 0, 100b + 1, ...] (token value = 100 b + t)."""
    width = width or max(lengths)
    ids = torch.full((len(lengths), width), PAD)
    for b, n in enumerate(lengths):
        ids[b, width - n :] = 100 * b + torch.arange(n)
    return ids


def test_gather_positions_counts_negative_indices_from_the_end_of_left_padded_rows():
    lengths = [3, 5, 2]
    ids = left_padded(lengths)  # (3, 5)
    # Hidden state = token value repeated over 4 dims, so the result says which token it was.
    hidden = ids[..., None].float().expand(-1, -1, 4)

    got = gather_positions(hidden, np.array([-1, -4, -2]))

    assert got.shape == (3, 4)
    # Row 0 last token = 2; row 1 is unpadded, -4 -> token 1; row 2 -2 -> its first token 200.
    assert got[:, 0].tolist() == [2.0, 101.0, 200.0]


def test_gather_positions_same_token_whatever_the_padding():
    # The same 3-token sentence in batches of different widths: -2 is always its 2nd token.
    for width in (3, 6, 10):
        hidden = left_padded([3], width)[..., None].float()
        assert gather_positions(hidden, [-2]).item() == 1.0


def test_gather_positions_rejects_bad_indices():
    hidden = torch.zeros(2, 4, 3)
    with pytest.raises(IndexError):
        gather_positions(hidden, [-5, -1])
    with pytest.raises(ValueError, match="one index per row"):
        gather_positions(hidden, [-1, -1, -1])


def fake_logits(ids, vocab=500, seed=0):
    return torch.randn(*ids.shape, vocab, generator=torch.Generator().manual_seed(seed))


def brute_force(logits, ids, first, last):
    """Loop version: x_p is scored by log_softmax(z[p - 1])."""
    t = ids.shape[1]
    out = []
    for b in range(ids.shape[0]):
        a, z = first[b] % t, last[b] % t
        lp = torch.log_softmax(logits[b].double(), dim=-1)
        out.append(sum(lp[p - 1, ids[b, p]].item() for p in range(a, z + 1)))
    return np.array(out)


def test_word_logprob_matches_a_loop_with_one_position_shift():
    ids = left_padded([4, 7, 5])
    logits = fake_logits(ids)
    first, last = np.array([-2, -5, -3]), np.array([-1, -3, -3])

    got = word_logprob(logits, ids, first, last)

    assert got.dtype == torch.float32 and got.shape == (3,)
    np.testing.assert_allclose(got.numpy(), brute_force(logits, ids, first, last), rtol=1e-5)


def test_word_logprob_uses_the_previous_position_not_the_same_one():
    ids = left_padded([4, 4])
    logits = torch.full((*ids.shape, 500), -30.0)
    # Position p - 1 predicts x_p with certainty; position p predicts nothing useful.
    for b in range(2):
        for p in range(1, ids.shape[1]):
            logits[b, p - 1, ids[b, p]] = 30.0

    got = word_logprob(logits, ids, [-2, -3], [-1, -2])

    np.testing.assert_allclose(got.numpy(), [0.0, 0.0], atol=1e-6)
    # Unshifted (z_p scoring x_p) would give ~-60 per token instead.


def test_word_logprob_sums_over_subtokens_and_accepts_logprobs():
    ids = left_padded([6])
    logits = fake_logits(ids)
    single = [word_logprob(logits, ids, [p], [p]).item() for p in (-3, -2, -1)]

    total = word_logprob(logits, ids, [-3], [-1]).item()
    from_logprobs = word_logprob(torch.log_softmax(logits, -1), ids, [-3], [-1]).item()

    assert total == pytest.approx(sum(single), rel=1e-5)  # sum, not mean
    assert from_logprobs == pytest.approx(total, rel=1e-5)
    assert total < 0


def test_word_logprob_bf16_logits_are_normalised_in_float32():
    ids = left_padded([5, 3])
    logits = fake_logits(ids)
    got = word_logprob(logits.bfloat16(), ids, [-2, -1], [-1, -1])
    want = brute_force(logits.bfloat16().float(), ids, [-2, -1], [-1, -1])
    assert got.dtype == torch.float32
    np.testing.assert_allclose(got.numpy(), want, rtol=1e-5)


def test_word_logprob_rejects_spans_without_a_previous_position():
    ids = left_padded([3])
    with pytest.raises(ValueError, match="preceding position"):
        word_logprob(fake_logits(ids), ids, [-3], [-1])
    with pytest.raises(ValueError, match="last must be >= first"):
        word_logprob(fake_logits(ids), ids, [-1], [-2])


def test_store_dtype_falls_back_to_float32_only_on_overflow():
    ok = {"first_diff": [1.0, 300.0], "last_subtok": [2.0, FLOAT16_MAX]}
    big = {"first_diff": [1.0, 300.0], "last_subtok": [2.0, 7e4]}
    assert choose_store_dtype("auto", ok) == "float16"
    assert choose_store_dtype("auto", big) == "float32"
    assert choose_store_dtype("float32", ok) == "float32"
    with pytest.raises(ValueError, match="overflows float16"):
        choose_store_dtype("float16", big)


def test_length_batches_never_mix_lengths_and_cover_every_row_once():
    n_tokens = np.array([9, 7, 9, 12, 7, 9, 9, 7, 12, 9])
    batches = length_batches(n_tokens, batch_size=3)

    rows = np.concatenate(batches)
    assert sorted(rows.tolist()) == list(range(len(n_tokens)))
    assert all(len(set(n_tokens[b])) == 1 and len(b) <= 3 for b in batches)
    assert [len(b) for b in batches] == [3, 3, 2, 2]  # 7 x3; 9 x5 -> 3 + 2; 12 x2
    assert batches[1].tolist() == [0, 2, 5]  # meta order within a length


class _StubTokenizer:
    """Whitespace tokenizer with BOS = 1 and left padding with 0 (enough for extract)."""

    bos_token_id = 1

    def __init__(self, sentences):
        words = sorted({w for s in sentences for w in s.split()})
        self.ids = {w: i + 2 for i, w in enumerate(words)}
        self.words = {i: w for w, i in self.ids.items()} | {1: "<s>", 0: "<pad>"}

    def __call__(self, text, padding=False, return_tensors=None):
        if isinstance(text, str):
            return {"input_ids": [1] + [self.ids.get(w, 2) for w in text.split()]}
        seqs = [[1] + [self.ids[w] for w in s.split()] for s in text]
        width = max(map(len, seqs))
        ids = torch.tensor([[0] * (width - len(q)) + q for q in seqs])
        mask = torch.tensor([[0] * (width - len(q)) + [1] * len(q) for q in seqs])
        return {"input_ids": ids, "attention_mask": mask}

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(self.words[int(i)] for i in ids)

    def convert_ids_to_tokens(self, ids):
        return [self.words[int(i)] for i in ids] if isinstance(ids, list) else self.words[ids]


class _StubModel(torch.nn.Module):
    """hidden_states[k] = embedding + k * (position + 1) * (number of padded tokens + 1): any
    padding, a lost position or a row written to the wrong place changes the result."""

    def __init__(self, vocab, hidden=4, n_layers=2):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(torch.nn.Identity() for _ in range(n_layers))
        self.config = type("C", (), {"get_text_config": lambda self: self})()
        self.config.hidden_size = hidden

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids, attention_mask, position_ids, output_hidden_states=True):
        pads = (attention_mask == 0).sum(1, keepdim=True)[..., None].float()
        e = self.embed(input_ids)
        pos = position_ids[..., None].float()
        hs = [e + k * (pos + 1) * (pads + 1) for k in range(len(self.model.layers) + 1)]
        logits = hs[-1] @ self.embed.weight.T
        return type("Out", (), {"hidden_states": tuple(hs), "logits": logits})()


def test_extract_in_equal_length_batches_matches_single_sentences_row_by_row():
    from probing.extract import _model_inputs, extract

    sentences = [
        "the caring mother",
        "the uncaring mother",
        "a very strong man indeed",
        "a very weak man indeed",
        "she bakes",
        "he bakes",
        "the nurse was kind",
        "the nurse was rude",
    ]
    tok = _StubTokenizer(sentences)
    torch.manual_seed(0)
    model = _StubModel(vocab=len(tok.ids) + 2)
    n = np.array([len(s.split()) + 1 for s in sentences])
    meta = pd.DataFrame(
        {
            "row": range(len(sentences)),
            "sentence": sentences,
            "word": [s.split()[-2] if len(s.split()) > 2 else s.split()[-1] for s in sentences],
            "n_tokens": n,
            "idx_first_diff": [-2 if len(s.split()) > 2 else -1 for s in sentences],
            "idx_last_subtok": [-2 if len(s.split()) > 2 else -1 for s in sentences],
        }
    )

    acts, ell, checks = extract(model, tok, meta, batch_size=1 + 2)  # mixed lengths, small

    assert checks["decode_mismatches"] == [] and checks["padding_max_rel_diff"] == 0.0
    for r, s in enumerate(sentences):
        single = model(**_model_inputs(tok, [s], "cpu")).hidden_states
        for p in POSITIONS:
            idx = [int(meta.loc[r, POSITION_COLUMNS[p]])]
            for k, h in enumerate(single):
                np.testing.assert_array_equal(
                    acts[p][k][r], gather_positions(h, idx)[0].detach().numpy()
                )
    assert np.isfinite(ell).all() and ell.shape == (len(sentences),)
