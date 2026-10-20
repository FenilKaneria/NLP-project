import random

import pytest
import torch
import torch.nn.functional as F

from hinglish_mt.data import collate, encode_pairs, make_batches, pair_lengths
from hinglish_mt.decode import beam_search, greedy, translate
from hinglish_mt.evaluate import evaluate_rows, restore_row, score
from hinglish_mt.model import Seq2Seq
from hinglish_mt.tokenise import BOS, EOS, PAD, UNK, BpeTokeniser, WordTokeniser
from hinglish_mt.train import batch_loss

ROWS = [
    {"src": "<user> kya haal hai bhai", "tgt": "<user> how are you brother"},
    {"src": "main ghar ja raha hoon", "tgt": "i am going home"},
    {"src": "ye movie bahut acchi hai !", "tgt": "this movie is very good !"},
    {"src": "kal milte hai", "tgt": "see you tomorrow"},
]


@pytest.fixture(scope="module")
def tokeniser():
    return WordTokeniser.build([text for row in ROWS for text in row.values()], min_freq=1)


def small_model(tokeniser, dropout=0.0, copy=False):
    torch.manual_seed(0)
    return Seq2Seq(tokeniser.size, emb_dim=32, hidden_dim=32, layers=2, dropout=dropout, copy=copy)


def test_word_tokeniser_round_trip_and_unknown(tokeniser):
    ids = tokeniser.encode("<user> kya haal hai zzz")
    assert ids[-1] == UNK and UNK not in ids[:-1]
    assert tokeniser.decode([BOS] + ids + [EOS, PAD]) == "<user> kya haal hai"


def test_word_tokeniser_minimum_frequency():
    tokeniser = WordTokeniser.build(["a a b", "a c"], min_freq=2)
    assert tokeniser.words[4:] == ["a"]


def test_bpe_tokeniser_round_trip(tmp_path):
    texts = [text for row in ROWS for text in row.values()] * 20
    BpeTokeniser.train(texts, tmp_path / "bpe.model", vocab_size=60)
    bpe = BpeTokeniser(tmp_path / "bpe.model")
    assert (bpe.sp.pad_id(), bpe.sp.unk_id(), bpe.sp.bos_id(), bpe.sp.eos_id()) == (PAD, UNK, BOS, EOS)
    assert bpe.pieces("<user> kya").count("<user>") == 1
    for text in texts[:8]:
        assert bpe.decode(bpe.encode(text)) == text


def test_batches_cover_every_example_within_the_token_limit():
    lengths = [random.Random(i).randint(1, 50) for i in range(500)]
    batches = make_batches(lengths, max_tokens=200, rng=random.Random(0))
    assert sorted(i for batch in batches for i in batch) == list(range(500))
    assert all(len(batch) * max(lengths[i] for i in batch) <= 200 for batch in batches)
    assert make_batches(lengths, 200, random.Random(0)) == batches
    assert make_batches(lengths, 200, random.Random(1)) != batches


def test_collate_shapes_and_boundary_marks(tokeniser):
    examples = encode_pairs(ROWS, tokeniser, max_len=100)
    batch = collate(examples, "cpu")
    assert batch["src"].shape == (4, max(len(src) for src, _ in examples))
    assert batch["tgt_in"].shape == batch["tgt_out"].shape
    assert (batch["tgt_in"][:, 0] == BOS).all()
    assert all(src[-1] == EOS for src, _ in examples)
    assert batch["tgt_out"][3].tolist().count(EOS) == 1
    assert pair_lengths(examples) == [max(len(src), len(tgt) + 1) for src, tgt in examples]


def test_long_pairs_are_cut_to_max_len(tokeniser):
    row = {"src": "hai " * 50, "tgt": "you " * 50}
    src, tgt = encode_pairs([row], tokeniser, max_len=10)[0]
    assert len(src) == 10 and src[-1] == EOS and len(tgt) == 9


def test_forward_shape_and_padding_does_not_change_the_result(tokeniser):
    model = small_model(tokeniser).eval()
    examples = encode_pairs(ROWS, tokeniser, max_len=100)
    together = model(**{k: v for k, v in collate(examples, "cpu").items() if k != "tgt_out"})
    assert together.shape[:2] == collate(examples, "cpu")["tgt_in"].shape and together.size(2) == tokeniser.size
    alone = model(**{k: v for k, v in collate(examples[3:], "cpu").items() if k != "tgt_out"})
    assert torch.allclose(together[3, : alone.size(1)], alone[0], atol=1e-5)


@pytest.mark.parametrize("copy", [False, True])
def test_beam_of_one_equals_greedy(tokeniser, copy):
    model = small_model(tokeniser, copy=copy).eval()
    for src, _ in encode_pairs(ROWS, tokeniser, max_len=100):
        src = torch.tensor(src)
        assert beam_search(model, src, beam=1) == greedy(model, src.unsqueeze(0), torch.tensor([len(src)]))[0]


def test_loss_is_the_same_as_pytorch_label_smoothing(tokeniser):
    model = small_model(tokeniser).eval()
    batch = collate(encode_pairs(ROWS, tokeniser, max_len=100), "cpu")
    scores = model(batch["src"], batch["src_len"], batch["tgt_in"]).flatten(0, 1)
    targets = batch["tgt_out"].flatten()
    expected = F.cross_entropy(scores, targets, ignore_index=PAD, label_smoothing=0.1)
    assert batch_loss(model, batch, label_smoothing=0.1).item() == pytest.approx(expected.item(), abs=1e-5)


def test_copy_scores_are_probabilities_and_favour_source_words(tokeniser):
    model = small_model(tokeniser, copy=True).eval()
    batch = collate(encode_pairs(ROWS, tokeniser, max_len=100), "cpu")
    scores = model(batch["src"], batch["src_len"], batch["tgt_in"])
    assert torch.allclose(scores.exp().sum(dim=-1), torch.ones(scores.shape[:2]), atol=1e-4)

    # With the gate forced to "copy only", every word outside the source gets no probability.
    torch.nn.init.zeros_(model.copy_gate.weight)
    torch.nn.init.constant_(model.copy_gate.bias, -50.0)
    scores = model(batch["src"], batch["src_len"], batch["tgt_in"])
    in_source = torch.zeros(tokeniser.size, dtype=torch.bool)
    in_source[batch["src"][1]] = True
    assert scores[1].exp()[:, ~in_source].max().item() < 1e-6


@pytest.mark.parametrize("copy", [False, True])
def test_model_memorises_a_few_pairs_and_decodes_them(tokeniser, copy):
    model = small_model(tokeniser, copy=copy)
    batch = collate(encode_pairs(ROWS, tokeniser, max_len=100), "cpu")
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(150):
        loss = batch_loss(model, batch, label_smoothing=0.0)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    assert loss.item() < 0.05
    sources, targets = [row["src"] for row in ROWS], [row["tgt"] for row in ROWS]
    assert translate(model, tokeniser, sources) == targets
    assert translate(model, tokeniser, sources, beam=4) == targets
    assert model.training


def test_scores_are_perfect_for_identical_text_and_restore_uses_the_source():
    perfect = score(["how are you brother"], ["how are you brother"])
    assert perfect["chrf"] == pytest.approx(100) and perfect["bleu4"] == pytest.approx(100)
    row = {"src_raw": "@Ravi kya haal hai #Monday", "tgt_raw": "@Ravi how are you #Monday", "tgt": "<user> how are you <hashtag>"}
    assert restore_row(row, "<user> how are you <hashtag>") == ("@ravi how are you #monday", "@ravi how are you #monday")
    result = evaluate_rows([row], ["<user> how are you <hashtag>"])
    assert result["masked"]["chrf"] == pytest.approx(100) and result["restored"]["chrf"] == pytest.approx(100)
