"""Greedy and beam-search decoding."""

import torch

from hinglish_mt import OUTPUT_DIR
from hinglish_mt.data import encode_source, make_batches, pad
from hinglish_mt.model import Seq2Seq
from hinglish_mt.tokenise import BOS, EOS, PAD, load_tokeniser

RUNS_DIR = OUTPUT_DIR / "runs"


def max_steps(src_len, max_len):
    return min(max_len, int(src_len * 1.5) + 10)


def next_word_scores(model, state, weights, src):
    """Log-probability of every word. <pad> and <s> can never be chosen."""
    scores = model.word_scores(state, weights, src)
    scores[:, PAD] = float("-inf")
    scores[:, BOS] = float("-inf")
    return scores


@torch.no_grad()
def greedy(model, src, src_len, max_len=100):
    """Translate a batch. At every step take the most likely word."""
    memory, keys, mask, hidden = model.encode(src, src_len)
    batch_size = src.size(0)
    tokens = torch.full((batch_size,), BOS, device=src.device)
    done = torch.zeros(batch_size, dtype=torch.bool, device=src.device)
    steps = []
    for _ in range(max_steps(src_len.max().item(), max_len)):
        state, hidden, weights = model.step(tokens, hidden, memory, keys, mask)
        tokens = next_word_scores(model, state, weights, src).argmax(dim=-1)
        steps.append(tokens)
        done = done | (tokens == EOS)
        if done.all():
            break

    outputs = []
    for row in torch.stack(steps, dim=1).tolist():
        if EOS in row:
            row = row[: row.index(EOS)]
        outputs.append(row)
    return outputs


@torch.no_grad()
def beam_search(model, src, beam=4, max_len=100):
    """Translate one sentence. Keep the `beam` best partial translations at every step.

    Finished translations are ranked by average log-probability per word.
    """
    src = src.unsqueeze(0)
    src_len = torch.tensor([src.size(1)])
    memory, keys, mask, hidden = model.encode(src, src_len)

    # Each entry is (score, tokens, hidden state).
    live = [(0.0, [BOS], hidden)]
    finished = []

    for _ in range(max_steps(src.size(1), max_len)):
        candidates = []
        for score, tokens, hidden in live:
            last_word = torch.tensor([tokens[-1]], device=src.device)
            state, new_hidden, weights = model.step(last_word, hidden, memory, keys, mask)
            word_scores = next_word_scores(model, state, weights, src)[0]
            best = word_scores.topk(beam)
            for word_score, word in zip(best.values.tolist(), best.indices.tolist()):
                candidates.append((score + word_score, tokens + [word], new_hidden))

        candidates.sort(key=candidate_score, reverse=True)
        live = []
        for score, tokens, hidden in candidates[:beam]:
            if tokens[-1] == EOS:
                words = tokens[1:-1]
                average = score / (len(tokens) - 1)
                finished.append((average, words))
            else:
                live.append((score, tokens, hidden))

        if len(live) == 0 or len(finished) >= beam:
            break

    if len(finished) == 0:
        return live[0][1][1:]
    finished.sort(key=candidate_score, reverse=True)
    return finished[0][1]


def candidate_score(candidate):
    return candidate[0]


def translate(model, tokeniser, sources, beam=1, max_len=100, max_tokens=4000):
    """Translate cleaned, masked sentences. Returns masked translations in the same order."""
    device = next(model.parameters()).device
    use_amp = device.type == "cuda"
    was_training = model.training
    model.eval()

    encoded = []
    for text in sources:
        encoded.append(encode_source(tokeniser, text, max_len))
    outputs = [""] * len(sources)

    with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
        if beam > 1:
            for i, ids in enumerate(encoded):
                src = torch.tensor(ids, device=device)
                output_ids = beam_search(model, src, beam, max_len)
                outputs[i] = tokeniser.decode(output_ids)
        else:
            lengths = []
            for ids in encoded:
                lengths.append(len(ids))
            for indices in make_batches(lengths, max_tokens):
                batch_ids = []
                batch_lengths = []
                for i in indices:
                    batch_ids.append(encoded[i])
                    batch_lengths.append(len(encoded[i]))
                src = pad(batch_ids, device)
                src_len = torch.tensor(batch_lengths)
                batch_outputs = greedy(model, src, src_len, max_len)
                for i, output_ids in zip(indices, batch_outputs):
                    outputs[i] = tokeniser.decode(output_ids)

    model.train(was_training)
    return outputs


def load_run(name):
    """Load a trained run. Returns the model, its tokeniser and the saved checkpoint."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(RUNS_DIR / name / "best.pt", map_location=device, weights_only=True)
    model = Seq2Seq(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    tokeniser = load_tokeniser(checkpoint["config"]["tokeniser"])
    return model, tokeniser, checkpoint
