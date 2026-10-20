"""Turn sentence pairs into padded tensors, batched by token count."""

import torch

from hinglish_mt.tokenise import BOS, EOS, PAD


def encode_source(tokeniser, text, max_len):
    ids = tokeniser.encode(text)
    ids = ids[: max_len - 1]
    return ids + [EOS]


def encode_pairs(rows, tokeniser, max_len):
    """Returns a list of (source ids, target ids)."""
    examples = []
    for row in rows:
        src = encode_source(tokeniser, row["src"], max_len)
        tgt = tokeniser.encode(row["tgt"])
        tgt = tgt[: max_len - 1]
        examples.append((src, tgt))
    return examples


def pair_lengths(examples):
    lengths = []
    for src, tgt in examples:
        lengths.append(max(len(src), len(tgt) + 1))
    return lengths


def make_batches(lengths, max_tokens, rng=None):
    """Group sentences of similar length. No padded batch is larger than `max_tokens`.

    If `rng` is given, the batches are returned in random order.
    """
    def length_of(i):
        return lengths[i]

    order = list(range(len(lengths)))
    if rng is not None:
        rng.shuffle(order)
    order.sort(key=length_of)

    batches = []
    batch = []
    for i in order:
        size_if_added = (len(batch) + 1) * lengths[i]
        if len(batch) > 0 and size_if_added > max_tokens:
            batches.append(batch)
            batch = []
        batch.append(i)
    batches.append(batch)

    if rng is not None:
        rng.shuffle(batches)
    return batches


def pad(sequences, device):
    width = 0
    for seq in sequences:
        width = max(width, len(seq))
    padded = []
    for seq in sequences:
        padded.append(seq + [PAD] * (width - len(seq)))
    return torch.tensor(padded, device=device)


def collate(examples, device):
    sources = []
    source_lengths = []
    decoder_inputs = []
    decoder_outputs = []
    for src, tgt in examples:
        sources.append(src)
        source_lengths.append(len(src))
        decoder_inputs.append([BOS] + tgt)
        decoder_outputs.append(tgt + [EOS])
    return {
        "src": pad(sources, device),
        "src_len": torch.tensor(source_lengths),
        "tgt_in": pad(decoder_inputs, device),
        "tgt_out": pad(decoder_outputs, device),
    }
