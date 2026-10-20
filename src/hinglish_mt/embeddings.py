"""Nearest neighbours of common words in a model's embedding table.

Usage: python -m hinglish_mt.embeddings [run_d run_b]

Hinglish and English share one embedding table. If the model has learned
meaning, a Hinglish word should sit close to its other spellings and to its
English translation. Writes outputs/analysis/embedding_neighbours.md.
"""

import sys

import torch.nn.functional as F

from hinglish_mt import OUTPUT_DIR
from hinglish_mt.decode import load_run

WORDS = ["nahi", "bahut", "kyu", "accha", "ghar", "paise", "sarkar", "log", "love", "good"]
NEIGHBOURS = 8


def nearest(model, tokeniser, word):
    """Returns the closest word pieces to `word` by cosine similarity, or None if it is not one piece."""
    piece_id = tokeniser.sp.piece_to_id("▁" + word)
    if tokeniser.sp.id_to_piece(piece_id) != "▁" + word:
        return None

    table = model.embedding.weight
    similarities = F.cosine_similarity(table[piece_id].unsqueeze(0), table, dim=-1)
    best = similarities.topk(NEIGHBOURS + 1)

    neighbours = []
    for similarity, other_id in zip(best.values.tolist(), best.indices.tolist()):
        if other_id == piece_id:
            continue
        piece = tokeniser.sp.id_to_piece(other_id)
        neighbours.append(f"{piece} ({similarity:.2f})")
    return neighbours[:NEIGHBOURS]


def main():
    runs = sys.argv[1:]
    if len(runs) == 0:
        runs = ["run_d", "run_b"]

    lines = ["`▁` marks the start of a word. The number is the cosine similarity.", ""]
    for run in runs:
        model, tokeniser, _ = load_run(run)
        lines.append(f"## {run}")
        lines.append("")
        lines.append("| Word | Nearest neighbours |")
        lines.append("|---|---|")
        for word in WORDS:
            neighbours = nearest(model, tokeniser, word)
            if neighbours is None:
                lines.append(f"| {word} | not a single word piece |")
            else:
                lines.append(f"| {word} | {', '.join(neighbours)} |")
        lines.append("")

    text = "\n".join(lines)
    path = OUTPUT_DIR / "analysis" / "embedding_neighbours.md"
    path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
