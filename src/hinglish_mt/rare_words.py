"""How well a run handles rare words and names.

Usage: python -m hinglish_mt.rare_words run_b run_d

Two tables are written to outputs/results/rare_words.md and rare_words.json.
* Copy rate: a word that is the same in the tweet and in the reference should
  be kept. How often is it in the output?
* Reference word recall: how often is a word of the reference in the output?
Both are grouped by how often the word occurs in the PHINC training part.
"""

import argparse
import json
from collections import Counter

from hinglish_mt import PROCESSED_DIR
from hinglish_mt.clean import tokenize
from hinglish_mt.evaluate import RESULTS_DIR
from hinglish_mt.splits import read_tsv

GROUPS = ["never", "1 to 5", "6 to 50", "more than 50"]


def group_of(count):
    if count == 0:
        return "never"
    if count <= 5:
        return "1 to 5"
    if count <= 50:
        return "6 to 50"
    return "more than 50"


def words_of(text):
    words = set()
    for token in tokenize(text):
        if token.isalpha():
            words.add(token)
    return words


def training_counts():
    """How often each word occurs in the training part, on either side and on the English side."""
    both_sides = Counter()
    english_side = Counter()
    for row in read_tsv(PROCESSED_DIR / "phinc_train.tsv"):
        both_sides.update(tokenize(row["src"]))
        both_sides.update(tokenize(row["tgt"]))
        english_side.update(tokenize(row["tgt"]))
    return both_sides, english_side


def empty_table():
    table = {}
    for group in GROUPS:
        table[group] = {"words": 0, "found": 0}
    return table


def add_shares(table):
    for group in GROUPS:
        cell = table[group]
        if cell["words"] > 0:
            cell["share"] = cell["found"] / cell["words"]
        else:
            cell["share"] = 0.0


def measure(rows, both_sides, english_side):
    copy = empty_table()
    recall = empty_table()
    for row in rows:
        source_words = words_of(row["src"])
        reference_words = words_of(row["tgt"])
        output_words = words_of(row["hyp"])

        for word in reference_words:
            found = word in output_words
            group = group_of(english_side[word])
            recall[group]["words"] += 1
            if found:
                recall[group]["found"] += 1

            if word in source_words:
                group = group_of(both_sides[word])
                copy[group]["words"] += 1
                if found:
                    copy[group]["found"] += 1

    add_shares(copy)
    add_shares(recall)
    return {"copy": copy, "recall": recall}


def table_lines(title, key, results):
    names = list(results)
    lines = [f"## {title}", ""]
    lines.append("| Times in training | Words | " + " | ".join(names) + " |")
    lines.append("|---|---|" + "---|" * len(names))
    for group in GROUPS:
        cells = []
        for name in names:
            share = results[name][key][group]["share"]
            cells.append(f"{100 * share:.1f}%")
        words = results[names[0]][key][group]["words"]
        lines.append(f"| {group} | {words} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()

    both_sides, english_side = training_counts()
    results = {}
    for name in args.runs:
        rows = read_tsv(RESULTS_DIR / f"{name}_test.tsv")
        results[name] = measure(rows, both_sides, english_side)

    lines = []
    lines += table_lines("Copy rate: words that are the same in the tweet and the reference", "copy", results)
    lines += table_lines("Reference word recall", "recall", results)
    text = "\n".join(lines)

    (RESULTS_DIR / "rare_words.md").write_text(text, encoding="utf-8")
    (RESULTS_DIR / "rare_words.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
