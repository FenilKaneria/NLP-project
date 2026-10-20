"""Clean PHINC, remove duplicates and write seeded train/val/test splits.

Usage: python -m hinglish_mt.splits
"""

import argparse
import csv
import json
import random
import re
from pathlib import Path

from hinglish_mt import DATA_DIR, PROCESSED_DIR
from hinglish_mt.clean import has_devanagari, preprocess

PHINC_CSV = DATA_DIR / "PHINC" / "English-Hindi code-mixed parallel corpus.csv"
FIELDS = ["id", "src", "tgt", "src_raw", "tgt_raw"]
SPLIT_NAMES = ("train", "val", "test")

_NON_WORD_RE = re.compile(r"[\W_]+")


def norm_key(text):
    """Key for finding duplicates: lowercase letters and digits only."""
    return _NON_WORD_RE.sub(" ", text.lower()).strip()


def load_phinc(path=PHINC_CSV):
    rows = []
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            rows.append({"id": i, "src_raw": row["Sentence"], "tgt_raw": row["English_Translation"]})
    return rows


def clean_rows(rows):
    """Clean both sides, drop unusable rows and rows whose source was already seen."""
    report = {"input": len(rows), "dropped_devanagari": 0, "dropped_empty": 0, "dropped_duplicate": 0}
    seen = set()
    kept = []
    for row in rows:
        if has_devanagari(row["src_raw"]):
            report["dropped_devanagari"] += 1
            continue
        src, _ = preprocess(row["src_raw"])
        tgt, _ = preprocess(row["tgt_raw"])
        key = norm_key(src)
        if key == "" or norm_key(tgt) == "":
            report["dropped_empty"] += 1
            continue
        if key in seen:
            report["dropped_duplicate"] += 1
            continue
        seen.add(key)

        cleaned = dict(row)
        cleaned["src"] = src
        cleaned["tgt"] = tgt
        kept.append(cleaned)

    already_english = 0
    outside_5_40 = 0
    for row in kept:
        if norm_key(row["src"]) == norm_key(row["tgt"]):
            already_english += 1
        n_tokens = len(row["src"].split())
        if n_tokens < 5 or n_tokens > 40:
            outside_5_40 += 1

    report["kept"] = len(kept)
    report["already_english"] = already_english
    report["src_tokens_outside_5_40"] = outside_5_40
    return kept, report


def split_rows(rows, seed=42, fractions=(0.8, 0.1, 0.1)):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n_train = int(len(rows) * fractions[0])
    n_val = int(len(rows) * fractions[1])
    return {
        "train": rows[:n_train],
        "val": rows[n_train : n_train + n_val],
        "test": rows[n_train + n_val :],
    }


def write_tsv(rows, path, fields=FIELDS):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PHINC_CSV)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows, report = clean_rows(load_phinc(args.input))
    splits = split_rows(rows, seed=args.seed)
    for name, part in splits.items():
        write_tsv(part, args.out_dir / f"phinc_{name}.tsv")
        report[name] = len(part)
    report["seed"] = args.seed

    report_text = json.dumps(report, indent=2)
    (args.out_dir / "phinc_split_report.json").write_text(report_text, encoding="utf-8")
    print(report_text)


if __name__ == "__main__":
    main()
