"""Build synthetic Hinglish-English pairs from the IIT Bombay English-Hindi corpus.

Usage: python -m hinglish_mt.synth

Filters the corpus, samples pairs with a fixed seed, and romanises the Hindi
side twice: once clean (run C) and once with spelling noise (run D). Both
files hold the same pairs in the same order, so the two runs differ only in
the noise. A small validation part, separate from the training sample, is
written for choosing the best model during pre-training.
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from hinglish_mt import DATA_DIR, OUTPUT_DIR, PROCESSED_DIR
from hinglish_mt.clean import mask, preprocess, tokenize
from hinglish_mt.langid import HI, OTHER, tag
from hinglish_mt.noise import NoiseConfig, noisy_romanise
from hinglish_mt.romanise import romanise
from hinglish_mt.splits import read_tsv, write_tsv

IITB_DIR = DATA_DIR / "IIT bombay data" / "parallel-n"
MIN_TOKENS = 5
MAX_TOKENS = 40
MAX_LENGTH_RATIO = 2.0
MIN_DEVANAGARI_SHARE = 0.9
MIN_LATIN_SHARE = 0.9

_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_LATIN_RE = re.compile(r"[a-zA-Z]")
_LETTER_RE = re.compile(r"[^\W\d_]")
# Text from software menus and web pages: shortcut keys, format codes, tags, HTML leftovers.
_MARKUP_RE = re.compile(r"_|%[a-z0-9]|[<>{}\[\]|\\=$]|&\w|\w&|#\d+;")
_SPACE_RE = re.compile(r"\s+")
# The corpus often has a space before endings and punctuation ("writer 's fancy .", "do n't").
_DETACHED_RE = re.compile(r"\s+(?=(?:'(?:s|re|ve|ll|d|m|t)|n't)\b|[,.;:!?](?:\s|$))")


def placeholder_counts(text):
    """How many mentions, hashtags and links the text has."""
    masked, originals = mask(text)
    counts = Counter()
    for kind, original in originals:
        counts[kind] += 1
    return counts


def reject_reason(en, hi):
    """Why a pair is unusable, or None if it passes. Duplicates are handled by the caller."""
    n_en = len(en.split())
    n_hi = len(hi.split())
    if n_en < MIN_TOKENS or n_en > MAX_TOKENS or n_hi < MIN_TOKENS or n_hi > MAX_TOKENS:
        return "length"
    if max(n_en, n_hi) > MAX_LENGTH_RATIO * min(n_en, n_hi):
        return "length_ratio"
    if _MARKUP_RE.search(en) or _MARKUP_RE.search(hi):
        return "markup"

    devanagari = len(_DEVANAGARI_RE.findall(hi))
    latin = len(_LATIN_RE.findall(hi))
    if devanagari < MIN_DEVANAGARI_SHARE * (devanagari + latin):
        return "hindi_script"

    letters = len(_LETTER_RE.findall(en))
    latin_letters = len(_LATIN_RE.findall(en))
    if _DEVANAGARI_RE.search(en) or latin_letters < MIN_LATIN_SHARE * max(letters, 1):
        return "english_script"

    if placeholder_counts(en) != placeholder_counts(hi):
        return "mask_mismatch"
    return None


def detokenise(text):
    """Join endings and punctuation that the corpus split off with a space."""
    return _DETACHED_RE.sub("", text)


def filter_pairs(pairs):
    """Keep usable (line number, English, Hindi) triples, one per distinct Hindi sentence."""
    report = Counter()
    seen = set()
    kept = []
    for i, (en, hi) in enumerate(pairs):
        report["input"] += 1
        en = en.strip()
        hi = hi.strip()

        reason = reject_reason(en, hi)
        if reason is None:
            key = _SPACE_RE.sub(" ", hi)
            if key in seen:
                reason = "duplicate"
            else:
                seen.add(key)

        if reason is None:
            kept.append((i, en, hi))
        else:
            report[f"dropped_{reason}"] += 1
    report["kept"] = len(kept)
    return kept, dict(report)


def build_rows(triples, seed, config):
    """Romanise each pair clean and noisy. The noise for a pair depends only on the seed and its id."""
    clean_rows = []
    noisy_rows = []
    for i, en, hi in triples:
        tgt, _ = preprocess(detokenise(en))

        clean_src, _ = preprocess(detokenise(romanise(hi)))
        clean_rows.append({"id": i, "tgt": tgt, "src_raw": hi, "tgt_raw": en, "src": clean_src})

        rng = random.Random(f"{seed}:{i}")
        noisy_src, _ = preprocess(detokenise(noisy_romanise(hi, rng, config)))
        noisy_rows.append({"id": i, "tgt": tgt, "src_raw": hi, "tgt_raw": en, "src": noisy_src})
    return clean_rows, noisy_rows


def changed_word_share(clean_rows, noisy_rows):
    """Share of source words whose spelling the noise changed."""
    changed = 0
    total = 0
    for clean, noisy in zip(clean_rows, noisy_rows):
        clean_words = clean["src"].split()
        noisy_words = noisy["src"].split()
        if len(clean_words) != len(noisy_words):
            continue
        total += len(clean_words)
        for clean_word, noisy_word in zip(clean_words, noisy_words):
            if clean_word != noisy_word:
                changed += 1
    return changed / total


def phinc_coverage(phinc_rows, synth_rows):
    """How much of PHINC's source vocabulary also occurs in the synthetic source.

    Given for all words and for Hindi words only. The second is what the
    romaniser and the noise can influence.
    """
    vocabulary = set()
    for row in synth_rows:
        vocabulary.update(tokenize(row["src"]))

    all_words = Counter()
    hindi_words = Counter()
    for row in phinc_rows:
        tokens = tokenize(row["src"])
        for token, label in zip(tokens, tag(tokens)):
            if label != OTHER:
                all_words[token] += 1
            if label == HI:
                hindi_words[token] += 1

    result = {"synthetic_vocabulary": len(vocabulary)}
    for name, counts in [("all_words", all_words), ("hindi_words", hindi_words)]:
        covered_types = 0
        covered_tokens = 0
        for word, count in counts.items():
            if word in vocabulary:
                covered_types += 1
                covered_tokens += count
        result[f"phinc_{name}_types_covered"] = covered_types / len(counts)
        result[f"phinc_{name}_tokens_covered"] = covered_tokens / sum(counts.values())
    return result


def write_sample(clean_rows, noisy_rows, path, n=200):
    """Write the first `n` pairs as text, to read the clean and noisy spelling side by side."""
    lines = []
    for clean, noisy in zip(clean_rows[:n], noisy_rows[:n]):
        lines.append(f"hindi   {clean['src_raw']}")
        lines.append(f"clean   {clean['src']}")
        lines.append(f"noisy   {noisy['src']}")
        lines.append(f"english {clean['tgt']}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iitb-dir", type=Path, default=IITB_DIR)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--train-size", type=int, default=300_000)
    parser.add_argument("--val-size", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    en_path = args.iitb_dir / "IITB.en-hi.en"
    hi_path = args.iitb_dir / "IITB.en-hi.hi"
    with open(en_path, encoding="utf-8") as en_file, open(hi_path, encoding="utf-8") as hi_file:
        kept, report = filter_pairs(zip(en_file, hi_file))

    config = NoiseConfig()
    sample = random.Random(args.seed).sample(kept, args.train_size + args.val_size)
    clean_rows, noisy_rows = build_rows(sample, args.seed, config)

    clean_train = clean_rows[: args.train_size]
    noisy_train = noisy_rows[: args.train_size]
    write_tsv(clean_train, args.out_dir / "synth_clean.tsv")
    write_tsv(clean_rows[args.train_size :], args.out_dir / "synth_clean_val.tsv")
    write_tsv(noisy_train, args.out_dir / "synth_noisy.tsv")
    write_tsv(noisy_rows[args.train_size :], args.out_dir / "synth_noisy_val.tsv")
    write_sample(clean_rows, noisy_rows, OUTPUT_DIR / "analysis" / "synth_sample.txt")

    phinc_train = read_tsv(args.out_dir / "phinc_train.tsv")
    report["seed"] = args.seed
    report["train"] = args.train_size
    report["val"] = args.val_size
    report["noise_config"] = vars(config)
    report["words_changed_by_noise"] = changed_word_share(clean_rows, noisy_rows)
    report["phinc_train_coverage"] = {
        "clean": phinc_coverage(phinc_train, clean_train),
        "noisy": phinc_coverage(phinc_train, noisy_train),
    }

    report_text = json.dumps(report, indent=2)
    (args.out_dir / "synth_report.json").write_text(report_text, encoding="utf-8")
    print(report_text)


if __name__ == "__main__":
    main()
