"""Corpus statistics for a parallel Hinglish-English file set.

Usage:
    python -m hinglish_mt.analysis                      # PHINC train+val+test
    python -m hinglish_mt.analysis --name synth --input data/processed/synth_noisy.tsv

Writes outputs/analysis/<name>_stats.json and <name>_summary.md.
"""

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path

from hinglish_mt import OUTPUT_DIR, PROCESSED_DIR
from hinglish_mt.clean import tokenize
from hinglish_mt.cmi import cmi, corpus_cmi
from hinglish_mt.langid import EN, HI, OTHER, is_ambiguous, is_english_word, tag
from hinglish_mt.splits import SPLIT_NAMES, read_tsv

# (lowest, highest) sentence length of each histogram bin. None means no upper limit.
LENGTH_BINS = [(1, 4), (5, 9), (10, 14), (15, 19), (20, 29), (30, 40), (41, None)]
TOP_N = 25

_VOWEL_RE = re.compile(r"[aeiou]")
_REPEAT_RE = re.compile(r"(.)\1+")


def skeleton(word):
    """Loose spelling key: first letter, then the consonants with repeats removed.

    "nahi", "nhi", "nahii" -> "nh". Unrelated words can share a key ("bhi",
    "bhai"), so this key gives an upper bound on spelling variation.
    """
    word = _REPEAT_RE.sub(r"\1", word)
    first_letter = word[0]
    rest = _VOWEL_RE.sub("", word[1:])
    return first_letter + rest


def strict_key(word):
    """Strict spelling key: only stretching, long vowels and a final n or h are ignored.

    "nahi", "nahii", "nahin", "naahi" -> "nahi". Misses dropped vowels ("nhi"),
    so this key gives a lower bound.
    """
    word = word.replace("ee", "i").replace("oo", "u")
    word = _REPEAT_RE.sub(r"\1", word)
    if len(word) > 2 and word[-1] in "nh" and word[-2] in "aeiou":
        word = word[:-1]
    return word


def count_words(token_lists, tag_lists, wanted_tags):
    """Count every token whose language tag is one of `wanted_tags`."""
    counts = Counter()
    for tokens, tags in zip(token_lists, tag_lists):
        for token, label in zip(tokens, tags):
            if label in wanted_tags:
                counts[token] += 1
    return counts


def _length_stats(lengths):
    histogram = {}
    for low, high in LENGTH_BINS:
        if high is None:
            label = f"{low}+"
        else:
            label = f"{low}-{high}"
        count = 0
        for n in lengths:
            if n >= low and (high is None or n <= high):
                count += 1
        histogram[label] = count
    return {
        "mean": statistics.fmean(lengths),
        "median": statistics.median(lengths),
        "max": max(lengths),
        "histogram": histogram,
    }


def side_stats(token_lists, tag_lists):
    """Counts for one side of the corpus. Word statistics skip tokens tagged other."""
    unigrams = Counter()
    bigrams = Counter()
    lengths = []
    for tokens, tags in zip(token_lists, tag_lists):
        lengths.append(len(tokens))
        words = []
        for token, label in zip(tokens, tags):
            if label != OTHER:
                words.append(token)
        unigrams.update(words)
        bigrams.update(zip(words, words[1:]))

    n_words = sum(unigrams.values())
    seen_once = 0
    for count in unigrams.values():
        if count == 1:
            seen_once += 1

    top_bigrams = []
    for pair, count in bigrams.most_common(TOP_N):
        top_bigrams.append((" ".join(pair), count))

    return {
        "sentences": len(token_lists),
        "tokens": sum(lengths),
        "word_tokens": n_words,
        "vocabulary": len(unigrams),
        "type_token_ratio": len(unigrams) / n_words,
        "singleton_type_share": seen_once / len(unigrams),
        "length": _length_stats(lengths),
        "top_unigrams": unigrams.most_common(TOP_N),
        "top_bigrams": top_bigrams,
    }


def oov_stats(token_lists, tag_lists):
    """Share of words that the English dictionary does not know."""
    counts = count_words(token_lists, tag_lists, [HI, EN])
    unknown_tokens = 0
    unknown_types = 0
    for word, count in counts.items():
        if not is_english_word(word):
            unknown_tokens += count
            unknown_types += 1
    return {
        "token_rate": unknown_tokens / sum(counts.values()),
        "type_rate": unknown_types / len(counts),
    }


def group_size(item):
    key, members = item
    return sum(members.values())


def _group_by(counts, key_function, top):
    """Group words by a spelling key and describe the groups with more than one spelling."""
    groups = {}
    for word, count in counts.items():
        key = key_function(word)
        if key not in groups:
            groups[key] = Counter()
        groups[key][word] = count

    with_variants = {}
    for key, members in groups.items():
        if len(members) > 1 and len(key) > 1:
            with_variants[key] = members

    n_types = 0
    n_tokens = 0
    group_sizes = []
    for members in with_variants.values():
        n_types += len(members)
        n_tokens += sum(members.values())
        group_sizes.append(len(members))

    ranked = sorted(with_variants.items(), key=group_size, reverse=True)
    top_groups = []
    for key, members in ranked[:top]:
        top_groups.append({"key": key, "variants": members.most_common(8)})

    return {
        "groups_with_variants": len(with_variants),
        "types_in_variant_groups_share": n_types / len(counts),
        "tokens_in_variant_groups_share": n_tokens / sum(counts.values()),
        "mean_variants_per_group": statistics.fmean(group_sizes),
        "top_groups": top_groups,
    }


def variant_groups(token_lists, tag_lists, top=20):
    """Group Hindi words by the strict and the loose spelling key (lower and upper bound)."""
    counts = count_words(token_lists, tag_lists, [HI])
    return {
        "hindi_types": len(counts),
        "strict": _group_by(counts, strict_key, top),
        "loose": _group_by(counts, skeleton, top),
    }


def unseen_stats(train_tokens, heldout_tokens):
    """How much of the held-out source a word list built on train would miss."""
    vocabulary = set()
    for tokens in train_tokens:
        vocabulary.update(tokens)

    counts = Counter()
    sentences_with_unseen = 0
    for tokens in heldout_tokens:
        counts.update(tokens)
        has_unseen = False
        for token in tokens:
            if token not in vocabulary:
                has_unseen = True
        if has_unseen:
            sentences_with_unseen += 1

    unseen_tokens = 0
    unseen_types = 0
    for token, count in counts.items():
        if token not in vocabulary:
            unseen_tokens += count
            unseen_types += 1

    return {
        "token_rate": unseen_tokens / sum(counts.values()),
        "type_rate": unseen_types / len(counts),
        "sentences_with_unseen_share": sentences_with_unseen / len(heldout_tokens),
    }


def cmi_histogram(scores):
    histogram = {"0": 0, "0-10": 0, "10-20": 0, "20-30": 0, "30-40": 0, "40-50": 0}
    for score in scores:
        if score == 0:
            histogram["0"] += 1
        elif score <= 10:
            histogram["0-10"] += 1
        elif score <= 20:
            histogram["10-20"] += 1
        elif score <= 30:
            histogram["20-30"] += 1
        elif score <= 40:
            histogram["30-40"] += 1
        else:
            histogram["40-50"] += 1
    return histogram


def analyse(rows):
    src_tokens = []
    tgt_tokens = []
    src_tags = []
    tgt_tags = []
    for row in rows:
        source = tokenize(row["src"])
        target = tokenize(row["tgt"])
        src_tokens.append(source)
        tgt_tokens.append(target)
        src_tags.append(tag(source))

        # The target side is English, so only words and non-words are told apart.
        target_tags = []
        for token in target:
            if token[0].isalpha():
                target_tags.append(EN)
            else:
                target_tags.append(OTHER)
        tgt_tags.append(target_tags)

    word_count = 0
    ambiguous = 0
    scores = []
    for tokens, tags in zip(src_tokens, src_tags):
        word_count += len(tags) - tags.count(OTHER)
        for token in tokens:
            if is_ambiguous(token):
                ambiguous += 1
        scores.append(cmi(tags))

    return {
        "source": side_stats(src_tokens, src_tags),
        "target": side_stats(tgt_tokens, tgt_tags),
        "source_oov_vs_english_dictionary": oov_stats(src_tokens, src_tags),
        "language_tags": corpus_cmi(src_tags),
        "cmi_histogram": cmi_histogram(scores),
        "ambiguous_word_share": ambiguous / word_count,
        "spelling_variants": variant_groups(src_tokens, src_tags),
    }


def _table(headers, rows):
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "---|" * len(headers))
    for row in rows:
        cells = []
        for cell in row:
            cells.append(str(cell))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _pct(value):
    return f"{100 * value:.1f}%"


def tag_shares(language_tags):
    """Hindi / English / other token shares as one string."""
    shares = []
    for label in (HI, EN, OTHER):
        shares.append(_pct(language_tags["tag_share"][label]))
    return " / ".join(shares)


def word_and_count(pair):
    word, count = pair
    return f"{word} ({count})"


def render_summary(name, stats):
    src = stats["source"]
    tgt = stats["target"]
    lang = stats["language_tags"]
    variants = stats["spelling_variants"]
    oov = stats["source_oov_vs_english_dictionary"]

    lines = [f"# Corpus analysis: {name}", ""]

    size_rows = [
        ["Sentences", f"{src['sentences']:,}", f"{tgt['sentences']:,}"],
        ["Tokens", f"{src['tokens']:,}", f"{tgt['tokens']:,}"],
        ["Word tokens", f"{src['word_tokens']:,}", f"{tgt['word_tokens']:,}"],
        ["Vocabulary (word types)", f"{src['vocabulary']:,}", f"{tgt['vocabulary']:,}"],
        ["Types seen once", _pct(src["singleton_type_share"]), _pct(tgt["singleton_type_share"])],
        ["Mean length (tokens)", f"{src['length']['mean']:.1f}", f"{tgt['length']['mean']:.1f}"],
        ["Median length", src["length"]["median"], tgt["length"]["median"]],
    ]
    lines.append("## Size")
    lines.append("")
    lines.append(_table(["", "Source (Hinglish)", "Target (English)"], size_rows))
    lines.append("")

    length_rows = []
    for label in src["length"]["histogram"]:
        length_rows.append([label, src["length"]["histogram"][label], tgt["length"]["histogram"][label]])
    lines.append("## Sentence length (tokens)")
    lines.append("")
    lines.append(_table(["Bin", "Source", "Target"], length_rows))
    lines.append("")

    dominant = []
    for key in (HI, EN, "tie"):
        dominant.append(_pct(lang["dominant_language_share"][key]))
    language_rows = [
        ["Hindi / English / other tokens", tag_shares(lang)],
        ["Sentences with both languages", _pct(lang["mixed_share"])],
        ["Mean CMI, all sentences", f"{lang['mean_all']:.1f}"],
        ["Mean CMI, mixed sentences only", f"{lang['mean_mixed']:.1f}"],
        ["CMI on pooled corpus counts", f"{lang['pooled']:.1f}"],
        ["Mean switch points per sentence", f"{lang['mean_switch_points']:.2f}"],
        ["Dominant language: Hindi / English / tie", " / ".join(dominant)],
        ["Words that are valid in both languages", _pct(stats["ambiguous_word_share"])],
        [
            "Words not in English dictionary (tokens / types)",
            f"{_pct(oov['token_rate'])} / {_pct(oov['type_rate'])}",
        ],
    ]
    histogram_parts = []
    for label, count in stats["cmi_histogram"].items():
        histogram_parts.append(f"{label}: {count}")
    lines.append("## Language tags and code-mixing (source)")
    lines.append("")
    lines.append(_table(["Measure", "Value"], language_rows))
    lines.append("")
    lines.append("Sentence CMI distribution: " + ", ".join(histogram_parts))
    lines.append("")

    if "unseen_in_train" in stats:
        unseen_rows = []
        for split, unseen in stats["unseen_in_train"].items():
            unseen_rows.append([
                split,
                _pct(unseen["token_rate"]),
                _pct(unseen["type_rate"]),
                _pct(unseen["sentences_with_unseen_share"]),
            ])
        lines.append("## Source words unseen in train (word-level vocabulary)")
        lines.append("")
        lines.append(_table(["Split", "Tokens", "Types", "Sentences affected"], unseen_rows))
        lines.append("")

    variant_rows = []
    for key, label in [("strict", "Strict (lower bound)"), ("loose", "Loose (upper bound)")]:
        variant_rows.append([
            label,
            f"{variants[key]['groups_with_variants']:,}",
            f"{variants[key]['mean_variants_per_group']:.1f}",
            _pct(variants[key]["types_in_variant_groups_share"]),
            _pct(variants[key]["tokens_in_variant_groups_share"]),
        ])
    group_rows = []
    for group in variants["strict"]["top_groups"]:
        spellings = []
        for pair in group["variants"]:
            spellings.append(word_and_count(pair))
        group_rows.append([group["key"], ", ".join(spellings)])
    lines.append("## Spelling variation (Hindi-tagged words)")
    lines.append("")
    lines.append(
        "Two keys bracket the true amount. Strict: spellings differ only by stretched letters, long vowels "
        "or a final n/h (misses vowel drops such as nhi). Loose: same first letter and consonants "
        "(also merges unrelated words such as bhi and bhai)."
    )
    lines.append("")
    lines.append(_table(
        ["Key", "Groups with >1 spelling", "Spellings per group", "Hindi types covered", "Hindi tokens covered"],
        variant_rows,
    ))
    lines.append("")
    lines.append("Largest groups under the strict key:")
    lines.append("")
    lines.append(_table(["Key", "Spellings (count)"], group_rows))
    lines.append("")

    frequent_rows = []
    columns = zip(src["top_unigrams"], src["top_bigrams"], tgt["top_unigrams"], tgt["top_bigrams"])
    for src_unigram, src_bigram, tgt_unigram, tgt_bigram in columns:
        frequent_rows.append([
            word_and_count(src_unigram),
            word_and_count(src_bigram),
            word_and_count(tgt_unigram),
            word_and_count(tgt_bigram),
        ])
    lines.append("## Most frequent words and bigrams")
    lines.append("")
    lines.append(_table(
        ["Source unigram", "Source bigram", "Target unigram", "Target bigram"],
        frequent_rows,
    ))
    lines.append("")
    return "\n".join(lines)


def comparison_column(stats):
    """The (measure, value) pairs of one corpus for the side-by-side table."""
    src = stats["source"]
    tgt = stats["target"]
    lang = stats["language_tags"]
    oov = stats["source_oov_vs_english_dictionary"]
    variants = stats["spelling_variants"]
    return [
        ("Sentences", f"{src['sentences']:,}"),
        ("Source word types", f"{src['vocabulary']:,}"),
        ("Source types seen once", _pct(src["singleton_type_share"])),
        ("Mean source length (tokens)", f"{src['length']['mean']:.1f}"),
        ("Mean target length (tokens)", f"{tgt['length']['mean']:.1f}"),
        ("Target word types", f"{tgt['vocabulary']:,}"),
        ("Hindi / English / other tokens", tag_shares(lang)),
        ("Sentences with both languages", _pct(lang["mixed_share"])),
        ("Mean CMI, all sentences", f"{lang['mean_all']:.1f}"),
        ("Mean switch points per sentence", f"{lang['mean_switch_points']:.2f}"),
        ("Words not in English dictionary (tokens)", _pct(oov["token_rate"])),
        (
            "Hindi tokens with >1 spelling, strict key",
            _pct(variants["strict"]["tokens_in_variant_groups_share"]),
        ),
        (
            "Hindi tokens with >1 spelling, loose key",
            _pct(variants["loose"]["tokens_in_variant_groups_share"]),
        ),
    ]


def render_comparison(stats_by_name):
    """One column per corpus, for comparing real and synthetic text."""
    names = list(stats_by_name)
    columns = []
    for name in names:
        columns.append(comparison_column(stats_by_name[name]))

    rows = []
    for i in range(len(columns[0])):
        measure = columns[0][i][0]
        row = [measure]
        for column in columns:
            row.append(column[i][1])
        rows.append(row)
    return _table(["Measure"] + names, rows) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="phinc")
    parser.add_argument("--input", type=Path, nargs="+", help="TSV files with src and tgt columns")
    parser.add_argument("--limit", type=int, help="use only the first N rows, to compare corpora at equal size")
    parser.add_argument("--compare", nargs="+", metavar="NAME", help="write a side-by-side table of saved stats")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR / "analysis")
    args = parser.parse_args()

    if args.compare:
        stats_by_name = {}
        for name in args.compare:
            stats_path = args.out_dir / f"{name}_stats.json"
            stats_by_name[name] = json.loads(stats_path.read_text(encoding="utf-8"))
        table = render_comparison(stats_by_name)
        table_path = args.out_dir / f"compare_{'_'.join(args.compare)}.md"
        table_path.write_text(table, encoding="utf-8")
        print(table)
        return

    if args.input:
        rows = []
        for path in args.input:
            rows.extend(read_tsv(path))
        if args.limit is not None:
            rows = rows[: args.limit]
        stats = analyse(rows)
    else:
        rows = []
        tokens = {}
        for name in SPLIT_NAMES:
            split_rows = read_tsv(PROCESSED_DIR / f"phinc_{name}.tsv")
            rows.extend(split_rows)
            tokens[name] = []
            for row in split_rows:
                tokens[name].append(tokenize(row["src"]))
        stats = analyse(rows)
        stats["unseen_in_train"] = {
            "val": unseen_stats(tokens["train"], tokens["val"]),
            "test": unseen_stats(tokens["train"], tokens["test"]),
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats_text = json.dumps(stats, indent=2, ensure_ascii=False)
    (args.out_dir / f"{args.name}_stats.json").write_text(stats_text, encoding="utf-8")
    summary = render_summary(args.name, stats)
    (args.out_dir / f"{args.name}_summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
