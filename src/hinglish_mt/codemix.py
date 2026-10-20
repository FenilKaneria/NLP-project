"""Make the synthetic Hinglish code-mixed: put English words into the Hindi sentences.

Usage: python -m hinglish_mt.codemix

Step 1 learns a Hindi to English word dictionary from the synthetic pairs
with IBM Model 1 (Brown et al. 1993). Step 2 goes through every pair and
replaces some Hindi words by their English word, but only if that English
word is in the pair's own English sentence. Words that are not replaced get
the same noisy spelling as in synth_noisy.tsv, so the two files differ only
in the mixing.
"""

import argparse
import json
import random
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from hinglish_mt import OUTPUT_DIR, PROCESSED_DIR
from hinglish_mt.clean import preprocess, tokenize
from hinglish_mt.noise import NoiseConfig, noisy_romanise_word
from hinglish_mt.romanise import romanise
from hinglish_mt.splits import read_tsv, write_tsv
from hinglish_mt.synth import detokenise

DICTIONARY_PATH = PROCESSED_DIR / "codemix_dictionary.json"
DICTIONARY_ROWS = 150_000
FREQUENT_WORDS_SKIPPED = 40
MIN_WORD_COUNT = 5
MIN_PROBABILITY = 0.2
ROUNDS = 5
# People mix in English nouns, adjectives and verbs, not words like "can", "when" or "am".
GRAMMAR_WORDS = set(ENGLISH_STOP_WORDS) | {"said", "shall", "does", "did", "having", "like", "used"}

_HINDI_WORD_RE = re.compile(r"[ऀ-ॣॱ-ॿ]+")


@dataclass(frozen=True)
class MixConfig:
    sentence_share: float = 0.7
    word_chance: float = 0.9


def normal_form(word):
    return unicodedata.normalize("NFC", word)


def hindi_words(text):
    words = []
    for word in _HINDI_WORD_RE.findall(text):
        words.append(normal_form(word))
    return words


def english_words(text):
    words = []
    for token in tokenize(text):
        if token.isalpha():
            words.append(token)
    return words


def content_words(sentences):
    """Words that are neither very frequent (grammar words) nor very rare."""
    counts = Counter()
    for sentence in sentences:
        counts.update(sentence)
    kept = set()
    for rank, (word, count) in enumerate(counts.most_common()):
        if rank >= FREQUENT_WORDS_SKIPPED and count >= MIN_WORD_COUNT:
            kept.add(word)
    return kept


def keep_only(sentence, allowed):
    kept = []
    for word in sentence:
        if word in allowed:
            kept.append(word)
    return kept


def translation_table(pairs, rounds=ROUNDS, filter_words=True):
    """IBM Model 1. `pairs` is a list of (Hindi words, English words).

    Returns probability[hindi][english]: how likely the Hindi word is translated by the English word.
    """
    if filter_words:
        hindi_allowed = content_words([hindi for hindi, english in pairs])
        english_allowed = content_words([english for hindi, english in pairs])
        filtered = []
        for hindi, english in pairs:
            hindi = keep_only(hindi, hindi_allowed)
            english = keep_only(english, english_allowed)
            if len(hindi) > 0 and len(english) > 0:
                filtered.append((hindi, english))
        pairs = filtered

    probability = {}
    for round_number in range(rounds):
        # In the first round nothing is known, so every word pair counts the same.
        # Later a pair that was dropped from the table counts as almost impossible.
        if round_number == 0:
            missing = 1.0
        else:
            missing = 1e-6

        counts = {}
        for hindi, english in pairs:
            for english_word in english:
                values = []
                for hindi_word in hindi:
                    values.append(probability.get(hindi_word, {}).get(english_word, missing))
                total = sum(values)
                for hindi_word, value in zip(hindi, values):
                    share = value / total
                    if share < 0.001:
                        continue
                    if hindi_word not in counts:
                        counts[hindi_word] = {}
                    row = counts[hindi_word]
                    row[english_word] = row.get(english_word, 0.0) + share

        probability = {}
        for hindi_word, row in counts.items():
            row_total = sum(row.values())
            new_row = {}
            for english_word, count in row.items():
                value = count / row_total
                if value > 0.001:
                    new_row[english_word] = value
            probability[hindi_word] = new_row
    return probability


def probability_of(pair):
    return pair[1]


def build_dictionary(pairs, rounds=ROUNDS, min_probability=MIN_PROBABILITY, filter_words=True):
    """Returns {Hindi word: its likely English words, most likely first}."""
    probability = translation_table(pairs, rounds, filter_words)
    dictionary = {}
    for hindi_word, row in probability.items():
        likely = []
        for english_word, value in row.items():
            if value >= min_probability and english_word not in GRAMMAR_WORDS:
                likely.append((english_word, value))
        likely.sort(key=probability_of, reverse=True)
        if len(likely) > 0:
            words = []
            for english_word, value in likely:
                words.append(english_word)
            dictionary[hindi_word] = words
    return dictionary


def english_for(hindi_word, english_sentence_words, dictionary):
    """The English word for this Hindi word that the English sentence really uses, or None."""
    for english_word in dictionary.get(normal_form(hindi_word), []):
        if english_word in english_sentence_words:
            return english_word
    return None


def mixed_romanise(text, english_sentence_words, dictionary, noise_rng, mix_rng, noise_config, mix_config):
    """Romanise with noise, and replace some Hindi words by English. Returns the text and the number replaced."""
    mix_this_sentence = mix_rng.random() < mix_config.sentence_share
    replaced = [0]

    def spelling(word):
        noisy = noisy_romanise_word(word, noise_rng, noise_config)
        english = english_for(word, english_sentence_words, dictionary)
        if not mix_this_sentence or english is None:
            return noisy
        if mix_rng.random() < mix_config.word_chance:
            replaced[0] += 1
            return english
        return noisy

    return romanise(text, spelling), replaced[0]


def mix_rows(rows, dictionary, seed, noise_config, mix_config):
    mixed_rows = []
    words_total = 0
    words_replaced = 0
    sentences_mixed = 0
    for row in rows:
        noise_rng = random.Random(f"{seed}:{row['id']}")
        mix_rng = random.Random(f"mix:{seed}:{row['id']}")
        target_words = set(english_words(row["tgt"]))
        text, replaced = mixed_romanise(
            row["src_raw"], target_words, dictionary, noise_rng, mix_rng, noise_config, mix_config
        )
        src, _ = preprocess(detokenise(text))

        mixed_row = dict(row)
        mixed_row["src"] = src
        mixed_rows.append(mixed_row)

        words_total += len(hindi_words(row["src_raw"]))
        words_replaced += replaced
        if replaced > 0:
            sentences_mixed += 1

    report = {
        "sentences": len(rows),
        "sentences_with_english": sentences_mixed / len(rows),
        "hindi_words_replaced": words_replaced / words_total,
    }
    return mixed_rows, report


def write_samples(noisy_rows, mixed_rows, dictionary, usage):
    lines = []
    for noisy, mixed in list(zip(noisy_rows, mixed_rows))[:200]:
        lines.append(f"hindi   {noisy['src_raw']}")
        lines.append(f"noisy   {noisy['src']}")
        lines.append(f"mixed   {mixed['src']}")
        lines.append(f"english {noisy['tgt']}")
        lines.append("")
    (OUTPUT_DIR / "analysis" / "codemix_sample.txt").write_text("\n".join(lines), encoding="utf-8")

    lines = []
    for hindi_word, count in usage.most_common(200):
        lines.append(f"{hindi_word}\t{', '.join(dictionary[hindi_word])}\t{count}")
    (OUTPUT_DIR / "analysis" / "codemix_dictionary_sample.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--word-chance", type=float, default=MixConfig.word_chance)
    args = parser.parse_args()

    noise_config = NoiseConfig()
    mix_config = MixConfig(word_chance=args.word_chance)
    train_rows = read_tsv(PROCESSED_DIR / "synth_noisy.tsv")
    val_rows = read_tsv(PROCESSED_DIR / "synth_noisy_val.tsv")

    pairs = []
    for row in train_rows[:DICTIONARY_ROWS]:
        pairs.append((hindi_words(row["src_raw"]), english_words(row["tgt"])))
    dictionary = build_dictionary(pairs)
    DICTIONARY_PATH.write_text(json.dumps(dictionary, ensure_ascii=False, indent=0), encoding="utf-8")

    mixed_train, report = mix_rows(train_rows, dictionary, args.seed, noise_config, mix_config)
    mixed_val, _ = mix_rows(val_rows, dictionary, args.seed, noise_config, mix_config)
    write_tsv(mixed_train, PROCESSED_DIR / "synth_mixed.tsv")
    write_tsv(mixed_val, PROCESSED_DIR / "synth_mixed_val.tsv")

    # How often each dictionary word could be used, for the sample file.
    usage = Counter()
    for row in train_rows:
        target_words = set(english_words(row["tgt"]))
        for word in hindi_words(row["src_raw"]):
            if english_for(word, target_words, dictionary) is not None:
                usage[word] += 1
    write_samples(train_rows, mixed_train, dictionary, usage)

    report["dictionary_entries"] = len(dictionary)
    report["dictionary_rows"] = DICTIONARY_ROWS
    report["seed"] = args.seed
    report["mix_config"] = asdict(mix_config)
    (PROCESSED_DIR / "codemix_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
