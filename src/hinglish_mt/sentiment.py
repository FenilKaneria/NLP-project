"""Sentiment analysis of Hinglish tweets with the help of the translator.

Usage: python -m hinglish_mt.sentiment [--run run_f]

Each tweet of the SentiMix test set is translated to English, and an English
sentiment tool (VADER) then labels the translation as positive, negative or
neutral. This is compared with the same tool on the untranslated tweet.

Writes outputs/results/sentiment_test.tsv (the translations) and
outputs/results/sentiment.md and sentiment.json (the scores).
Run it again with --score-only to repeat the scoring without translating.
"""

import argparse
import json
import random
import re
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from hinglish_mt import DATA_DIR, PROCESSED_DIR
from hinglish_mt.buckets import GROUPS, group_of
from hinglish_mt.clean import fix_text, preprocess, restore
from hinglish_mt.evaluate import RESULTS_DIR
from hinglish_mt.splits import norm_key, read_tsv, write_tsv

SENTIMIX_DIR = DATA_DIR / "SentiMix"
TRAIN_FILE = SENTIMIX_DIR / "train_14k_split_conll.txt"
TEST_FILE = SENTIMIX_DIR / "Hindi_test_unalbelled_conll_updated.txt"
TEST_LABEL_FILE = SENTIMIX_DIR / "test_labels_hinglish.txt"

LABELS = ["negative", "neutral", "positive"]
# Thresholds recommended by the authors of VADER. They are not tuned on our data.
POSITIVE_FROM = 0.05
NEGATIVE_FROM = -0.05
SAMPLES = 1000
EXAMPLES_PER_KIND = 10
EXAMPLE_MAX_LENGTH = 140
FIELDS = ["id", "label", "text", "src", "hyp", "translation"]

# The data has every link split into pieces: "https // t . co / abc".
_SPLIT_LINK_RE = re.compile(r"\bhttps?\s*:?\s*//\s*t\s*\.?\s*co\s*/\s*\S+", re.IGNORECASE)


def join_words(words):
    """Build the tweet from its words. '@' and '#' are joined to the word after them."""
    pieces = []
    prefix = ""
    for word in words:
        if word == "@" or word == "#":
            prefix = word
        else:
            pieces.append(prefix + word)
            prefix = ""
    text = " ".join(pieces)
    text = _SPLIT_LINK_RE.sub("https://t.co/link", text)
    return fix_text(text)


def read_sentimix(path):
    """Read a SentiMix file. Returns rows with id, text and label (empty if the file has none)."""
    rows = []
    words = []
    tweet_id = None
    label = ""
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    for line in lines + [""]:
        parts = line.split("\t")
        if parts[0] == "meta" and len(parts) >= 2:
            tweet_id = parts[1].strip()
            if len(parts) >= 3:
                label = parts[2].strip()
            else:
                label = ""
            words = []
        elif line.strip() == "":
            if tweet_id is not None:
                rows.append({"id": tweet_id, "text": join_words(words), "label": label})
            tweet_id = None
        else:
            words.append(parts[0])
    return rows


def read_test_labels(path):
    labels = {}
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    for line in lines[1:]:
        if line.strip() == "":
            continue
        tweet_id, label = line.strip().split(",")
        labels[tweet_id] = label
    return labels


def load_test_rows():
    rows = read_sentimix(TEST_FILE)
    labels = read_test_labels(TEST_LABEL_FILE)
    for row in rows:
        row["label"] = labels[row["id"]]
    return rows


def translator_training_keys():
    """Keys of all tweets the translator was trained or tuned on."""
    keys = set()
    for name in ["phinc_train.tsv", "phinc_val.tsv"]:
        for row in read_tsv(PROCESSED_DIR / name):
            keys.add(norm_key(row["src"]))
    return keys


def remove_seen(rows, seen_keys):
    """Drop tweets that the translator has seen. Returns the rows kept and the number dropped."""
    kept = []
    dropped = 0
    for row in rows:
        cleaned, originals = preprocess(row["text"])
        if norm_key(cleaned) in seen_keys:
            dropped += 1
        else:
            kept.append(row)
    return kept, dropped


def translate_rows(rows, run):
    # Imported here, so that scoring alone does not need PyTorch.
    from hinglish_mt.decode import load_run, translate

    model, tokeniser, checkpoint = load_run(run)
    sources = []
    all_originals = []
    for row in rows:
        cleaned, originals = preprocess(row["text"])
        sources.append(cleaned)
        all_originals.append(originals)
    outputs = translate(model, tokeniser, sources, beam=4)

    for row, source, output, originals in zip(rows, sources, outputs, all_originals):
        row["src"] = source
        row["hyp"] = output
        row["translation"] = restore(output, originals)
    return rows


def label_of(compound):
    if compound >= POSITIVE_FROM:
        return "positive"
    if compound <= NEGATIVE_FROM:
        return "negative"
    return "neutral"


def vader_labels(texts):
    analyser = SentimentIntensityAnalyzer()
    labels = []
    for text in texts:
        compound = analyser.polarity_scores(text)["compound"]
        labels.append(label_of(compound))
    return labels


def trained_classifier_labels(train_rows, test_texts):
    """Reference system: a classifier trained on the labelled Hinglish tweets themselves."""
    train_texts = []
    train_labels = []
    for row in train_rows:
        if row["label"] in LABELS:
            train_texts.append(row["text"].lower())
            train_labels.append(row["label"])
    vectoriser = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), sublinear_tf=True, min_df=2)
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(vectoriser.fit_transform(train_texts), train_labels)

    lowered = []
    for text in test_texts:
        lowered.append(text.lower())
    return list(classifier.predict(vectoriser.transform(lowered)))


def scores_of(true_labels, predicted):
    per_class = f1_score(true_labels, predicted, labels=LABELS, average=None, zero_division=0)
    result = {
        "accuracy": 100 * accuracy_score(true_labels, predicted),
        "macro_f1": 100 * f1_score(true_labels, predicted, labels=LABELS, average="macro", zero_division=0),
    }
    for label, value in zip(LABELS, per_class):
        result[f"f1_{label}"] = 100 * value
    return result


def paired_bootstrap(true_labels, predicted_1, predicted_2, key, seed=42):
    """Compare two systems on the same tweets, as hinglish_mt.ablation does for translation scores."""
    n = len(true_labels)
    real_difference = scores_of(true_labels, predicted_1)[key] - scores_of(true_labels, predicted_2)[key]

    rng = random.Random(seed)
    differences = []
    wins = 0
    for _ in range(SAMPLES):
        true_sample = []
        sample_1 = []
        sample_2 = []
        for _ in range(n):
            i = rng.randrange(n)
            true_sample.append(true_labels[i])
            sample_1.append(predicted_1[i])
            sample_2.append(predicted_2[i])
        difference = scores_of(true_sample, sample_1)[key] - scores_of(true_sample, sample_2)[key]
        differences.append(difference)
        if difference > 0:
            wins += 1

    differences.sort()
    return {
        "difference": real_difference,
        "low": differences[int(0.025 * SAMPLES)],
        "high": differences[int(0.975 * SAMPLES)],
        "p_value": 1 - wins / SAMPLES,
    }


def confusion(true_labels, predicted):
    table = {}
    for true_label in LABELS:
        table[true_label] = {}
        for predicted_label in LABELS:
            table[true_label][predicted_label] = 0
    for true_label, predicted_label in zip(true_labels, predicted):
        table[true_label][predicted_label] += 1
    return table


def scores_by_group(rows, true_labels, raw_labels, translated_labels):
    """Scores of both VADER systems for each group of tweets (Hindi only, mixed, English only)."""
    members = {}
    for group in GROUPS:
        members[group] = []
    for i, row in enumerate(rows):
        members[group_of(row["src"])].append(i)
        members["All"].append(i)

    result = {}
    for group in GROUPS:
        true_part = []
        raw_part = []
        translated_part = []
        for i in members[group]:
            true_part.append(true_labels[i])
            raw_part.append(raw_labels[i])
            translated_part.append(translated_labels[i])
        result[group] = {
            "tweets": len(members[group]),
            "raw": scores_of(true_part, raw_part),
            "translated": scores_of(true_part, translated_part),
        }
    return result


def changed_examples(rows, raw_labels, translated_labels):
    """Short tweets where translating changed the label: first those corrected, then those made wrong."""
    corrected = []
    made_wrong = []
    for row, raw_label, translated_label in zip(rows, raw_labels, translated_labels):
        if len(row["text"]) > EXAMPLE_MAX_LENGTH or raw_label == translated_label:
            continue
        example = {
            "text": row["text"],
            "translation": row["translation"],
            "true": row["label"],
            "raw": raw_label,
            "translated": translated_label,
        }
        if translated_label == row["label"] and len(corrected) < EXAMPLES_PER_KIND:
            corrected.append(example)
        if raw_label == row["label"] and len(made_wrong) < EXAMPLES_PER_KIND:
            made_wrong.append(example)
    return corrected, made_wrong


def p_value_text(p_value):
    if p_value == 0:
        return "< 0.001"
    return f"{p_value:.3f}"


def table_cell(text):
    return text.replace("|", "/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="run_f")
    parser.add_argument("--score-only", action="store_true", help="use the saved translations")
    args = parser.parse_args()

    translation_path = RESULTS_DIR / "sentiment_test.tsv"
    if args.score_only:
        rows = read_tsv(translation_path)
        dropped = None
    else:
        rows, dropped = remove_seen(load_test_rows(), translator_training_keys())
        print(f"{len(rows)} test tweets, {dropped} removed because the translator has seen them")
        rows = translate_rows(rows, args.run)
        write_tsv(rows, translation_path, FIELDS)

    true_labels = []
    raw_texts = []
    translations = []
    for row in rows:
        true_labels.append(row["label"])
        raw_texts.append(row["text"])
        translations.append(row["translation"])

    label_counts = Counter(true_labels)
    most_frequent = label_counts.most_common(1)[0][0]
    train_rows = read_sentimix(TRAIN_FILE)

    predictions = {
        "majority": [most_frequent] * len(rows),
        "raw": vader_labels(raw_texts),
        "translated": vader_labels(translations),
        "trained": trained_classifier_labels(train_rows, raw_texts),
    }
    titles = [
        ("majority", f"Always \"{most_frequent}\"", "no"),
        ("raw", "VADER on the Hinglish tweet", "no"),
        ("translated", f"VADER on our translation ({args.run})", "no"),
        ("trained", "Classifier trained on labelled Hinglish tweets", f"yes, {len(train_rows):,}"),
    ]

    summary = {
        "run": args.run,
        "tweets": len(rows),
        "removed_as_seen": dropped,
        "label_counts": dict(label_counts),
        "systems": {},
        "comparisons": {},
    }

    lines = []
    lines.append(f"## Sentiment of {len(rows):,} SentiMix test tweets")
    lines.append("")
    lines.append("| System | Hinglish labels used | Accuracy | Macro F1 | F1 negative | F1 neutral | F1 positive |")
    lines.append("|---|---|---|---|---|---|---|")
    for key, title, labels_used in titles:
        scores = scores_of(true_labels, predictions[key])
        summary["systems"][key] = scores
        lines.append(
            f"| {title} | {labels_used} | {scores['accuracy']:.1f} | {scores['macro_f1']:.1f} "
            f"| {scores['f1_negative']:.1f} | {scores['f1_neutral']:.1f} | {scores['f1_positive']:.1f} |"
        )

    lines.append("")
    lines.append(f"## Paired bootstrap, translated minus untranslated, {SAMPLES} samples")
    lines.append("")
    lines.append("| Metric | Difference | 95% interval | p-value |")
    lines.append("|---|---|---|---|")
    for key, title in [("macro_f1", "Macro F1"), ("accuracy", "Accuracy")]:
        test = paired_bootstrap(true_labels, predictions["translated"], predictions["raw"], key)
        summary["comparisons"][key] = test
        lines.append(
            f"| {title} | {test['difference']:+.1f} | {test['low']:+.1f} to {test['high']:+.1f} "
            f"| {p_value_text(test['p_value'])} |"
        )

    groups = scores_by_group(rows, true_labels, predictions["raw"], predictions["translated"])
    summary["groups"] = groups
    lines.append("")
    lines.append("## By how mixed the tweet is (VADER untranslated, translated, difference)")
    lines.append("")
    lines.append("| Group | Tweets | Macro F1 untranslated | Macro F1 translated | Difference | Accuracy untranslated | Accuracy translated | Difference |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for group in GROUPS:
        raw = groups[group]["raw"]
        translated = groups[group]["translated"]
        lines.append(
            f"| {group} | {groups[group]['tweets']} | {raw['macro_f1']:.1f} | {translated['macro_f1']:.1f} "
            f"| {translated['macro_f1'] - raw['macro_f1']:+.1f} | {raw['accuracy']:.1f} | {translated['accuracy']:.1f} "
            f"| {translated['accuracy'] - raw['accuracy']:+.1f} |"
        )

    for key, title in [("raw", "VADER on the Hinglish tweet"), ("translated", "VADER on our translation")]:
        table = confusion(true_labels, predictions[key])
        summary[f"confusion_{key}"] = table
        lines.append("")
        lines.append(f"## {title}: true label (rows) against given label (columns)")
        lines.append("")
        lines.append("| | " + " | ".join(LABELS) + " |")
        lines.append("|---|---|---|---|")
        for true_label in LABELS:
            counts = []
            for predicted_label in LABELS:
                counts.append(str(table[true_label][predicted_label]))
            lines.append(f"| {true_label} | " + " | ".join(counts) + " |")

    corrected, made_wrong = changed_examples(rows, predictions["raw"], predictions["translated"])
    summary["examples_corrected"] = corrected
    summary["examples_made_wrong"] = made_wrong
    for title, examples in [("corrected the label", corrected), ("made the label wrong", made_wrong)]:
        lines.append("")
        lines.append(f"## Examples where translation {title}")
        lines.append("")
        lines.append("| Tweet | Translation | True | Untranslated | Translated |")
        lines.append("|---|---|---|---|---|")
        for example in examples:
            lines.append(
                f"| {table_cell(example['text'])} | {table_cell(example['translation'])} "
                f"| {example['true']} | {example['raw']} | {example['translated']} |"
            )

    text = "\n".join(lines) + "\n"
    (RESULTS_DIR / "sentiment.md").write_text(text, encoding="utf-8")
    (RESULTS_DIR / "sentiment.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
