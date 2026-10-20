"""Ablation table and significance tests for runs A to F.

Usage: python -m hinglish_mt.ablation

Reads the test results written by `hinglish_mt.evaluate` and writes
outputs/results/ablation.md and ablation.json.

Significance uses paired bootstrap resampling (Koehn 2004): draw test
sentences with replacement 1,000 times and count how often one system
scores higher than the other on the same draw.
"""

import json
import random

from hinglish_mt.evaluate import BLEU4_METRIC, CHRF_METRIC, RESULTS_DIR
from hinglish_mt.splits import read_tsv

RUNS = [
    ("A", "PHINC only, word", ["run_a"]),
    ("B", "PHINC only, BPE", ["run_b", "run_b_seed43", "run_b_seed44"]),
    ("C", "clean synthetic, then PHINC, BPE", ["run_c", "run_c_seed43", "run_c_seed44"]),
    ("D", "noisy synthetic, then PHINC, BPE", ["run_d", "run_d_seed43", "run_d_seed44"]),
    ("E", "as D, with copy mechanism", ["run_e", "run_e_seed43", "run_e_seed44"]),
    ("F", "as E, synthetic data code-mixed", ["run_f", "run_f_seed43", "run_f_seed44"]),
]
COMPARISONS = [
    ("run_d", "run_b"),
    ("run_d", "run_c"),
    ("run_c", "run_b"),
    ("run_b", "run_a"),
    ("run_e", "run_d"),
    ("run_f", "run_e"),
    ("run_f", "run_d"),
    ("run_f", "run_b"),
]
SAMPLES = 1000


def load_scores(name):
    path = RESULTS_DIR / f"{name}_test.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    return result["masked"]


def load_outputs(name):
    rows = read_tsv(RESULTS_DIR / f"{name}_test.tsv")
    outputs = []
    references = []
    for row in rows:
        outputs.append(row["hyp"])
        references.append(row["tgt"])
    return outputs, references


def sentence_statistics(metric, outputs, references):
    # sacreBLEU's own bootstrap code uses these two internal methods:
    # the first counts matches per sentence, the second turns summed counts into a score.
    return metric._extract_corpus_statistics(outputs, [references])


def score_sample(metric, statistics, sample):
    chosen = []
    for i in sample:
        chosen.append(statistics[i])
    return metric._aggregate_and_compute(chosen).score


def paired_bootstrap(name_1, name_2, metric, seed=42):
    """Compare two systems. Returns the score difference, its 95% interval and the p-value."""
    outputs_1, references = load_outputs(name_1)
    outputs_2, _ = load_outputs(name_2)
    statistics_1 = sentence_statistics(metric, outputs_1, references)
    statistics_2 = sentence_statistics(metric, outputs_2, references)
    n = len(references)

    everything = list(range(n))
    real_difference = score_sample(metric, statistics_1, everything) - score_sample(metric, statistics_2, everything)

    rng = random.Random(seed)
    differences = []
    wins = 0
    for _ in range(SAMPLES):
        sample = []
        for _ in range(n):
            sample.append(rng.randrange(n))
        difference = score_sample(metric, statistics_1, sample) - score_sample(metric, statistics_2, sample)
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


def mean(values):
    return sum(values) / len(values)


def main():
    lines = []
    summary = {"runs": {}, "comparisons": {}}

    lines.append("## Test scores (seed 42)")
    lines.append("")
    lines.append("| Run | Training data | chrF | BLEU-4 | BLEU-1 |")
    lines.append("|---|---|---|---|---|")
    copy = load_scores("copy")
    lines.append(f"| Copy | source unchanged | {copy['chrf']:.2f} | {copy['bleu4']:.2f} | {copy['bleu1']:.2f} |")
    for letter, description, names in RUNS:
        scores = load_scores(names[0])
        lines.append(f"| {letter} | {description} | {scores['chrf']:.2f} | {scores['bleu4']:.2f} | {scores['bleu1']:.2f} |")
        summary["runs"][names[0]] = scores

    lines.append("")
    lines.append("## Three seeds (42, 43, 44)")
    lines.append("")
    lines.append("| Run | chrF mean | chrF lowest | chrF highest | BLEU-4 mean | BLEU-4 lowest | BLEU-4 highest |")
    lines.append("|---|---|---|---|---|---|---|")
    for letter, description, names in RUNS:
        chrf_scores = []
        bleu_scores = []
        for name in names:
            scores = load_scores(name)
            if scores is not None:
                chrf_scores.append(scores["chrf"])
                bleu_scores.append(scores["bleu4"])
                summary["runs"][name] = scores
        if len(chrf_scores) < 2:
            continue
        lines.append(
            f"| {letter} | {mean(chrf_scores):.2f} | {min(chrf_scores):.2f} | {max(chrf_scores):.2f} "
            f"| {mean(bleu_scores):.2f} | {min(bleu_scores):.2f} | {max(bleu_scores):.2f} |"
        )

    lines.append("")
    lines.append(f"## Paired bootstrap, seed 42 outputs, {SAMPLES} samples")
    lines.append("")
    lines.append("| Comparison | Metric | Difference | 95% interval | p-value |")
    lines.append("|---|---|---|---|---|")
    for name_1, name_2 in COMPARISONS:
        for metric_name, metric in [("chrF", CHRF_METRIC), ("BLEU-4", BLEU4_METRIC)]:
            test = paired_bootstrap(name_1, name_2, metric)
            summary["comparisons"][f"{name_1} vs {name_2} {metric_name}"] = test
            if test["p_value"] == 0:
                p_value = "< 0.001"
            else:
                p_value = f"{test['p_value']:.3f}"
            lines.append(
                f"| {name_1} minus {name_2} | {metric_name} | {test['difference']:+.2f} "
                f"| {test['low']:+.2f} to {test['high']:+.2f} | {p_value} |"
            )

    text = "\n".join(lines) + "\n"
    (RESULTS_DIR / "ablation.md").write_text(text, encoding="utf-8")
    (RESULTS_DIR / "ablation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
