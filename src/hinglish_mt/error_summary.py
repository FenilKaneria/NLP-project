"""Count the answers in the filled-in error-analysis sheets.

Usage: python -m hinglish_mt.error_summary

Reads compare_b_d_annotated.csv and check_run_d_annotated.csv from
outputs/error_analysis/ and writes outputs/results/error_analysis.md and .json.
"""

import csv
import json
from collections import Counter
from math import comb

from hinglish_mt.error_sheets import SHEET_DIR
from hinglish_mt.evaluate import RESULTS_DIR

BETTER_VALUES = ["D", "B", "same"]
ERROR_VALUES = [
    "none",
    "wrong meaning",
    "wrong word",
    "missing",
    "grammar",
    "untranslated",
    "repetition",
    "bad reference",
]
VERDICT_VALUES = ["correct", "needs change"]


def read_sheet(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def count_column(rows, column, allowed):
    """Count the answers in one column. An answer outside the allowed list is an error."""
    counts = Counter()
    for row in rows:
        value = row[column].strip()
        if value not in allowed:
            raise ValueError(f"row {row['id']}: unknown value '{value}' in column {column}")
        counts[value] += 1
    return counts


def sign_test(wins, losses):
    """Two-sided sign test: chance of a split at least this uneven if both systems were equally good."""
    n = wins + losses
    larger = max(wins, losses)
    tail = 0
    for k in range(larger, n + 1):
        tail += comb(n, k)
    return min(1.0, 2 * tail / 2**n)


def share(count, total):
    return 100 * count / total


def main():
    compare_rows = read_sheet(SHEET_DIR / "compare_b_d_annotated.csv")
    check_rows = read_sheet(SHEET_DIR / "check_run_d_annotated.csv")

    better = count_column(compare_rows, "better", BETTER_VALUES)
    errors = count_column(compare_rows, "error_run_d", ERROR_VALUES)
    verdicts = count_column(check_rows, "verdict", VERDICT_VALUES)
    p_value = sign_test(better["D"], better["B"])

    lines = []
    lines.append(f"## Which translation is better ({len(compare_rows)} test tweets)")
    lines.append("")
    lines.append("| Answer | Tweets | Share |")
    lines.append("|---|---|---|")
    titles = {"D": "Run D is better", "B": "Run B is better", "same": "Equally good or equally bad"}
    for value in BETTER_VALUES:
        lines.append(f"| {titles[value]} | {better[value]} | {share(better[value], len(compare_rows)):.0f}% |")
    lines.append("")
    lines.append(f"Sign test on the {better['D'] + better['B']} tweets with a preference: p = {p_value:.3f}")

    lines.append("")
    lines.append(f"## Main problem in the translation of run D ({len(compare_rows)} test tweets)")
    lines.append("")
    lines.append("| Problem | Tweets | Share |")
    lines.append("|---|---|---|")
    for value, count in errors.most_common():
        lines.append(f"| {value} | {count} | {share(count, len(compare_rows)):.0f}% |")

    lines.append("")
    lines.append(f"## Quality check of run D ({len(check_rows)} other test tweets)")
    lines.append("")
    lines.append("| Verdict | Tweets | Share |")
    lines.append("|---|---|---|")
    for value in VERDICT_VALUES:
        lines.append(f"| {value} | {verdicts[value]} | {share(verdicts[value], len(check_rows)):.0f}% |")

    summary = {
        "compared": len(compare_rows),
        "better": dict(better),
        "sign_test_p_value": p_value,
        "error_run_d": dict(errors),
        "checked": len(check_rows),
        "verdict": dict(verdicts),
    }
    text = "\n".join(lines) + "\n"
    (RESULTS_DIR / "error_analysis.md").write_text(text, encoding="utf-8")
    (RESULTS_DIR / "error_analysis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
