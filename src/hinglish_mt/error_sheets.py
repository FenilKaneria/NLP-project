"""Write the sheets for manual error analysis.

Usage: python -m hinglish_mt.error_sheets [--force]

Sheets that already exist are kept, because they may be filled in.
--force writes new empty sheets over them.

Picks test tweets at random (fixed seed) and writes two CSV files to
outputs/error_analysis/ that open in Excel:

* compare_b_d.csv: 100 tweets with the outputs of run B and run D side by side.
* check_run_d.csv: 50 other tweets with the output of run D only.

The empty columns are filled in by hand. guide.txt explains how.
"""

import argparse
import csv
import random

from hinglish_mt import OUTPUT_DIR
from hinglish_mt.evaluate import RESULTS_DIR
from hinglish_mt.splits import read_tsv

SHEET_DIR = OUTPUT_DIR / "error_analysis"
COMPARE_SIZE = 100
CHECK_SIZE = 50
SEED = 42

GUIDE = """How to fill in the error-analysis sheets
=========================================

compare_b_d.csv (100 tweets)
----------------------------
Column "better": which translation is closer to the meaning of the source.
    B       run B is better
    D       run D is better
    same    equally good or equally bad

Column "error_run_d": the main problem in run D's translation. Use one of:
    none            meaning is correct
    wrong meaning   fluent English, but says something else
    missing         part of the source is left out
    untranslated    a Hindi word is copied instead of translated
    wrong word      one word is mistranslated (often a name or a rare word)
    repetition      words or phrases are repeated
    grammar         meaning is right, English is broken
    bad reference   the human translation itself is wrong or incomplete

Column "notes": anything worth remembering, for example the word that went wrong.

check_run_d.csv (50 tweets)
---------------------------
Column "verdict":
    correct         a reader would understand the same thing as from the source
    needs change    anything else

Rules
-----
* Judge against the source tweet first. The reference is a help, not the truth.
* Ignore lowercase, punctuation and the placeholders <user>, <hashtag>, <url>.
* When two people fill in the same sheet, do it separately and compare afterwards.
"""


def write_sheet(path, columns, rows):
    # utf-8-sig lets Excel show Hindi words and emoji correctly.
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="overwrite existing sheets")
    args = parser.parse_args()

    if (SHEET_DIR / "compare_b_d.csv").exists() and not args.force:
        print(f"sheets already exist in {SHEET_DIR}, nothing written (use --force to overwrite)")
        return

    run_b = read_tsv(RESULTS_DIR / "run_b_test.tsv")
    run_d = read_tsv(RESULTS_DIR / "run_d_test.tsv")

    positions = list(range(len(run_d)))
    random.Random(SEED).shuffle(positions)
    compare_positions = positions[:COMPARE_SIZE]
    check_positions = positions[COMPARE_SIZE : COMPARE_SIZE + CHECK_SIZE]

    compare_rows = []
    for i in compare_positions:
        compare_rows.append({
            "id": run_d[i]["id"],
            "source": run_d[i]["src"],
            "reference": run_d[i]["tgt"],
            "run_b": run_b[i]["hyp"],
            "run_d": run_d[i]["hyp"],
            "better": "",
            "error_run_d": "",
            "notes": "",
        })

    check_rows = []
    for i in check_positions:
        check_rows.append({
            "id": run_d[i]["id"],
            "source": run_d[i]["src"],
            "reference": run_d[i]["tgt"],
            "run_d": run_d[i]["hyp"],
            "verdict": "",
            "notes": "",
        })

    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    compare_columns = ["id", "source", "reference", "run_b", "run_d", "better", "error_run_d", "notes"]
    check_columns = ["id", "source", "reference", "run_d", "verdict", "notes"]
    write_sheet(SHEET_DIR / "compare_b_d.csv", compare_columns, compare_rows)
    write_sheet(SHEET_DIR / "check_run_d.csv", check_columns, check_rows)
    (SHEET_DIR / "guide.txt").write_text(GUIDE, encoding="utf-8")
    print(f"wrote {len(compare_rows)} and {len(check_rows)} rows to {SHEET_DIR}")


if __name__ == "__main__":
    main()
