"""Test scores by how mixed the tweet is.

Usage: python -m hinglish_mt.buckets

Each test tweet is put in one group by its language tags and Code-Mixing
Index (CMI). Every system is then scored on each group separately.
Writes outputs/results/cmi_buckets.md and cmi_buckets.json.
"""

import json

from hinglish_mt.clean import tokenize
from hinglish_mt.cmi import cmi
from hinglish_mt.evaluate import RESULTS_DIR, score
from hinglish_mt.langid import EN, HI, tag
from hinglish_mt.splits import read_tsv

SYSTEMS = ["copy", "run_b", "run_c", "run_d", "run_e", "run_f"]
TITLES = ["Copy", "Run B", "Run C", "Run D", "Run E", "Run F"]
GROUPS = ["Hindi only", "Mixed, low (CMI up to 20)", "Mixed, high (CMI above 20)", "English only", "All"]
HIGH_CMI = 20


def group_of(source):
    tags = tag(tokenize(source))
    value = cmi(tags)
    if value > HIGH_CMI:
        return "Mixed, high (CMI above 20)"
    if value > 0:
        return "Mixed, low (CMI up to 20)"
    if EN in tags and HI not in tags:
        return "English only"
    return "Hindi only"


def main():
    summary = {}
    sizes = {}
    for system in SYSTEMS:
        rows = read_tsv(RESULTS_DIR / f"{system}_test.tsv")
        outputs = {}
        references = {}
        for group in GROUPS:
            outputs[group] = []
            references[group] = []
        for row in rows:
            for group in [group_of(row["src"]), "All"]:
                outputs[group].append(row["hyp"])
                references[group].append(row["tgt"])

        summary[system] = {}
        for group in GROUPS:
            sizes[group] = len(outputs[group])
            summary[system][group] = score(outputs[group], references[group])

    lines = []
    for metric, title in [("chrf", "chrF"), ("bleu4", "BLEU-4")]:
        lines.append(f"## {title} by group")
        lines.append("")
        lines.append("| Group | Tweets | " + " | ".join(TITLES) + " | D minus B | F minus D |")
        lines.append("|---|---|" + "---|" * (len(TITLES) + 2))
        for group in GROUPS:
            cells = []
            for system in SYSTEMS:
                value = summary[system][group][metric]
                cells.append(f"{value:.2f}")
            gain_d = summary["run_d"][group][metric] - summary["run_b"][group][metric]
            gain_f = summary["run_f"][group][metric] - summary["run_d"][group][metric]
            lines.append(f"| {group} | {sizes[group]} | " + " | ".join(cells) + f" | {gain_d:+.2f} | {gain_f:+.2f} |")
        lines.append("")

    text = "\n".join(lines)
    (RESULTS_DIR / "cmi_buckets.md").write_text(text, encoding="utf-8")
    result = {"sizes": sizes, "scores": summary}
    (RESULTS_DIR / "cmi_buckets.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
