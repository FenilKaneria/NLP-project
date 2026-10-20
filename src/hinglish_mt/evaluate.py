"""Score trained runs on a PHINC split with sacreBLEU.

Usage: python -m hinglish_mt.evaluate run_a run_b [--split test] [--beam 4]

Two sets of scores are written for each run:
* masked (primary): output and reference keep the <user>, <hashtag>, <url> placeholders.
* restored: placeholders in the output are replaced by the originals from the source,
  and the output is compared with the raw reference.
All text is lowercased. The run name `copy` scores the untranslated source.
"""

import argparse
import json

from sacrebleu.metrics import BLEU, CHRF

from hinglish_mt import OUTPUT_DIR, PROCESSED_DIR
from hinglish_mt.clean import fix_text, mask, restore
from hinglish_mt.decode import load_run, translate
from hinglish_mt.splits import read_tsv, write_tsv

RESULTS_DIR = OUTPUT_DIR / "results"

CHRF_METRIC = CHRF()
BLEU4_METRIC = BLEU()
BLEU1_METRIC = BLEU(max_ngram_order=1)


def chrf(outputs, references):
    return CHRF_METRIC.corpus_score(outputs, [references]).score


def score(outputs, references):
    return {
        "chrf": CHRF_METRIC.corpus_score(outputs, [references]).score,
        "bleu4": BLEU4_METRIC.corpus_score(outputs, [references]).score,
        "bleu1": BLEU1_METRIC.corpus_score(outputs, [references]).score,
    }


def signatures():
    return {
        "chrf": str(CHRF_METRIC.get_signature()),
        "bleu4": str(BLEU4_METRIC.get_signature()),
        "bleu1": str(BLEU1_METRIC.get_signature()),
    }


def restore_row(row, output):
    """Returns the output with the source's mentions, hashtags and links put back, and the raw reference."""
    masked_source, originals = mask(fix_text(row["src_raw"]))
    restored_output = restore(output, originals).lower()
    raw_reference = fix_text(row["tgt_raw"]).lower()
    return restored_output, raw_reference


def evaluate_rows(rows, outputs):
    masked_references = []
    restored_outputs = []
    raw_references = []
    for row, output in zip(rows, outputs):
        restored_output, raw_reference = restore_row(row, output)
        masked_references.append(row["tgt"])
        restored_outputs.append(restored_output)
        raw_references.append(raw_reference)
    return {
        "sentences": len(rows),
        "masked": score(outputs, masked_references),
        "restored": score(restored_outputs, raw_references),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--beam", type=int, default=4)
    args = parser.parse_args()

    rows = read_tsv(PROCESSED_DIR / f"phinc_{args.split}.tsv")
    sources = []
    for row in rows:
        sources.append(row["src"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for name in args.runs:
        result = {"run": name, "split": args.split}
        if name == "copy":
            outputs = sources
        else:
            model, tokeniser, checkpoint = load_run(name)
            config = checkpoint["config"]
            outputs = translate(model, tokeniser, sources, beam=args.beam, max_len=config["max_len"])
            result["beam"] = args.beam
            result["checkpoint_epoch"] = checkpoint["epoch"]
            result["checkpoint_val_chrf_greedy"] = checkpoint["val_chrf"]
            result["parameters"] = model.n_parameters()
            result["vocabulary"] = tokeniser.size
            result["config"] = config

        result.update(evaluate_rows(rows, outputs))
        result["signatures"] = signatures()
        result_path = RESULTS_DIR / f"{name}_{args.split}.json"
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        output_rows = []
        for row, output in zip(rows, outputs):
            output_rows.append({"id": row["id"], "src": row["src"], "tgt": row["tgt"], "hyp": output})
        write_tsv(output_rows, RESULTS_DIR / f"{name}_{args.split}.tsv", fields=["id", "src", "tgt", "hyp"])

        masked = result["masked"]
        print(f"{name}  {args.split}  chrF {masked['chrf']:.2f}  BLEU-4 {masked['bleu4']:.2f}  BLEU-1 {masked['bleu1']:.2f}")


if __name__ == "__main__":
    main()
