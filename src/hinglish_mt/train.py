"""Train one run from a config file.

Usage: python -m hinglish_mt.train configs/run_b.yaml [--resume] [--set key=value ...]

After each epoch the validation file is translated greedily and scored with
chrF. The best model is saved as best.pt. Training stops after `patience`
epochs without a new best.
"""

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from hinglish_mt import ROOT
from hinglish_mt.data import collate, encode_pairs, make_batches, pair_lengths
from hinglish_mt.decode import RUNS_DIR, translate
from hinglish_mt.evaluate import chrf
from hinglish_mt.model import Seq2Seq
from hinglish_mt.splits import read_tsv
from hinglish_mt.tokenise import PAD, load_tokeniser


@dataclass
class Config:
    name: str
    train: str
    val: str
    tokeniser: str = "bpe"
    init_from: str | None = None
    emb_dim: int = 256
    hidden_dim: int = 256
    layers: int = 2
    dropout: float = 0.3
    copy: bool = False
    lr: float = 1e-3
    clip: float = 1.0
    label_smoothing: float = 0.1
    max_tokens: int = 4000
    max_len: int = 100
    max_epochs: int = 60
    patience: int = 5
    seed: int = 42
    train_limit: int | None = None


def load_config(path, overrides=()):
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for item in overrides:
        key, value = item.split("=", 1)
        values[key] = yaml.safe_load(value)
    return Config(**values)


def batch_loss(model, batch, label_smoothing):
    """Average cross-entropy per target word, with label smoothing."""
    scores = model(batch["src"], batch["src_len"], batch["tgt_in"])
    scores = scores.flatten(0, 1)
    targets = batch["tgt_out"].flatten()
    real_words = targets != PAD

    correct_word_loss = F.nll_loss(scores, targets, ignore_index=PAD)
    all_words_loss = -scores[real_words].mean()
    return (1 - label_smoothing) * correct_word_loss + label_smoothing * all_words_loss


def train(config, resume=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    run_dir = RUNS_DIR / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)

    tokeniser = load_tokeniser(config.tokeniser)
    train_rows = read_tsv(ROOT / config.train)
    if config.train_limit is not None:
        train_rows = train_rows[: config.train_limit]
    val_rows = read_tsv(ROOT / config.val)
    val_sources = []
    val_references = []
    for row in val_rows:
        val_sources.append(row["src"])
        val_references.append(row["tgt"])

    examples = encode_pairs(train_rows, tokeniser, config.max_len)
    lengths = pair_lengths(examples)

    model = Seq2Seq(tokeniser.size, config.emb_dim, config.hidden_dim, config.layers, config.dropout, config.copy)
    model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.lr)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    epoch = 0
    best_chrf = -1.0
    best_epoch = 0

    if resume:
        last = torch.load(run_dir / "last.pt", map_location=device, weights_only=True)
        model.load_state_dict(last["model"])
        optimiser.load_state_dict(last["optimiser"])
        scaler.load_state_dict(last["scaler"])
        epoch = last["epoch"]
        best_chrf = last["best_chrf"]
        best_epoch = last["best_epoch"]
    else:
        log_path = run_dir / "log.jsonl"
        if log_path.exists():
            log_path.unlink()
        if config.init_from is not None:
            start = torch.load(RUNS_DIR / config.init_from / "best.pt", map_location=device, weights_only=True)
            model.load_state_dict(start["model"])

    print(f"{config.name}: {len(examples)} training pairs, vocabulary {tokeniser.size}, "
          f"{model.n_parameters()} parameters, {device}", flush=True)

    while epoch < config.max_epochs and epoch - best_epoch < config.patience:
        epoch += 1
        started = time.time()
        shuffle_rng = random.Random(config.seed * 1000 + epoch)
        batches = make_batches(lengths, config.max_tokens, shuffle_rng)

        model.train()
        loss_sum = 0.0
        word_count = 0
        for indices in batches:
            batch_examples = []
            for i in indices:
                batch_examples.append(examples[i])
            batch = collate(batch_examples, device)

            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                loss = batch_loss(model, batch, config.label_smoothing)

            optimiser.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
            scaler.step(optimiser)
            scaler.update()

            words = (batch["tgt_out"] != PAD).sum().item()
            loss_sum += loss.item() * words
            word_count += words

        val_outputs = translate(model, tokeniser, val_sources, max_len=config.max_len)
        val_chrf = chrf(val_outputs, val_references)

        if val_chrf > best_chrf:
            best_chrf = val_chrf
            best_epoch = epoch
            best = {
                "model": model.state_dict(),
                "model_args": model.args,
                "config": asdict(config),
                "epoch": epoch,
                "val_chrf": val_chrf,
            }
            torch.save(best, run_dir / "best.pt")

        last = {
            "model": model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_chrf": best_chrf,
            "best_epoch": best_epoch,
        }
        torch.save(last, run_dir / "last.pt")

        record = {
            "epoch": epoch,
            "train_loss": round(loss_sum / word_count, 4),
            "val_chrf": round(val_chrf, 2),
            "updates": len(batches),
            "seconds": round(time.time() - started),
        }
        with open(run_dir / "log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(record, flush=True)

    print(f"best validation chrF {best_chrf:.2f} at epoch {best_epoch}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    train(config, resume=args.resume)


if __name__ == "__main__":
    main()
