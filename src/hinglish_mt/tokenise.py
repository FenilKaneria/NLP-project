"""Tokenisers: word-level (run A) and SentencePiece BPE (runs B, C, D).

Usage: python -m hinglish_mt.tokenise

Vocabularies are built from training text only: PHINC train for the word
list, PHINC train plus the noisy synthetic pairs for BPE.
"""

import json
from collections import Counter

import sentencepiece as spm

from hinglish_mt import OUTPUT_DIR, PROCESSED_DIR
from hinglish_mt.clean import MASKS, tokenize
from hinglish_mt.splits import read_tsv

PAD = 0
UNK = 1
BOS = 2
EOS = 3
SPECIALS = ["<pad>", "<unk>", "<s>", "</s>"]

TOKENISER_DIR = OUTPUT_DIR / "tokenisers"
WORD_VOCAB = TOKENISER_DIR / "word_vocab.json"
BPE_MODEL = TOKENISER_DIR / "bpe8k.model"


class WordTokeniser:
    def __init__(self, words):
        self.words = SPECIALS + words
        self.ids = {}
        for i, word in enumerate(self.words):
            self.ids[word] = i

    @staticmethod
    def build(texts, min_freq=2):
        counts = Counter()
        for text in texts:
            counts.update(tokenize(text))
        words = []
        for word, count in counts.most_common():
            if count >= min_freq:
                words.append(word)
        return WordTokeniser(words)

    @staticmethod
    def load(path=WORD_VOCAB):
        words = json.loads(path.read_text(encoding="utf-8"))
        return WordTokeniser(words)

    def save(self, path=WORD_VOCAB):
        path.parent.mkdir(parents=True, exist_ok=True)
        words = self.words[len(SPECIALS) :]
        path.write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")

    @property
    def size(self):
        return len(self.words)

    def encode(self, text):
        ids = []
        for token in tokenize(text):
            ids.append(self.ids.get(token, UNK))
        return ids

    def decode(self, ids):
        words = []
        for i in ids:
            if i >= len(SPECIALS):
                words.append(self.words[i])
        return " ".join(words)


class BpeTokeniser:
    def __init__(self, path=BPE_MODEL):
        self.sp = spm.SentencePieceProcessor(model_file=str(path))

    @staticmethod
    def train(texts, path=BPE_MODEL, vocab_size=8000):
        path.parent.mkdir(parents=True, exist_ok=True)
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(texts),
            model_prefix=str(path.with_suffix("")),
            model_type="bpe",
            vocab_size=vocab_size,
            character_coverage=0.99999,
            user_defined_symbols=list(MASKS),
            pad_id=PAD,
            unk_id=UNK,
            bos_id=BOS,
            eos_id=EOS,
            minloglevel=1,
        )

    @property
    def size(self):
        return self.sp.get_piece_size()

    def encode(self, text):
        return self.sp.encode(text)

    def decode(self, ids):
        kept = []
        for i in ids:
            if i >= len(SPECIALS):
                kept.append(i)
        return self.sp.decode(kept)

    def pieces(self, text):
        return self.sp.encode(text, out_type=str)


def load_tokeniser(kind):
    if kind == "word":
        return WordTokeniser.load()
    return BpeTokeniser()


def all_texts(rows):
    texts = []
    for row in rows:
        texts.append(row["src"])
        texts.append(row["tgt"])
    return texts


def describe(tokeniser, rows, side):
    """Share of unknown tokens and tokens per word on one side of a file."""
    n_tokens = 0
    n_unknown = 0
    n_words = 0
    for row in rows:
        ids = tokeniser.encode(row[side])
        n_tokens += len(ids)
        n_unknown += ids.count(UNK)
        n_words += len(tokenize(row[side]))
    return {"unknown": n_unknown / n_tokens, "tokens_per_word": n_tokens / n_words}


def main():
    phinc_train = read_tsv(PROCESSED_DIR / "phinc_train.tsv")
    phinc_val = read_tsv(PROCESSED_DIR / "phinc_val.tsv")
    synthetic = read_tsv(PROCESSED_DIR / "synth_noisy.tsv")

    word = WordTokeniser.build(all_texts(phinc_train))
    word.save()
    BpeTokeniser.train(all_texts(phinc_train) + all_texts(synthetic))
    bpe = BpeTokeniser()

    report = {
        "word": {"trained_on": ["phinc_train.tsv"], "vocabulary": word.size},
        "bpe": {"trained_on": ["phinc_train.tsv", "synth_noisy.tsv"], "vocabulary": bpe.size},
    }
    # The validation file is only measured here, never trained on.
    for name, tokeniser in [("word", word), ("bpe", bpe)]:
        for side in ["src", "tgt"]:
            report[name]["phinc_train_" + side] = describe(tokeniser, phinc_train, side)
            report[name]["phinc_val_" + side] = describe(tokeniser, phinc_val, side)

    report_path = TOKENISER_DIR / "tokeniser_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
