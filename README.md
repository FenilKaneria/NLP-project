# Hinglish to English translation for social media text

A GRU encoder-decoder with attention that translates Hinglish tweets (Hindi and English mixed, typed in English letters) into English. It is trained from random weights: first on synthetic Hinglish made from the IIT Bombay Hindi-English corpus, then on the real PHINC tweet pairs. The repository holds the full pipeline, four controlled experiments and a web demo.

## Requirements

- Python 3.11 or newer (developed on 3.13.9)
- PyTorch 2.6.0 with CUDA 12.4. Training was done on an NVIDIA RTX 3050 with 6 GB.
- The packages in `requirements.txt`
- Windows PowerShell for `scripts/run_all.ps1`

## Installation

```text
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
```

## Data

The data is not part of the repository. Download the corpora and place the files at these paths (the third is needed only for the sentiment application):

| Corpus | Files | Path |
|---|---|---|
| PHINC ([Zenodo record 3605597](https://zenodo.org/records/3605597)) | `English-Hindi code-mixed parallel corpus.csv` | `data/PHINC/` |
| IIT Bombay English-Hindi corpus, training part ([paper](https://arxiv.org/abs/1710.02855), [OPUS page](https://opus.nlpl.eu/IITB/corpus/version/IITB)) | `IITB.en-hi.en`, `IITB.en-hi.hi` | `data/IIT bombay data/parallel-n/` |
| SentiMix, SemEval-2020 Task 9, Hinglish part (only for the sentiment application; a public copy is at [Hugging Face, RTT1/SentiMix](https://huggingface.co/datasets/RTT1/SentiMix)) | `train_14k_split_conll.txt`, `Hindi_test_unalbelled_conll_updated.txt`, `test_labels_hinglish.txt` | `data/SentiMix/` |

The IIT Bombay corpus is released for non-commercial research.

## Quick start

With trained models in `outputs/runs/`, start the demo:

```text
streamlit run demo/app.py
```

Type a Hinglish sentence and press Translate. The page shows the English translation, the language of each input word and the Code-Mixing Index.

| Input | Output |
|---|---|
| `ye sarkar kuch nahi kar rahi hai` | this government is not doing anything |
| `aaj office me bahut kaam tha, i am so tired` | today i had a lot of work in office, i am so tired |

## Reproducing the results

One script runs everything from the raw data to the result tables (about 5 hours on the GPU above):

```text
.\scripts\run_all.ps1
```

Add `-SkipSeeds` to leave out the repeated runs with other random seeds. The single steps:

| Command | What it does |
|---|---|
| `python -m hinglish_mt.splits` | Cleans PHINC and writes the train, validation and test files |
| `python -m hinglish_mt.analysis` | Writes the data statistics |
| `python -m hinglish_mt.synth` | Makes the clean and noisy synthetic files |
| `python -m hinglish_mt.tokenise` | Builds the word list and the word-piece model |
| `python -m hinglish_mt.train configs/run_b.yaml` | Trains one run; `configs/` has one file per run |
| `python -m hinglish_mt.codemix` | Builds the word dictionary and the code-mixed synthetic files (needed for run F) |
| `python -m hinglish_mt.evaluate copy run_a run_b run_c run_d run_e run_f` | Scores the runs on the test part |
| `python -m hinglish_mt.ablation` | Writes the comparison table and significance tests |
| `python -m hinglish_mt.buckets` | Scores each code-mixing group separately |
| `python -m hinglish_mt.rare_words run_b run_d run_e run_f` | Measures how well rare words and names are kept |
| `python -m hinglish_mt.embeddings` | Lists the closest word vectors for ten words |
| `python -m hinglish_mt.sentiment` | Sentiment application: translates the SentiMix test tweets and scores an English sentiment tool on them |
| `python -m hinglish_mt.error_sheets` | Writes the sheets for manual error analysis (keeps existing sheets unless `--force` is given) |
| `python -m hinglish_mt.error_summary` | Counts the answers in the filled-in sheets (`*_annotated.csv`) |

Train in this order: `run_a`, `run_b`, `pretrain_clean`, `pretrain_noisy`, `run_c`, `run_d`, `pretrain_noisy_copy`, `run_e`, `pretrain_mixed_copy`, `run_f`. Runs C to F start from the pre-trained models. Run `hinglish_mt.codemix` before `pretrain_mixed_copy`.

## Results

PHINC test part, 1,341 tweets, beam size 4. Scores are from sacreBLEU on lowercased text with mentions, hashtags and links replaced by placeholders.

| System | Training data | chrF | BLEU-4 | BLEU-1 |
|---|---|---|---|---|
| Copy baseline | none, the tweet is returned unchanged | 41.83 | 29.29 | 45.72 |
| Run A | PHINC, whole words | 41.50 | 28.49 | 54.45 |
| Run B | PHINC, word pieces | 46.71 | 30.38 | 54.28 |
| Run C | clean synthetic data, then PHINC | 51.72 | 34.42 | 59.76 |
| Run D | noisy synthetic data, then PHINC | 51.64 | 34.68 | 60.40 |
| Run E | as run D, with copy mechanism | 51.59 | 35.44 | 59.88 |
| Run F | as run E, synthetic data code-mixed | 52.23 | 35.93 | 60.68 |

- Pre-training on synthetic data adds about 5 chrF over PHINC alone (run D against run B, paired bootstrap p < 0.001).
- Adding spelling noise to the synthetic data gives no significant further gain (run D against run C, p = 0.597 on chrF).
- The copy mechanism does not change chrF (run E against run D, p = 0.563) but raises BLEU-4 by 0.8 (p = 0.014). Words never seen in training that must pass through unchanged are kept 82.0% of the time instead of 67.6% (`outputs/results/rare_words.md`).
- Mixing English words into the synthetic data adds 0.6 chrF (run F against run E, p = 0.003). Run F is the best run and is used by the demo.

The full tables, including three random seeds per run, are written to `outputs/results/ablation.md`.

### Application: sentiment analysis

An English sentiment tool (VADER) is given the Hinglish tweet as it is, or our translation of it (run F). SentiMix test set, 3,000 tweets.

| System | Accuracy | Macro F1 |
|---|---|---|
| Always "neutral" | 36.7 | 17.9 |
| VADER on the Hinglish tweet | 47.5 | 45.1 |
| VADER on our translation | 49.2 | 46.7 |
| Classifier trained on 14,000 labelled Hinglish tweets (reference) | 67.6 | 68.1 |

- Over all tweets the gain from translation is small and borderline (macro F1 +1.6, p = 0.049, 95% interval -0.3 to +3.5).
- On tweets written in Hindi only (173 tweets) macro F1 rises from 19.9 to 44.7. On highly mixed tweets translation makes the result slightly worse (-1.8).

Details are in `outputs/results/sentiment.md`.

## Documentation

- [DOCUMENTATION.md](DOCUMENTATION.md): the method, experiments, results, limitations and references, in plain language.
- [context.md](context.md): detailed notes and numbers for each phase of the work.

## Testing

```text
python -m pytest
```
