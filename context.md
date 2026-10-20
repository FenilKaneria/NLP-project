# Project context: Hinglish → English code-mixed MT

Living summary of the plan and progress. Updated at the end of every phase.
Source of requirements: `PRD Hinglish Social Media Code-Mixed Machine Translation.pdf`. Due 31 Oct 2026.

## Goal

Train a GRU-with-attention translator from scratch for Romanised Hinglish → English. Pre-train on synthetic Hinglish made from the IIT Bombay Hindi-English corpus, fine-tune on real PHINC pairs, and show run D beats run B on a held-out PHINC test split (chrF primary, BLEU-4, BLEU-1).

| Run | Training data | Tokens | Spelling noise |
|---|---|---|---|
| A | PHINC only | word | no |
| B | PHINC only | BPE | no |
| C | synthetic → PHINC | BPE | no |
| D (main) | synthetic → PHINC | BPE | yes |

## Decisions

- Work one phase at a time; stop at each gate and wait for go-ahead.
- Data in `data/` is enough for the core project. Not used for now: LinCE language identification set, LinCE Eng-Hinglish MT set, Dakshina/Aksharantar. So the n-gram language identifier is deferred and the rule-based tagger is used everywhere.
- No version control: the git repository was removed on 11 Oct, and the project is not published to GitHub.
- Both source and target are lowercased, so all scores are case-insensitive.
- Code style: simple, easy-to-read syntax (plain loops, named variables) and few comments.
- Mentions, hashtags and URLs are masked to `<user>`, `<hashtag>`, `<url>` and restored after decoding. Primary scores are on masked text because PHINC references corrupt handles and links.

## Phases

| # | Phase | Status |
|---|---|---|
| 1 | Setup, cleaning, splits | Done (10 Oct) |
| 2 | Corpus analysis, rule-based language tagger, Code-Mixing Index | Done (11 Oct) |
| 3 | Synthetic data: IITB filter, romaniser, spelling noise, 300k pairs | Done (11 Oct) |
| 4 | Tokenisers, GRU model, training and scoring code, runs A and B | Done (11 Oct) |
| 5 | Pre-training and fine-tuning, runs C and D, ablation table | Done (11 Oct) |
| 6 | Streamlit demo, scores by code-mixing level, embedding check | Done (11 Oct) |
| 7 | Error-analysis sheets, README, reproduction script | Done (11 Oct) |
| 8 | Extensions: copy mechanism (run E), code-mixed synthetic data (run F) | Done (11 Oct) |
| 9 | Application: sentiment analysis with the translator | Done (11 Oct) |
| 10 | Manual error analysis filled in and counted | Done (11 Oct) |

## Phase 1 — Setup, cleaning, splits (done)

Built:
- `src/hinglish_mt/clean.py`: mojibake repair (`ftfy`), whitespace collapse, masking and restoring of mentions, hashtags and URLs, lowercasing.
- `src/hinglish_mt/splits.py`: cleans PHINC, removes duplicates, writes seeded 80/10/10 splits. Run with `python -m hinglish_mt.splits`.
- `tests/test_clean.py`, `tests/test_splits.py`.
- `.venv` virtual environment, `requirements.txt`, `pyproject.toml` (package installed in editable mode).

Outputs: `data/processed/phinc_{train,val,test}.tsv` (tab-separated, columns `id, src, tgt, src_raw, tgt_raw`) and `phinc_split_report.json`.

Results (seed 42):

| | Count |
|---|---|
| PHINC pairs in | 13,738 |
| Dropped: Devanagari in source | 13 |
| Dropped: target is only punctuation (broken rows, e.g. `)`) | 217 |
| Dropped: duplicate source after masking and normalisation | 109 |
| Kept | 13,399 |
| Train / val / test | 10,719 / 1,339 / 1,341 |
| Kept pairs where source equals target (already English) | 460 |
| Kept sources outside 5–40 tokens | 523 |

Notes for the report:
- The 217 broken rows are a defect in the released corpus that the paper does not mention.
- Duplicate detection runs on masked text, so tweets that differ only in the handle count as duplicates. This is stricter than exact matching, which finds only 19.

## Phase 2 — Corpus analysis and rule-based tagger (done)

Built:
- `src/hinglish_mt/langid.py`: `tag(tokens)` returns `en`, `hi` or `other` per token. English = known to the `wordfreq` English list (Zipf ≥ 3.0); other = placeholder, number, symbol, laughter, link residue; Hindi otherwise. Two hand lists fix the dictionary rule: about 190 Romanised Hindi words that the English list also contains (`hai`, `ke`, `se`), and 56 words valid in both languages (`to`, `me`, `is`, `do`) that are tagged by the majority language of the two words on each side.
- `src/hinglish_mt/cmi.py`: Code-Mixing Index (Das and Gambäck 2014, the formula the PHINC paper cites) and switch points.
- `src/hinglish_mt/analysis.py`: all corpus statistics. Run with `python -m hinglish_mt.analysis`; takes `--name` and `--input` so Phase 3 can run it on the synthetic corpus.
- `tokenize()` added to `clean.py`. Tests in `tests/test_langid.py`.

Outputs in `outputs/analysis/`: `phinc_stats.json`, `phinc_summary.md` (full tables), `phinc_tagged_sample.txt` (200 tagged train sentences for reading).

Results on cleaned PHINC (13,399 pairs):

| Measure | Value |
|---|---|
| Source tokens / word types | 203,772 / 19,514 |
| Target tokens / word types | 192,759 / 13,982 |
| Mean source / target length (tokens) | 15.2 / 14.4 |
| Source types seen only once | 56.4% |
| Hindi / English / other tokens | 52.1% / 20.8% / 27.1% |
| Sentences containing both languages | 69.1% |
| Mean CMI, all sentences | 15.6 |
| Mean CMI, mixed sentences only | 22.7 |
| CMI on pooled corpus counts | 28.5 |
| Mean switch points per sentence | 2.16 |
| Hindi-dominant / English-dominant sentences | 84.8% / 13.1% |
| Words valid in both languages | 7.2% of words |
| Words not in the English dictionary | 44.6% of tokens, 66.0% of types |
| Test source words unseen in train | 6.8% of tokens, 27.0% of types, 53.3% of sentences |
| Hindi tokens whose word has more than one spelling | 60.8% (strict key) to 67.4% (loose key) |

Notes for the report:
- The paper's CMI of 77 cannot come from the formula it cites: with two languages that formula tops out at 50 per sentence. Our figures are 15.6 over all sentences and 22.7 over mixed ones, so the two numbers are not comparable.
- Over half of test sentences contain a source word never seen in train. This is the case for subword tokens (run A versus run B).
- Spelling variation is measured with two keys that bracket it. Strict: spellings differ only by stretched letters, long vowels or a final n/h (`nahi`, `nahin`, `naahi`). Loose: same first letter and consonants, which also catches `nhi` but merges unrelated words such as `bhi` and `bhai`.
- The tagger has no accuracy figure: there is no labelled data without LinCE. Known weak spots are named entities (`kejriwal` tagged Hindi, `kolkata` English) and ambiguous words with no clear neighbours.
- 4,144 sentences (30.9%) are not mixed: 3,652 Hindi only, 485 English only, 7 with no language-tagged word.

Change to Phase 1 code made in this phase: link masking now also catches links glued to a handle or hashtag (`#taghttps://...`), links with no scheme (`twitter.com/...`), PHINC's own `http/URL` marker and a truncated bare `https`. Leftover link fragments fell from 372 source sentences to 1. Splits were regenerated; sizes are unchanged (10,719 / 1,339 / 1,341).

## Phase 3 — Synthetic data (done)

Built:
- `src/hinglish_mt/romanise.py`: rule-based Devanagari to casual Roman spelling. Parses each word into syllables, deletes the inherent vowel at the end of a word and between two pronounced vowels (`करना` → `karna`, `समझना` → `samajhna`), and spells long vowels by convention (`काम` → `kaam`, `हमारा` → `hamara`). A short exception list covers frequent words the rules miss (`वह` → `wo`, `नहीं` → `nahi`).
- `src/hinglish_mt/noise.py`: spelling noise with probabilities set from PHINC train counts. Operations: drop final nasalisation (`hain` → `hai`), double a long vowel (`shadi` → `shaadi`), shortening table of 41 frequent words (`nahi` → `nhi`, `kyon` → `kyu`), vowel length, vowel drop (`karna` → `krna`), sound swap (`ph` → `f`), stretched last letter. All rates sit in `NoiseConfig`.
- `src/hinglish_mt/synth.py`: filters IITB, samples with seed 42, writes the clean and the noisy file. Run with `python -m hinglish_mt.synth` (about 3 minutes).
- `analysis.py` gained `--limit` and `--compare`. Tests in `tests/test_synth.py` (116 tests pass in total).

Outputs:
- `data/processed/synth_clean.tsv` and `synth_noisy.tsv`: 300,000 pairs each, same pairs in the same order, same columns as the PHINC files (`src_raw` holds the Devanagari).
- `data/processed/synth_clean_val.tsv` and `synth_noisy_val.tsv`: 2,000 pairs, disjoint from the 300,000, for choosing the pre-training checkpoint in Phase 5.
- `data/processed/synth_report.json`, `outputs/analysis/synth_sample.txt` (200 pairs: Hindi, clean, noisy, English), `outputs/analysis/compare_phinc_synth_clean_synth_noisy.md`.

IITB filter (1,659,083 pairs in, 658,759 kept, 302,000 sampled):

| Dropped because | Count |
|---|---|
| Either side outside 5–40 tokens | 739,368 |
| Duplicate Hindi sentence | 201,789 |
| Interface strings and markup (`_`, `%s`, tags) | 27,075 |
| Hindi side under 90% Devanagari | 16,126 |
| One side more than twice as long as the other | 14,447 |
| English side not Latin script | 1,146 |
| Mention, hashtag or link on one side only | 373 |

PHINC against the synthetic files (first 13,399 pairs of each, so sizes match):

| Measure | PHINC | Synthetic clean | Synthetic noisy |
|---|---|---|---|
| Source word types | 19,514 | 22,582 | 31,463 |
| Mean source / target length (tokens) | 15.2 / 14.4 | 18.8 / 17.7 | 18.8 / 17.7 |
| Hindi / English / other tokens | 52.1% / 20.8% / 27.1% | 85.8% / 2.8% / 11.4% | 85.7% / 2.9% / 11.5% |
| Mean CMI, all sentences | 15.6 | 3.2 | 3.3 |
| Hindi tokens whose word has more than one spelling (strict key) | 60.8% | 33.0% | 64.6% |
| Same, loose key | 67.4% | 58.9% | 68.2% |

How much of PHINC train's source vocabulary the 300,000 synthetic pairs contain:

| | Clean | Noisy |
|---|---|---|
| Hindi-tagged tokens | 84.6% | 90.2% |
| Hindi-tagged word types | 41.9% | 54.5% |
| All word tokens | 74.0% | 78.9% |
| All word types | 34.8% | 43.8% |

The noise changes 20.2% of source words.

Notes for the report:
- The noise brings spelling variation from 33.0% to 64.6% of Hindi tokens, close to PHINC's 60.8%, and raises coverage of PHINC's Hindi tokens from 84.6% to 90.2%. This is the measured case for run D over run C.
- The synthetic source is romanised Hindi, not code-mixed text: 2.8% English tokens against PHINC's 20.8%, mean CMI 3.2 against 15.6. Most of that 2.8% is romanised Hindi that happens to match an English word (`and` from `अंड`, `use` from `उसे`). The model meets real code-mixing only in fine-tuning.
- Domain differs too: IITB is formal (government, religious and technical text) and its sentences are longer than tweets.
- IITB has 1.66M pairs, not the PRD's 1.49M, and 12% of them are duplicate Hindi sentences.
- Romaniser limits: schwa deletion ignores word structure (`अनुमति` → `anumti`); English loanwords are spelled as they sound (`बैंक` → `baink`, `इंडिया` → `indiya`), so they do not match the English spelling used in real Hinglish. The romaniser is checked by 58 unit-test words and by reading the sample, not against a transliteration set.
- Noise probabilities were set from PHINC train only (never val or test). The vowel drop works on the spelled word, so it sometimes removes a long vowel that a person would keep (`rajnitik` → `rjnitik`).

## Phase 4 — Model code, runs A and B (done)

Built:
- `src/hinglish_mt/tokenise.py`: word tokeniser (PHINC train, words seen at least twice) and SentencePiece BPE with 8,000 pieces (PHINC train plus `synth_noisy.tsv`). Both use one vocabulary for Hinglish and English. Run with `python -m hinglish_mt.tokenise`.
- `src/hinglish_mt/data.py`: encoding, padding and batching by token count.
- `src/hinglish_mt/model.py`: 2-layer bidirectional GRU encoder, 2-layer GRU decoder, additive attention, hidden and embedding size 256, dropout 0.3, one embedding table shared by source, target and output layer.
- `src/hinglish_mt/train.py`: Adam, gradient clipping 1.0, label smoothing 0.1, mixed precision, early stopping on validation chrF (patience 5), `--resume`. Run with `python -m hinglish_mt.train configs/run_b.yaml`.
- `src/hinglish_mt/decode.py`: greedy decoding (validation) and beam search with beam 4 (final scores).
- `src/hinglish_mt/evaluate.py`: chrF, BLEU-4 and BLEU-1 with sacreBLEU. Run with `python -m hinglish_mt.evaluate copy run_a run_b`.
- `configs/run_a.yaml`, `configs/run_b.yaml`. Tests in `tests/test_model.py` (126 tests pass in total).

Outputs: `outputs/tokenisers/` (both tokenisers and `tokeniser_report.json`, which lists the training files), `outputs/runs/<run>/` (`best.pt`, `last.pt`, `log.jsonl`), `outputs/results/<run>_test.json` and `.tsv` (scores and every test translation).

Tokenisers on PHINC:

| | Word (run A) | BPE (runs B, C, D) |
|---|---|---|
| Vocabulary | 12,695 | 8,000 |
| Trained on | `phinc_train.tsv` | `phinc_train.tsv`, `synth_noisy.tsv` |
| Unknown tokens, validation source | 7.8% | 0.04% |
| Unknown tokens, validation target | 5.3% | 0.02% |
| Tokens per word, train source | 1.00 | 1.35 |

Test split results (1,341 sentences, beam 4, seed 42):

| Run | chrF | BLEU-4 | BLEU-1 | chrF, placeholders restored | Best epoch | Validation chrF (greedy) |
|---|---|---|---|---|---|---|
| Copy the source unchanged | 41.83 | 29.29 | 45.72 | 48.79 | | |
| A: PHINC, word | 41.50 | 28.49 | 54.45 | 48.18 | 22 | 39.86 |
| B: PHINC, BPE | 46.71 | 30.38 | 54.28 | 52.76 | 32 | 44.06 |

Training took about 4 minutes for run A and 7 minutes for run B on the RTX 3050.

Notes for the report:
- BPE beats words by 5.2 chrF and 1.9 BLEU-4 on the same data. BLEU-1 is the same, so the gain is in word forms and longer matches, not in how many words are right.
- Copying the source unchanged already scores 41.83 chrF and 29.29 BLEU-4, because tweets contain English words, names, placeholders and some are fully English. Run A is no better than copying on chrF; run B is 4.9 above it. Every later result should be read against this floor.
- Scores with placeholders restored are about 6 chrF higher for every system, because handles and links are long strings copied from the source. The ranking is the same. The masked scores stay primary.
- Checks done before the runs: the model reaches loss 0.005 on 200 pairs; beam 1 gives the same output as greedy (unit test); no memory problem (run A used about 0.9 GB of the 6 GB GPU memory).
- Changes from the plan: (1) no `tanh` before the output layer, because with the shared embedding table it made learning very slow; (2) batches hold 2,000 tokens, not 4,000, so that the small PHINC set gets more updates per epoch; (3) the BPE model keeps 99.999% of characters, because at the usual 99.95% the `!` sign became unknown (PHINC is only 3% of the BPE training text); (4) beam search ranks finished translations by average log-probability per word; (5) the word model's unknown-word outputs are left out of the translation.
- One seed only so far. A difference of a point or less between two runs should not be trusted without more seeds or the bootstrap test planned for Phase 5.
- Output quality is low in absolute terms: sentences are fluent but often wrong in meaning. Reading `outputs/results/run_b_test.tsv` shows this.

## Phase 5 — Pre-training, runs C and D (done)

Built:
- `configs/pretrain_clean.yaml`, `pretrain_noisy.yaml`: 10 epochs on the 300,000 synthetic pairs, learning rate 1e-3, 4,000 tokens per batch, checkpoint chosen on the 2,000-pair synthetic validation file.
- `configs/run_c.yaml`, `run_d.yaml`: start from the pre-trained weights (`init_from`), fine-tune on PHINC train at learning rate 2e-4.
- `src/hinglish_mt/ablation.py`: results table, three-seed summary and paired bootstrap test (1,000 samples). Run with `python -m hinglish_mt.ablation`.
- `DOCUMENTATION.md`: project documentation written for the faculty presentation. Update it when results change.

Outputs: `outputs/runs/` (two pre-trained models, runs C and D, and seeds 43 and 44 of runs B, C, D), `outputs/results/ablation.md` and `ablation.json`, one `_test.json` and `_test.tsv` per run.

Pre-training took about 40 minutes per model (4 minutes per epoch). Validation chrF on the synthetic validation file: clean 40.35 (epoch 10), noisy 38.49 (epoch 9). The clean model was still improving slowly at epoch 10.

Test split, seed 42, beam 4:

| Run | Training data | chrF | BLEU-4 | BLEU-1 |
|---|---|---|---|---|
| Copy | source unchanged | 41.83 | 29.29 | 45.72 |
| A | PHINC only, word | 41.50 | 28.49 | 54.45 |
| B | PHINC only, BPE | 46.71 | 30.38 | 54.28 |
| C | clean synthetic, then PHINC, BPE | 51.72 | 34.42 | 59.76 |
| D | noisy synthetic, then PHINC, BPE | 51.64 | 34.68 | 60.40 |

Three seeds (42, 43, 44):

| Run | chrF mean | chrF range | BLEU-4 mean | BLEU-4 range |
|---|---|---|---|---|
| B | 45.99 | 45.29 to 46.71 | 30.56 | 29.96 to 31.33 |
| C | 51.66 | 51.25 to 52.00 | 34.40 | 34.33 to 34.46 |
| D | 51.64 | 51.56 to 51.73 | 34.80 | 34.68 to 34.90 |

Paired bootstrap on the seed 42 outputs:

| Comparison | Metric | Difference | 95% interval | p-value |
|---|---|---|---|---|
| D minus B | chrF | +4.93 | +4.39 to +5.53 | < 0.001 |
| D minus B | BLEU-4 | +4.29 | +3.64 to +4.94 | < 0.001 |
| D minus C | chrF | -0.07 | -0.54 to +0.40 | 0.597 |
| D minus C | BLEU-4 | +0.26 | -0.35 to +0.89 | 0.169 |
| C minus B | chrF | +5.01 | +4.46 to +5.55 | < 0.001 |
| C minus B | BLEU-4 | +4.03 | +3.30 to +4.68 | < 0.001 |
| B minus A | chrF | +5.21 | +4.43 to +6.08 | < 0.001 |
| B minus A | BLEU-4 | +1.89 | +1.11 to +2.89 | < 0.001 |

Notes for the report:
- The main claim of the PRD holds: run D beats run B by 4.9 chrF and 4.3 BLEU-4, significant at p < 0.001, and the gap is far larger than the spread between seeds.
- Pre-training on synthetic data is what helps. It is worth about as much as the step from words to BPE (5.0 against 5.2 chrF).
- The spelling noise gives no reliable gain. Run D equals run C on chrF. Run D is slightly higher on BLEU-4 in all three seeds (34.68 to 34.90 against 34.33 to 34.46), but the bootstrap test does not confirm it (p = 0.169). This must be reported as "no significant difference".
- Likely reason (not tested): BPE already splits unusual spellings into known pieces, and fine-tuning on PHINC teaches the real spellings anyway. The noise raised vocabulary coverage (Phase 3), but that did not turn into better translations.
- For runs C and D the three seeds change only the fine-tuning; all three start from the same pre-trained model. For run B the seed changes the whole training. So the seed spread of C and D understates their true variation.
- Clean pre-training was stopped at 10 epochs while the score was still rising, so the gain from pre-training is probably a lower bound.
- One epoch of the last run shows 57 minutes because the laptop went to sleep; training itself was not affected.

## Phase 6 — Demo and supporting results (done)

Built:
- `demo/app.py`: Streamlit page that uses run D only. Type or pick a sentence, press Translate. Shows the translation with mentions and hashtags put back, each input word coloured by language, and the sentence's CMI. Run with `streamlit run demo/app.py`. Checked in a browser; screenshot in `outputs/demo_screenshot.png`.
- `src/hinglish_mt/buckets.py`: test scores per mixing group. Run with `python -m hinglish_mt.buckets`.
- `src/hinglish_mt/embeddings.py`: nearest neighbours of ten words in the embedding table. Run with `python -m hinglish_mt.embeddings`.
- `tests/test_buckets.py` (127 tests pass in total).

Outputs: `outputs/results/cmi_buckets.md` and `.json`, `outputs/analysis/embedding_neighbours.md`.

chrF on the test split by mixing group (seed 42):

| Group | Tweets | Copy | Run B | Run C | Run D | D minus B |
|---|---|---|---|---|---|---|
| Hindi only | 351 | 23.94 | 36.67 | 41.56 | 40.92 | +4.25 |
| Mixed, low (CMI up to 20) | 497 | 40.82 | 45.78 | 50.79 | 50.90 | +5.12 |
| Mixed, high (CMI above 20) | 451 | 47.08 | 49.02 | 54.39 | 54.42 | +5.40 |
| English only | 42 | 82.91 | 79.89 | 81.74 | 81.13 | +1.24 |
| All | 1,341 | 41.83 | 46.71 | 51.72 | 51.64 | +4.93 |

Selected embedding neighbours in run D (cosine similarity), with run B for comparison. Full lists of eight are in `outputs/analysis/embedding_neighbours.md`:

| Word | Run D | Run B |
|---|---|---|
| nahi | nhi 0.89, nahin 0.87, nai 0.77 | nhi 0.73, nahin 0.69, nai 0.62 |
| bahut | bohot 0.84, bahot 0.83, bht 0.76 | bahot 0.59, bohot 0.49 |
| sarkar | sarkaar 0.80, sarkari 0.53, government 0.44 | government 0.54, sarkaar 0.48 |
| ghar | gaanv 0.44, home 0.44, house 0.40 | house 0.48, home 0.45 |

Notes for the report:
- Pre-training helps every group that contains Hindi by 4 to 5 chrF. The gain is not limited to one kind of tweet.
- Hindi-only tweets are the hardest (chrF 40.9 for run D) although the synthetic data is closest to them. Mixed tweets score higher mainly because their English words can be copied: the copy baseline rises from 23.9 to 47.1 across the groups.
- The model's gain over copying shrinks as mixing grows: +17.0 chrF on Hindi-only, +10.1 on low mixing, +7.3 on high mixing.
- On English-only tweets no model beats copying (82.9). The models slightly damage text that needed no translation. Only 42 tweets, so this is an observation, not a tested result.
- The groups come from the rule-based tagger, which has no accuracy figure. The 20-point CMI border was chosen to give two mixed groups of similar size.
- Embeddings: in run D the spellings of one word sit very close together and the English translation is among the neighbours. In run B the same pattern exists but is weaker. This is an illustration with ten hand-picked words from one seed, not a measurement.
- Demo examples show the difference between runs: `ye sarkar kuch nahi kar rahi hai` gives "this government is not doing anything" in run D and "this is not able to sarkar the government." in run B.

## Phase 7 — Error-analysis material and write-up (done)

Built:
- `src/hinglish_mt/error_sheets.py`: picks test tweets with seed 42 and writes the sheets. Run with `python -m hinglish_mt.error_sheets`.
- `scripts/run_all.ps1`: repeats the pipeline from the raw data to the result tables, about 2.5 hours. `-SkipSeeds` leaves out seeds 43 and 44.
- `README.md`: requirements, installation, where to get the data, demo, reproduction steps, results table.

Outputs in `outputs/error_analysis/`:
- `compare_b_d.csv`: 100 test tweets with run B and run D side by side. Empty columns to fill in: `better` (B, D or same), `error_run_d` (one of eight error types), `notes`.
- `check_run_d.csv`: 50 other test tweets with run D only. Empty columns: `verdict` (correct or needs change), `notes`.
- `guide.txt`: how to fill in the sheets and the list of error types.

Still open (work for the team, not code):
- Filling in the two sheets by hand, then counting the error types and the share of correct outputs for the report. Rerunning `error_sheets` overwrites the files, so keep filled-in copies under another name.
- Checking the ten references in `DOCUMENTATION.md` against the papers.

Notes:
- `run_all.ps1` was checked for syntax and its helper function was tested, but the full 2.5-hour run was not executed. Each step in it was run on its own during Phases 1 to 7.
- The README links PHINC on Zenodo (record 3605597, confirmed). For the IIT Bombay corpus it links the paper and the OPUS page; the original download page was not confirmed.
- The error types in `guide.txt` are our own list, not taken from the PHINC paper.
- Not done: three full seeds of pre-training, the n-gram language identifier, a Transformer baseline. (English words mixed into the synthetic data was done later as run F, Phase 8.)

## Phase 8 — Extensions: copy mechanism and code-mixed synthetic data (done)

Reason: the stated limit of run D is "topic right, meaning wrong, weak on rare words and names". Measured, these are two problems, and each got one new run.

| Run | Pre-training data | Copy | Compared with |
|---|---|---|---|
| E | noisy synthetic | yes | D |
| F | noisy and code-mixed synthetic | yes | E |

Built:
- `src/hinglish_mt/rare_words.py`: copy rate and reference word recall by how often a word occurs in PHINC train. Run with `python -m hinglish_mt.rare_words run_b run_d run_e run_f`.
- Copy mechanism (pointer-generator, See et al. 2017): `copy` argument in `model.py` and `Config`. `Seq2Seq.word_scores` mixes the generated distribution with the attention weights scattered onto the source ids, weighted by a learned gate. 257 extra parameters. `forward` now returns log-probabilities and `batch_loss` computes the label-smoothed loss by hand (a test checks it equals PyTorch's). `decode.py` passes the attention weights and source ids on.
- `src/hinglish_mt/codemix.py`: IBM Model 1 dictionary from the first 150,000 rows of `synth_noisy.tsv`, then mixing. Run with `python -m hinglish_mt.codemix` (about 2 minutes). `noise.py` gained `noisy_romanise_word` so the mixed file reuses exactly the same noise.
- Configs `pretrain_noisy_copy.yaml`, `run_e.yaml`, `pretrain_mixed_copy.yaml`, `run_f.yaml`. Runs E and F added to `ablation.py`, `buckets.py` and `run_all.ps1`. `demo/app.py` now uses run F.
- Tests: `tests/test_codemix.py` and new cases in `tests/test_model.py` (136 tests pass in total).

Outputs: `data/processed/synth_mixed.tsv` and `synth_mixed_val.tsv`, `codemix_dictionary.json`, `codemix_report.json`, `outputs/analysis/codemix_sample.txt` and `codemix_dictionary_sample.txt`, `outputs/results/rare_words.md`, runs and results for `run_e`, `run_f` and their seeds 43 and 44.

Code-mixing settings and their effect (first 13,399 pairs):

| Measure | PHINC | Synthetic noisy | Synthetic mixed |
|---|---|---|---|
| English tokens | 20.8% | 2.9% | 13.6% |
| Sentences with both languages | 69.1% | 37.2% | 74.5% |
| Mean CMI, all sentences | 15.6 | 3.3 | 15.2 |

Dictionary: 12,775 Hindi words; the 40 most frequent words per side and words seen fewer than 5 times are left out; an English word is kept if its probability is at least 0.2 and it is not a grammar word (scikit-learn stop list plus 7 words). Mixing: 70% of sentences, 90% chance per eligible word, only if the English word is in the pair's English sentence. 13.0% of Hindi words were replaced, 62.1% of sentences got at least one English word.

Test split, seed 42, beam 4:

| Run | chrF | BLEU-4 | BLEU-1 | chrF, mean of 3 seeds | BLEU-4, mean of 3 seeds |
|---|---|---|---|---|---|
| D | 51.64 | 34.68 | 60.40 | 51.64 | 34.80 |
| E | 51.59 | 35.44 | 59.88 | 51.46 | 35.35 |
| F | 52.23 | 35.93 | 60.68 | 52.47 | 35.96 |

Paired bootstrap on the seed 42 outputs:

| Comparison | Metric | Difference | 95% interval | p-value |
|---|---|---|---|---|
| E minus D | chrF | -0.05 | -0.57 to +0.44 | 0.563 |
| E minus D | BLEU-4 | +0.77 | +0.16 to +1.43 | 0.014 |
| F minus E | chrF | +0.64 | +0.20 to +1.09 | 0.003 |
| F minus E | BLEU-4 | +0.49 | -0.10 to +1.06 | 0.048 |
| F minus D | chrF | +0.59 | +0.06 to +1.09 | 0.011 |
| F minus D | BLEU-4 | +1.26 | +0.61 to +1.91 | < 0.001 |
| F minus B | chrF | +5.53 | +4.96 to +6.09 | < 0.001 |

Copy rate for words that are the same in the tweet and the reference:

| Times in PHINC train | Words | Run B | Run D | Run E | Run F |
|---|---|---|---|---|---|
| never | 423 | 54.6% | 67.6% | 82.0% | 84.4% |
| 1 to 5 | 503 | 68.6% | 81.5% | 85.7% | 88.1% |
| 6 to 50 | 1,177 | 78.9% | 86.9% | 84.9% | 87.9% |
| more than 50 | 2,152 | 86.0% | 88.9% | 89.5% | 89.4% |

Notes for the report:
- The copy mechanism does what it was built for (unseen words kept: 67.6% to 82.0%) but does not move chrF. BLEU-4 rises significantly. On English-only tweets runs E and F reach the copy baseline (83.0 against 82.9); runs B and D were below it.
- Code-mixed synthetic data gives a small significant gain, largest on highly mixed tweets (+1.0 chrF over run D), smallest on Hindi-only tweets.
- Run F minus run E on BLEU-4 has p = 0.048 with an interval that includes zero (-0.10 to +1.06). Treat it as borderline. The chrF result is the clear one.
- Pre-training validation chrF: with copy 38.25 (without 38.41 to 38.49), so copying does not help on synthetic data, where little can be copied. The mixed model reaches 40.12 on its own mixed validation file; this is not comparable with the other two.
- The copy rate table is from one seed and has no significance test.
- Remaining failures: the model sometimes joins pieces of two words (`menopause k baad ki stage` gives "after stageopause stage"). About 16% of unseen words are still lost.
- Known flaw of the mixing: single words are replaced without grammar, so forms like `given gaya` and `gradually-gradually` occur. The dictionary was checked by reading samples only.
- Changes from the first plan: dictionary threshold 0.2 instead of 0.5 and 40 skipped words instead of 100 (the strict values left only 90 entries); grammar words excluded; several English candidates per Hindi word.
- The error-analysis sheets still compare runs B and D. They were not regenerated, because that would overwrite sheets that may be filled in.
- Not done: longer pre-training, more real data (LinCE), back-translation.

## Phase 9 — Application: sentiment analysis (done)

Reason: translation alone may look like a small result. This phase measures one use of the translator: an English sentiment tool reads our translation instead of the Hinglish tweet.

Built:
- `src/hinglish_mt/sentiment.py`. Run with `python -m hinglish_mt.sentiment` (about 20 minutes, 19 of them for translating 3,000 tweets with beam 4). `--score-only` repeats the scoring from the saved translations.
- Data: SentiMix (SemEval-2020 Task 9), Hinglish part, in `data/SentiMix/`. Taken from the public copy `RTT1/SentiMix` on Hugging Face. Train 14,000, test 3,000 (900 negative, 1,100 neutral, 1,000 positive, as published). The reader joins `@` and `#` to the next word and repairs links that the data splits into pieces.
- Sentiment tool: VADER (package `vaderSentiment`, added to `requirements.txt`), thresholds 0.05 and -0.05 from its authors, not tuned.
- Translator: run F, unchanged. No SentiMix test tweet occurs in PHINC train or validation (0 removed).
- `demo/app.py` shows the sentiment of the translation. `scripts/run_all.ps1` runs the step if the data is present.
- `error_sheets.py` no longer overwrites existing sheets unless `--force` is given (`run_all.ps1` calls it at the end).
- Tests: `tests/test_sentiment.py` (143 tests pass in total).

Outputs: `outputs/results/sentiment_test.tsv` (all translations), `sentiment.md`, `sentiment.json`.

| System | Accuracy | Macro F1 | F1 negative | F1 neutral | F1 positive |
|---|---|---|---|---|---|
| Always "neutral" | 36.7 | 17.9 | 0.0 | 53.7 | 0.0 |
| VADER on the Hinglish tweet | 47.5 | 45.1 | 31.3 | 42.3 | 61.6 |
| VADER on our translation (run F) | 49.2 | 46.7 | 40.7 | 37.8 | 61.7 |
| Classifier trained on 14,000 labelled Hinglish tweets | 67.6 | 68.1 | 70.7 | 61.1 | 72.6 |

Paired bootstrap, translated minus untranslated: macro F1 +1.6 (-0.3 to +3.5, p = 0.049), accuracy +1.7 (-0.1 to +3.4, p = 0.032). Borderline; both intervals include zero.

| Group | Tweets | Macro F1 untranslated | Macro F1 translated | Difference |
|---|---|---|---|---|
| Hindi only | 173 | 19.9 | 44.7 | +24.8 |
| Mixed, low (CMI up to 20) | 1,331 | 41.8 | 45.2 | +3.4 |
| Mixed, high (CMI above 20) | 1,325 | 49.7 | 47.9 | -1.8 |
| English only | 171 | 42.0 | 39.9 | -2.1 |

Notes for the report:
- Translation helps where the English tool knows none of the words (Hindi only) and hurts slightly where the tweet already has much English.
- Without translation VADER answers "neutral" by default for unknown words; this is right by accident for many neutral tweets, so neutral F1 falls after translation.
- The trained classifier is far ahead. The translation route is for tasks without labelled Hinglish data.
- The group table was added after seeing the overall result, but uses the groups already fixed in Phase 6. It has no significance test.
- Not done: translating only tweets that are mostly Hindi (the obvious next step), other sentiment tools, other applications.

## Phase 10 — Manual error analysis (done)

The two sheets were filled in and saved as `outputs/error_analysis/compare_b_d_annotated.csv` and `check_run_d_annotated.csv` (the empty sheets are kept beside them). Every row has a note.

Built: `src/hinglish_mt/error_summary.py` (`python -m hinglish_mt.error_summary`), which checks that every answer is one of the allowed values, counts them and writes `outputs/results/error_analysis.md` and `.json`. Tests in `tests/test_error_summary.py` (146 tests pass in total).

| Run D against run B, 100 tweets | Tweets |
|---|---|
| Run D better | 49 |
| Run B better | 25 |
| Same | 26 |

Sign test on the 74 tweets with a preference: p = 0.007.

| Main problem in run D, 100 tweets | Tweets |
|---|---|
| wrong meaning | 37 |
| none | 19 |
| wrong word | 14 |
| missing | 12 |
| grammar | 10 |
| untranslated | 6 |
| repetition | 2 |

Quality check, 50 other tweets: 10 correct, 40 need change.

Notes for the report:
- About one translation in five is fully correct. The main error is fluent English with the wrong meaning.
- One set of labels, no second annotator. The sheets judge run D, not run F.
- With this, nothing planned is open. Possible further work is listed in Section 10 of `DOCUMENTATION.md`.
