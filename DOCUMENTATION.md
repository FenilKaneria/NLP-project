# Hinglish to English Machine Translation for Social Media Text

Project documentation. Status: all seven phases of the implementation are complete, two extensions (runs E and F) were added afterwards, and the translator was then tested in one application, sentiment analysis (Sections 5.9 and 6.9). The manual error analysis is done (Section 6.10). The manual error analysis sheets are prepared but not yet filled in.

## 1. Summary

People on social media often write Hindi and English mixed in one sentence, with the Hindi typed in English letters. This is called Hinglish. Standard translation systems handle it poorly.

We build a system that translates Hinglish tweets into English. The model is a GRU encoder-decoder with attention, trained by us from random initial weights. No pre-trained translation model is used.

Our main question: **the real Hinglish data is very small. Can machine-made Hinglish, with realistic spelling mistakes added, make up for that?**

Short answer from our experiments: pre-training on machine-made Hinglish raises the main score (chrF) from 46.7 to 51.6, a statistically significant gain. Adding spelling mistakes to that data gave no further significant gain.

Two extensions followed. A copy mechanism raised the share of unseen words and names that are kept correctly from 67.6% to 82.0%. Mixing English words into the machine-made Hinglish raised chrF further to 52.2. Both are described in Section 5.8.

Last, we tested the translator in one application. An English sentiment tool was given our translations of 3,000 Hinglish tweets instead of the tweets themselves. Its macro F1 rose from 45.1 to 46.7 over all tweets (a borderline result), and from 19.9 to 44.7 on tweets written in Hindi only (Section 6.9).

A manual check of 150 test tweets confirms both the gain and the limit: the pre-trained model was judged better than the model without pre-training about twice as often as the reverse, but only about one translation in five is fully correct (Section 6.10).

## 2. Problem statement

Example input and the expected output:

| | |
|---|---|
| Hinglish | `mujhe tumhari aankho se peene do, waise bhi daaru ke liye paise nahi hai` |
| English | `let me drink from your eyes, i don't have money for alcohol` |

Three things make this hard:

1. **Little data.** The only public Hinglish-English corpus of tweets has about 13,000 sentence pairs. Translation models normally need hundreds of thousands.
2. **No fixed spelling.** Hindi typed in English letters has no standard. The word `नहीं` appears as `nahi`, `nhi`, `nahin` and `nai`.
3. **Two languages in one sentence.** The model must translate the Hindi words and keep the English words.

### 2.1 Applications

Translation is not the end goal here. It is a first step that makes Hinglish text usable by people and by software that only work with English. Most language tools (sentiment analysis, search, content filters, summarisers) are built for standard English and fail on Hinglish. Translating the text first lets all of these tools be used without building each one again for Hinglish.

| Area | Use | Who benefits |
|---|---|---|
| Customer support | Complaints and chat messages written in Hinglish are translated, so they can be sorted, routed and answered by English-language systems and staff | Banks, telecom and e-commerce companies |
| Content moderation | Abusive or harmful posts written in Hinglish are translated, so existing English filters and human reviewers can check them | Social media platforms |
| Opinion and sentiment analysis | Product reviews and tweets in Hinglish are translated, then analysed with existing English sentiment tools | Companies, market researchers |
| Public feedback and emergencies | Posts by citizens during floods, elections or about public services are translated, so officials and relief teams can read them quickly | Government offices, disaster response teams |
| Search and recommendation | A query such as `sasta phone kaha milega` is translated, so it can be matched with English product pages | Search engines, shopping sites |
| Reading across languages | A person who does not know Hindi can read Hinglish posts and messages | Non-Hindi speakers in India and abroad |
| Chatbots and voice assistants | Hinglish input is translated before it reaches an assistant that understands only English | Makers of assistants |

Parts of the project are also useful on their own, outside translation:

| Part | Use on its own |
|---|---|
| Devanagari to English-letter converter (`romanise.py`) | Typing tools, search that matches `ghar` with `घर`, converting Hindi text for people who cannot read Devanagari |
| Spelling-noise generator (`noise.py`) | Making other models (spell checkers, classifiers) robust to the many spellings of one Hindi word |
| Language tagger and Code-Mixing Index (`langid.py`, `cmi.py`) | Measuring how mixed any text is; finding the Hinglish posts in a large collection |
| Code-mixed data generator (`codemix.py`) | Creating training data for any Hinglish task where real data is short, for example Hinglish sentiment analysis |

The method itself is not tied to Hindi. Many language pairs have the same problem: mixed writing in English letters and very little parallel data, for example Tamil-English, Bengali-English and Marathi-English. The steps used here (convert the script, add spelling noise, mix in English words, pre-train, then fine-tune on the small real set) can be repeated for these pairs wherever a corpus in the native script exists.

**One application was built and measured.** Sentiment analysis, the third row of the first table, was tested on 3,000 labelled Hinglish tweets (method in Section 5.9, results in Section 6.9). The other rows are possible uses that we did not test.

**What we claim and what we do not.** Our model is a research prototype. With a chrF of 52.2 it is not accurate enough to be used in a product as it is (Section 8). The contribution of the project is the method and the measurements: we show, with controlled experiments, which steps help when real data is very small. A product would use the same steps with more data and a larger model.

## 3. Objectives

1. Build a complete Hinglish to English translation pipeline: data cleaning, model, training, scoring.
2. Create synthetic Hinglish training data from an existing Hindi-English corpus.
3. Measure, with controlled experiments, how much each of these helps:
   - splitting words into word pieces,
   - pre-training on synthetic data,
   - adding spelling noise to the synthetic data.
4. Report results honestly, including a simple baseline and statistical significance.

## 4. Datasets

| Dataset | Content | Size | Our use |
|---|---|---|---|
| PHINC (Srivastava and Singh, 2020) | Hinglish tweets with human English translations | 13,738 pairs | Fine-tuning, validation, testing |
| IIT Bombay English-Hindi corpus (Kunchukuttan et al., 2018) | Hindi in Devanagari script with English | 1,659,083 pairs | Source for synthetic Hinglish |
| SentiMix, SemEval-2020 Task 9 (Patwa et al., 2020) | Hinglish tweets labelled positive, negative or neutral by people | 14,000 train, 3,000 validation, 3,000 test | Only for the sentiment application (Section 5.9). Never used to train the translator |

## 5. Method

### 5.1 Overview

```
IIT Bombay corpus (Hindi in Devanagari + English)
        |  filter, convert to English letters, add spelling noise
        v
Synthetic Hinglish-English pairs (300,000)
        |  pre-training
        v
     Model  ---- fine-tuning on PHINC train (10,719 real pairs) ---->  Final model
                                                                           |
                                                              scored on PHINC test (1,341 pairs)
```

### 5.2 Data cleaning (Phase 1)

- Repaired broken characters.
- Replaced every @mention, #hashtag and link with a placeholder (`<user>`, `<hashtag>`, `<url>`). These must not be translated; the model copies the placeholder and the original is put back afterwards.
- Converted all text to lowercase.
- Removed unusable rows: 13 with Devanagari in the source, 217 with a broken translation, 109 duplicates.
- Split the remaining 13,399 pairs with a fixed random seed:

| Part | Pairs | Purpose |
|---|---|---|
| Train | 10,719 | The model learns from it |
| Validation | 1,339 | Decides when training stops |
| Test | 1,341 | Used once, for the final scores |

No sentence appears in more than one part.

### 5.3 Study of the data (Phase 2)

We wrote a rule-based tagger that marks each word as Hindi, English or other, using an English dictionary and a hand-made list of Hindi words that look English (`hai`, `ke`, `se`). Main findings on PHINC:

| Measure | Value |
|---|---|
| Hindi / English / other words | 52.1% / 20.8% / 27.1% |
| Tweets that contain both languages | 69.1% |
| Code-Mixing Index (Das and Gambäck, 2014), mean | 15.6 |
| Test tweets with at least one word never seen in training | 53.3% |
| Hindi words that belong to a word with more than one spelling | 60.8% |

The last two rows motivate the design: unseen words call for word pieces, and spelling variety calls for spelling noise.

### 5.4 Synthetic data (Phase 3)

1. **Filtering.** From the IIT Bombay corpus we kept pairs of 5 to 40 words, removed duplicates and software menu text. 658,759 pairs remained; 300,000 were sampled.
2. **Romanisation.** Our own rule-based program converts Devanagari into English letters the way people type: `करना` becomes `karna`, `काम` becomes `kaam`.
3. **Spelling noise.** Random changes that copy real typing habits:

| Change | Example |
|---|---|
| Common short forms | `nahi` to `nhi`, `kyon` to `kyu` |
| Dropped vowel | `karna` to `krna` |
| Long or short vowel | `shadi` to `shaadi` |
| Dropped final nasal sound | `hain` to `hai` |
| Similar-sounding letters | `phir` to `fir` |
| Stretched last letter | `jaldi` to `jaldiii` |

How often each change happens was set by counting in the PHINC training tweets only.

Two files were written from the same 300,000 sentences: one clean and one noisy.

**Check of the result:**

| Measure | Real tweets | Synthetic, clean | Synthetic, noisy |
|---|---|---|---|
| Hindi words with more than one spelling | 60.8% | 33.0% | 64.6% |
| Real Hindi words (in running text) also found in the synthetic data | | 84.6% | 90.2% |

The noise brings the spelling variety close to that of real tweets.

### 5.5 Model (Phase 4)

| Part | Choice |
|---|---|
| Type | Encoder-decoder with additive attention (Bahdanau et al., 2015) |
| Encoder | 2-layer bidirectional GRU, reads the Hinglish sentence |
| Decoder | 2-layer GRU, writes the English sentence one unit at a time |
| Attention | At each step the decoder looks at the most relevant input words |
| Sizes | Hidden size 256, embedding size 256, about 5.8 million parameters |
| Text units | Whole words (run A) or 8,000 word pieces made with BPE (Sennrich et al., 2016) |
| Training | Adam optimiser, dropout 0.3, label smoothing 0.1, gradient clipping |
| Stopping | Training stops when the validation score has not improved for 5 epochs |
| Translating | Beam search with 4 candidates |
| Hardware | One laptop GPU (NVIDIA RTX 3050, 6 GB) |

### 5.6 Experiments

Four models are trained. Neighbouring runs differ in one thing only, so each comparison isolates one factor.

| Run | Training data | Text units | Spelling noise |
|---|---|---|---|
| A | Real data only | Whole words | No |
| B | Real data only | Word pieces | No |
| C | Synthetic, then real | Word pieces | No |
| D (proposed) | Synthetic, then real | Word pieces | Yes |

- A against B: do word pieces help?
- B against C: does pre-training on synthetic data help?
- C against D: does the spelling noise help?

### 5.7 Evaluation

Model output is compared with the human translation on the test part. All scores range from 0 to 100; higher is better. We use the standard sacreBLEU tool (Post, 2018).

| Score | What it counts |
|---|---|
| chrF (main score; Popović, 2015) | Matching short letter sequences. Gives partial credit for a nearly correct word, which suits noisy text |
| BLEU-4 (Papineni et al., 2002) | Matching sequences of 1 to 4 words |
| BLEU-1 | Matching single words |

Two safeguards:

- **Copy baseline.** We also score the tweet returned unchanged, with no translation. Tweets contain English words, names and placeholders, so this already scores well. A model is useful only if it beats it.
- **Significance test.** Paired bootstrap resampling (Koehn, 2004) with 1,000 samples, and repeated runs with three random seeds, to make sure differences are not due to chance.

### 5.8 Two extensions (runs E and F)

After the four planned runs we looked at the weak points of run D and added two extensions. Each one is a new run that differs from the run before it in one thing only.

| Run | Pre-training data | Copy mechanism | Question |
|---|---|---|---|
| D | Noisy synthetic | No | |
| E | Noisy synthetic | Yes | E against D: does copying help? |
| F | Noisy and code-mixed synthetic | Yes | F against E: does code-mixed synthetic data help? |

**Weak point 1: rare words and names.** Some words must pass through unchanged, for example names and English words. We counted, on the test part, how often run D keeps a word that is the same in the tweet and in the reference translation (Section 6.8). For words it saw often in training it keeps 88.9%. For words it never saw it keeps only 67.6%. The model builds every word from word pieces, and for an unknown word it picks wrong pieces: `menopause` becomes `menops`, `casanova` becomes `casova`.

**Extension 1: copy mechanism (run E).** We follow the pointer-generator idea of See et al. (2017). At each step the decoder now has two ways to produce a word piece:

1. generate it from the vocabulary, as before;
2. copy a piece from the input tweet. The attention weights, which the model already computes, say which input piece to copy.

A small learned gate gives a number between 0 and 1 that decides how much weight each way gets. The final probability of a piece is `gate × (probability from generating) + (1 − gate) × (attention weight on that piece in the input)`. Hinglish and English share one vocabulary in our model, so a copied piece needs no extra lookup. The change adds 257 parameters.

**Weak point 2: the synthetic data is not mixed.** It has 2.9% English words; real tweets have 20.8% (Section 5.4). The model meets mixed sentences only in the small real data.

**Extension 2: code-mixed synthetic data (run F).** We replace some Hindi words in the synthetic sentences by English words, in two steps.

1. *Word dictionary.* We learn which Hindi word goes with which English word from 150,000 of our own sentence pairs, with IBM Model 1 (Brown et al., 1993). This classic method starts by treating all word pairs in a sentence pair as equally likely and then improves the estimate over five rounds. Very frequent words and very rare words are left out, and so are English grammar words such as `can`, `when` and `the`, because people rarely mix those in. The dictionary has 12,775 Hindi words. Examples: `सरकार` to `government`, `पानी` to `water`, `घर` to `home` or `house`.
2. *Mixing.* In 70% of the sentences, a Hindi word is replaced by its English word with a chance of 90%, but only if that English word occurs in the English translation of the same sentence. So the English word is always the right one for that sentence. All other words keep exactly the noisy spelling they have in run E's data.

| | Hinglish | English |
|---|---|---|
| Before | `rashtrapti ne kaha ki sarkaar ne ... abhiyan aarambh kiya hai` | `the president said the government has initiated the ... campaign` |
| After | `president ne kaha ki government ne ... campaign aarambh kiya hai` | same |

**Check of the result** (first 13,399 pairs of each file):

| Measure | Real tweets | Synthetic, noisy | Synthetic, code-mixed |
|---|---|---|---|
| English words | 20.8% | 2.9% | 13.6% |
| Sentences that contain both languages | 69.1% | 37.2% | 74.5% |
| Code-Mixing Index, mean | 15.6 | 3.3 | 15.2 |

The two rates (70% and 90%) were set so that the Code-Mixing Index comes close to that of the real training tweets. In total 13.0% of the Hindi words were replaced.

### 5.9 Application: sentiment analysis of Hinglish tweets

Sentiment analysis decides whether a text is positive, negative or neutral. Good tools exist for English. We test whether our translator makes such a tool usable on Hinglish.

| Step | What happens |
|---|---|
| 1 | The Hinglish tweet is translated to English by run F. The translator is not changed or retrained |
| 2 | VADER (Hutto and Gilbert, 2014) reads the translation and gives a score from -1 to +1 |
| 3 | A score of 0.05 or more is positive, -0.05 or less is negative, anything between is neutral. These are the limits given by the authors of VADER. We did not tune them |

VADER is a list of English words with a sentiment value for each, plus a few rules (for example for "not" and "very"). It is not a neural network and not a translation model, so the rule of the project, to train the translator from scratch, is kept.

**Data.** The SentiMix test set: 3,000 Hinglish tweets with labels given by people (900 negative, 1,100 neutral, 1,000 positive). We checked that none of these tweets occurs in the data the translator was trained on; none did.

**Systems compared.**

| System | Uses labelled Hinglish tweets? | Why it is included |
|---|---|---|
| Always answer "neutral" (the most frequent label) | No | The lowest sensible score |
| VADER on the Hinglish tweet as it is | No | What the English tool does without translation |
| VADER on our translation | No | Our application |
| A classifier trained on the 14,000 labelled SentiMix training tweets (character n-grams, logistic regression) | Yes | Shows what is possible when labelled Hinglish data exists |

The question is the difference between the second and the third row. Both use exactly the same tool, so any difference comes from the translation. It is tested with the paired bootstrap of Section 5.7.

**Scores.** Accuracy is the share of tweets with the right label. Macro F1 is the mean of the F1 scores of the three labels, so each label counts equally.

## 6. Results

### 6.1 Main table

Test part, 1,341 tweets:

| System | chrF | BLEU-4 | BLEU-1 |
|---|---|---|---|
| Copy baseline (no translation) | 41.83 | 29.29 | 45.72 |
| Run A: real data, whole words | 41.50 | 28.49 | 54.45 |
| Run B: real data, word pieces | 46.71 | 30.38 | 54.28 |
| Run C: clean synthetic, then real | 51.72 | 34.42 | 59.76 |
| Run D: noisy synthetic, then real | 51.64 | 34.68 | 60.40 |
| Run E: as D, with copy mechanism | 51.59 | 35.44 | 59.88 |
| Run F: as E, synthetic data code-mixed | 52.23 | 35.93 | 60.68 |

### 6.2 Are the differences real?

Each comparison was tested with paired bootstrap resampling (1,000 samples). A p-value below 0.05 means the difference is unlikely to be chance.

| Question | Comparison | chrF difference | p-value | Answer |
|---|---|---|---|---|
| Do word pieces help? | B minus A | +5.21 | < 0.001 | Yes |
| Does pre-training help? | C minus B | +5.01 | < 0.001 | Yes |
| Does spelling noise help? | D minus C | -0.07 | 0.597 | No significant difference |
| Is the proposed model better than real data alone? | D minus B | +4.93 | < 0.001 | Yes |
| Does the copy mechanism help? | E minus D | -0.05 | 0.563 | Not on chrF. On BLEU-4: +0.77, p = 0.014 |
| Does code-mixed synthetic data help? | F minus E | +0.64 | 0.003 | Yes. On BLEU-4: +0.49, p = 0.048 |
| Are both extensions together better than run D? | F minus D | +0.59 | 0.011 | Yes. On BLEU-4: +1.26, p < 0.001 |

### 6.3 Repeating the runs

Runs B to F were each trained three times with different random seeds.

| Run | chrF, mean of 3 | chrF, lowest to highest | BLEU-4, mean of 3 |
|---|---|---|---|
| B | 45.99 | 45.29 to 46.71 | 30.56 |
| C | 51.66 | 51.25 to 52.00 | 34.40 |
| D | 51.64 | 51.56 to 51.73 | 34.80 |
| E | 51.46 | 51.34 to 51.59 | 35.35 |
| F | 52.47 | 52.23 to 52.61 | 35.96 |

The gap between run B and the pre-trained runs (about 5.7 chrF) is much larger than the spread between seeds (about 1.4 for run B), so it is not luck. For runs C to F the seed changes only the fine-tuning; all three start from the same pre-trained model. Run F is above run D in every seed, on both scores.

### 6.4 What the results mean

1. **Word pieces help** (A to B: +5.2 chrF). With whole words, 7.8% of validation words are unknown to the model; with word pieces, 0.04%. Run A does not even beat the copy baseline on chrF.
2. **Pre-training on synthetic data helps** (B to C: +5.0 chrF, +4.0 BLEU-4). This holds although the synthetic data is formal text with almost no English words. Our main objective is met: the proposed model (D) is clearly better than training on real data alone (B).
3. **Spelling noise did not give a significant gain** (C to D). Run D is slightly higher on BLEU-4 in all three seeds, but the test does not confirm the difference. We report this as "no significant difference".
   - Likely reason (not tested): word pieces already break unusual spellings into known parts, and fine-tuning on real tweets teaches the real spellings anyway. The noise did make the synthetic vocabulary closer to real tweets (Section 5.4), but that did not turn into better translations.
4. **The copy mechanism fixes rare words, not the overall score** (D to E). chrF does not change. BLEU-4 rises by 0.8, which is significant, and far more unseen words and names are kept (Section 6.8). chrF gives partial credit for a nearly right word, so turning `casova` into `casanova` changes it little.
5. **Code-mixed synthetic data helps** (E to F: +0.6 chrF, significant). The gain is largest on the most mixed tweets (Section 6.5), which is where the old synthetic data differed most from real tweets.
6. **Absolute quality is still modest.** The best run (F) reaches chrF 52.2, about 10 points above doing nothing (copy baseline), and many translations still miss the meaning. The two extensions add about 0.6 chrF and 1.3 BLEU-4 to run D. They reduce the limits; they do not remove them.

### 6.5 Results by how mixed the tweet is

We grouped the test tweets by their Code-Mixing Index (CMI) and scored each group separately. The table shows chrF.

| Group | Tweets | Copy baseline | Run B | Run D | Run E | Run F | D minus B | F minus D |
|---|---|---|---|---|---|---|---|---|
| Hindi only | 351 | 23.94 | 36.67 | 40.92 | 40.97 | 41.28 | +4.25 | +0.35 |
| Mixed, low (CMI up to 20) | 497 | 40.82 | 45.78 | 50.90 | 50.77 | 51.14 | +5.12 | +0.24 |
| Mixed, high (CMI above 20) | 451 | 47.08 | 49.02 | 54.42 | 54.23 | 55.43 | +5.40 | +1.01 |
| English only | 42 | 82.91 | 79.89 | 81.13 | 83.04 | 82.97 | +1.24 | +1.84 |
| All | 1,341 | 41.83 | 46.71 | 51.64 | 51.59 | 52.23 | +4.93 | +0.59 |

- Pre-training helps every group that contains Hindi, by 4 to 5 chrF.
- Hindi-only tweets are the hardest. Mixed tweets score higher mainly because their English words can simply be copied, as the rising copy baseline shows.
- On tweets that are already English, runs B and D are below copying: they slightly damage text that needed no translation. With the copy mechanism (runs E and F) this damage is gone; the score equals that of copying. This group has only 42 tweets.
- Code-mixed synthetic data (run F) helps most on highly mixed tweets (+1.0 chrF over run D) and little on Hindi-only tweets, as one would expect.

### 6.6 What the model learned about words

Hinglish and English words share one table of word vectors in our model. Words the model treats as similar end up close together. For a few common words we looked at the eight closest neighbours in run D. The table shows a selection; the full lists are in `outputs/analysis/embedding_neighbours.md`.

| Word | Selected close neighbours (similarity from 0 to 1) |
|---|---|
| `nahi` | `nhi` 0.89, `nahin` 0.87, `nai` 0.77 |
| `bahut` | `bohot` 0.84, `bahot` 0.83, `bht` 0.76 |
| `sarkar` | `sarkaar` 0.80, `sarkari` 0.53, `government` 0.44 |
| `ghar` | `gaanv` 0.44, `home` 0.44, `house` 0.40 |
| `paise` | `dhan` 0.52, `rupaye` 0.43, `money` 0.37 |

The model has learned, without being told, that different spellings are the same word and that `ghar` is related to `home`. In run B (real data only) the same pattern exists but is weaker (`nahi` and `nhi`: 0.73). This is an illustration with a few hand-picked words, not a measurement.

### 6.7 Demo

A small web page (`streamlit run demo/app.py`) translates a typed sentence, shows the sentiment of the translation (Section 6.9), colours each input word by language and shows the CMI. It uses run F, the best run. Sample outputs:

| Input | Run B (real data only) | Run D | Run F (best) |
|---|---|---|---|
| `ye sarkar kuch nahi kar rahi hai` | this is not able to sarkar the government. | this government is not doing anything | this government is not doing anything. |
| `aaj office me bahut kaam tha, i am so tired` | today i was in work in office office, i am so tired | today i had a lot of work in office, i am so tired | today i have a lot of work in office, i am so tired. |
| `kal movie dekhne chalein? mujhe bahut excited feel ho raha hai` | yesterday movie? i feel very excited. | go to watch movie tomorrow? i am feeling very excited | go to see the movie tomorrow? i am feeling very excited. |
| `mujhe nahi pata tha ki wo itna accha gaata hai` | i didn't know that that that is better that you are good that | i didn't know that he gets very good | i didn't know that he is so good. |

Two errors remain in these examples. In the second row run F writes "have" where the tweet says "had". In the last row no run translates the word `gaata` (sings).

### 6.8 Rare words and names

Some words of a tweet must appear unchanged in the translation: names, and English words the writer already used. For every word that is the same in the tweet and in the reference translation, we checked whether it is in the model's output. The words are grouped by how often they occur in the real training data.

| Times in training | Words | Run B | Run D | Run E | Run F |
|---|---|---|---|---|---|
| Never | 423 | 54.6% | 67.6% | 82.0% | 84.4% |
| 1 to 5 | 503 | 68.6% | 81.5% | 85.7% | 88.1% |
| 6 to 50 | 1,177 | 78.9% | 86.9% | 84.9% | 87.9% |
| More than 50 | 2,152 | 86.0% | 88.9% | 89.5% | 89.4% |

- Without copying, the less often a word was seen, the more often it is lost. Run D keeps 88.9% of frequent words but 67.6% of unseen ones.
- With the copy mechanism (run E) unseen words are kept 82.0% of the time. The gap between unseen and frequent words shrinks from 21 points to under 8.
- Code-mixed synthetic data (run F) adds a little more. During pre-training the model now sees English words in the input that must be kept.

Examples from the test part:

| Tweet | Run D | Run F |
|---|---|---|
| `... hum to casanova bol rahe the` | ... we are calling casova | ... we are saying casanova |
| `zyada hosiery mat dikhao` | don't show more hopes | don't show more hosiery |
| `... - baburao ganpatrao apte` | ... - baburao ganeso apte | ... - baburao ganpatrao apte |
| `menopause k baad ki stage` | after the menops | after stageopause stage |

The last row is a failure that remains: the model copies pieces of two different words and joins them. About 16% of unseen words are still lost. This table is a count on one seed, not a significance test.

### 6.9 Application: sentiment analysis

Scores on the 3,000 SentiMix test tweets:

| System | Accuracy | Macro F1 | F1 negative | F1 neutral | F1 positive |
|---|---|---|---|---|---|
| Always "neutral" | 36.7 | 17.9 | 0.0 | 53.7 | 0.0 |
| VADER on the Hinglish tweet | 47.5 | 45.1 | 31.3 | 42.3 | 61.6 |
| **VADER on our translation (run F)** | **49.2** | **46.7** | **40.7** | 37.8 | 61.7 |
| Classifier trained on 14,000 labelled Hinglish tweets | 67.6 | 68.1 | 70.7 | 61.1 | 72.6 |

Is the gain from translation real?

| Metric | Translated minus untranslated | 95% interval | p-value |
|---|---|---|---|
| Macro F1 | +1.6 | -0.3 to +3.5 | 0.049 |
| Accuracy | +1.7 | -0.1 to +3.4 | 0.032 |

Over all tweets the gain is small and borderline: the p-values are just below 0.05, but both intervals include zero. We do not claim a clear gain over all tweets.

The picture is clearer when the tweets are grouped by how mixed they are (the same groups as in Section 6.5, fixed before this experiment):

| Group | Tweets | Macro F1, untranslated | Macro F1, translated | Difference |
|---|---|---|---|---|
| Hindi only | 173 | 19.9 | 44.7 | +24.8 |
| Mixed, low (CMI up to 20) | 1,331 | 41.8 | 45.2 | +3.4 |
| Mixed, high (CMI above 20) | 1,325 | 49.7 | 47.9 | -1.8 |
| English only | 171 | 42.0 | 39.9 | -2.1 |
| All | 3,000 | 45.1 | 46.7 | +1.6 |

What this shows:

1. **Translation helps most where the English tool is blind.** On tweets written in Hindi only, VADER without translation is almost as weak as always answering "neutral", because it knows none of the words. With translation its macro F1 more than doubles.
2. **Translation does not help where there is already enough English.** In highly mixed tweets the English words alone often carry the sentiment, and our translation errors then cost more than the translation adds.
3. **Negative tweets gain the most.** F1 for the negative label rises from 31.3 to 40.7. Of the 900 negative tweets, 309 are found after translation against 209 before.
4. **Neutral tweets lose.** F1 for neutral falls from 42.3 to 37.8. Without translation VADER says "neutral" whenever it knows no word, which is right by accident for many neutral tweets. After translation it sees more sentiment words, some of them wrong.
5. **Labelled data beats translation when it exists.** The classifier trained on 14,000 labelled Hinglish tweets is far ahead (68.1). The translation route is for the common case where no labelled Hinglish data exists for a task.

Examples (handles shortened to `@user`):

| Tweet | Our translation | True label | Untranslated | Translated |
|---|---|---|---|---|
| `@user You are stalking me bro . Just shut up and nikal . Le . Faltu logo ki mention mein jagah nahi haii` | `@user you are stalking me bro . just shut up and nikal . take . there is no place in the mention of false people` | negative | neutral | negative |
| `RT @user Wah bhai Shandaar jabab` | `retweet @user wow brother shan shan jabab` | positive | neutral | positive |
| `@user Nhi yaar yahan tw 1year walo k bhe khatam ho gye thy` | `@user there is no problem, the news of twit people here has gone over.` | neutral | neutral | positive |

The last row is a failure: the translation is wrong, and the wrong words carry sentiment. The second row shows that a translation need not be fully right to help; one correct word (`wow`) was enough.

The demo (Section 6.7) now also shows the sentiment of the translation. For example `service bahut kharab hai, mera paisa barbaad ho gaya` gives "the service is very bad, my money is ruined." and the label negative; VADER on the Hinglish sentence itself says neutral.

All figures are in `outputs/results/sentiment.md`, and all 3,000 translations in `outputs/results/sentiment_test.tsv`.

### 6.10 Manual error analysis

Automatic scores compare words with one reference translation. To see what the translations are really like, test tweets were read and labelled by hand. The tweets were picked at random with a fixed seed.

**Run D against run B, 100 tweets.** Both translations were shown side by side.

| Answer | Tweets |
|---|---|
| Run D is better | 49 |
| Run B is better | 25 |
| Equally good or equally bad | 26 |

Among the 74 tweets where one run was preferred, run D wins about two times out of three. A sign test gives p = 0.007, so this is unlikely to be chance. The reading by hand agrees with the automatic scores: pre-training on synthetic data helps.

**Main problem in the translation of run D, same 100 tweets.**

| Problem | Tweets | Example (tweet, then output of run D) |
|---|---|---|
| Wrong meaning: fluent English that says something else | 37 | `baadlon mein chhup raha hai chaand kyu?` gives "why are you shouting in the clouds?" (the moon is hiding, nobody shouts) |
| None: the meaning is correct | 19 | |
| Wrong word: one word is mistranslated | 14 | `ek hi aankh mein?` gives "in the same eyes?" (in one eye only) |
| Missing: part of the tweet is left out | 12 | `phd ke final year me kanhaiya` gives "kanhaiya" |
| Grammar: the meaning is right, the English is broken | 10 | `woh isiliye kyunki tu unko bola ke tu chelsea ka fan hai` gives "that's why because you say that you are fan of chelsea" |
| Untranslated: a Hindi word is copied | 6 | `husn o ishq ki raahon mein` gives "husse o in the path of ishq" |
| Repetition | 2 | `dhoni bhai ek din ball tod hi denge!` gives "dhoni brother will break a ball ball." |

**Quality check of run D, 50 other tweets.**

| Verdict | Tweets |
|---|---|
| Correct: a reader understands the same as from the tweet | 10 (20%) |
| Needs change | 40 (80%) |

What this shows:

1. **Only about one translation in five is fully correct** (19 of 100 in the first sheet, 10 of 50 in the second). This is the plain meaning of a chrF near 52, and it is why we call the system a research prototype.
2. **The main error is wrong meaning, not broken English.** In 37 of 100 tweets the output reads well but says something else. This is the typical weakness of a model with little real data: it has learned to write English better than it has learned Hindi words. It confirms the limit stated in Section 9.6.
3. **Single-word errors come second** (wrong word and untranslated, 20 tweets together). These are the errors the copy mechanism and the code-mixed data of runs E and F aim at.

Limits of this analysis: there is one set of labels and no second annotator, so agreement between annotators could not be measured. The sheets were prepared before runs E and F existed, so they judge run D, not run F. With 100 and 50 tweets the shares are rough; 20% correct out of 50 could be anywhere from about 10% to 34%.

The filled-in sheets are `compare_b_d_annotated.csv` and `check_run_d_annotated.csv` in `outputs/error_analysis/`; each row has a note explaining the label. The counts are in `outputs/results/error_analysis.md`.

## 7. What we built ourselves

| Written by us | Taken from libraries |
|---|---|
| Data cleaning and placeholder handling | PyTorch (tensors, GRU layer, optimiser) |
| Language tagger and Code-Mixing Index | SentencePiece (learning the word pieces) |
| Devanagari to English-letter converter | sacreBLEU (chrF and BLEU scores) |
| Spelling-noise generator | wordfreq (English word list) |
| Encoder, decoder and attention | scikit-learn (list of English grammar words) |
| Copy mechanism | |
| Word dictionary (IBM Model 1) and code-mixing of the synthetic data | |
| Training loop, greedy and beam search | |
| Significance test, result tables, rare-word check | |
| Sentiment application: reading the data, comparison of systems, significance test | vaderSentiment (English sentiment word list and rules) |

The code has 146 automatic tests.

## 8. Limitations

- **Synthetic data is only partly like real mixing.** In runs C to E it has almost no English words (2.9%, against 20.8% in real tweets). Run F raises this to 13.6% by replacing Hindi words with English ones. But our method replaces single words and knows no grammar, so some results are forms no person would write, for example `given gaya` for "was given". Real writers also switch whole phrases, which we do not imitate.
- **The word dictionary was not checked against a reference.** It was built automatically (IBM Model 1) and checked only by reading samples.
- **Rare words are still a weak point.** With the copy mechanism about 16% of unseen words are still lost, and Hindi words that are rare in the training data are still often mistranslated. Copying only helps for words that stay the same in English.
- **Different topic and style.** The IIT Bombay corpus is formal text (government, religious, technical); tweets are informal and shorter.
- **Rule-based romaniser.** It makes mistakes on some words (`अनुमति` becomes `anumti`) and spells English loanwords by sound (`बैंक` becomes `baink`, not `bank`). It was checked with test words and by reading samples, not against a reference dataset.
- **Language tagger has no accuracy figure,** because we have no hand-labelled data to test it on.
- **One reference translation per tweet,** and some PHINC references are themselves poor. Scores therefore understate quality somewhat for every system equally.
- **Small model and limited hardware.** A larger model or a Transformer might do better; this was outside our scope.
- **Pre-training was stopped early.** It ran for 10 passes while the score was still rising slowly, so the gain from pre-training may be understated.
- **The application gain is small.** In sentiment analysis the translator gives a clear gain only for tweets with little or no English (Section 6.9). Over all tweets the gain is borderline, and for highly mixed tweets translation makes the result slightly worse. A sensible system would translate only tweets that are mostly Hindi; we did not build or test this.
- **The sentiment test has its own weak points.** It uses one tool (VADER) and one dataset. Some SentiMix labels are doubtful, and some of its tweets are not Hinglish at all. The data was taken from a public copy; its label counts match those published by the task organisers.

## 9. Questions on the design choices

### 9.1 Did we convert the Devanagari text into English?

No. We changed the **script**, not the **language**.

Every row of the IIT Bombay corpus already has both sides:

| Side | Example |
|---|---|
| Hindi, in Devanagari | `वह घर जा रहा है` |
| English translation (given in the corpus) | `he is going home` |

We did not translate anything. We only rewrote the Hindi side in English letters, the way people type on a phone:

`वह घर जा रहा है` becomes `wo ghar ja raha hai`

The sentence is still Hindi. The result is a pair that looks like the real tweet data:

`wo ghar ja raha hai` paired with `he is going home`

### 9.2 How is the script converted?

By our own rules in `romanise.py`. No library is used.

| Step | Rule | Example |
|---|---|---|
| 1 | Each Devanagari letter is mapped to English letters | `क` to `k`, `ख` to `kh`, `ी` to `i` |
| 2 | Every Hindi consonant carries a hidden "a" sound. It is removed where people do not pronounce it: at the end of a word and in some middle positions | `करना` to `karna`, not `karana` |
| 3 | Long vowels are spelled the way people usually type them | `काम` to `kaam`, `ठीक` to `theek` |
| 4 | The nasal dot becomes `n` or `m` | `हैं` to `hain`, `में` to `mein` |
| 5 | A short list fixes common words where the rules give an unusual spelling | `वह` to `wo`, `नहीं` to `nahi` |

For the noisy version, `noise.py` then adds spelling changes such as `nahi` to `nhi` and `karna` to `krna`.

### 9.3 The corpus has 1.66 million pairs. Why use only 300,000?

- After filtering, 658,759 pairs are usable. The rest were removed: too short or too long (739,368), duplicates (201,789), software menu text and other unusable rows.
- 300,000 was chosen to fit the training time on one laptop GPU. One pass over 300,000 pairs takes about 4 minutes, and we pre-train two models for 10 passes each.
- Using all 658,759 pairs is possible and would take about 3 hours in total. But data size is not the current limit: the model is small (about 5.8 million parameters) and its score was still rising at the tenth pass. More passes or a larger model would be the first thing to try.

### 9.4 The synthetic data has no mixed Hindi-English sentences. How can it help?

A tweet has two kinds of words, and the synthetic data helps with the harder kind.

| Part of a tweet | Share of words | What the model must do | Where it learns this |
|---|---|---|---|
| Hindi words in English letters | 52% | Learn the meaning (`ghar` is home, `nahi` is not) and change the Hindi word order into English word order | Synthetic data (300,000 pairs) and real data |
| English words | 21% | Mostly copy them | Real data (10,719 pairs) |

Two more benefits:

- The model reads 300,000 English sentences during pre-training, against 10,719 in the real data. This helps it write fluent English.
- The noisy synthetic data contains 90.2% of the Hindi words (counted in running text) that occur in the real training tweets.

This was not assumed; it was measured. Run C, pre-trained on the synthetic data, scores 5.0 chrF above run B, which saw real data only (Section 6). So the synthetic data helps even without mixed sentences.

### 9.5 Should a different dataset be used instead?

No better dataset is available. This shortage is the core difficulty of the task.

| Option | Why it does not replace our approach |
|---|---|
| A large real Hinglish-English corpus | None exists. PHINC, with 13,738 pairs, is the largest for tweets |
| LinCE Hinglish translation set | Real mixed text, but only about 8,000 pairs. It could be added to fine-tuning later |
| Other large Hindi-English corpora | Hindi is in Devanagari, the same situation as the IIT Bombay corpus |

Creating synthetic data is therefore the usual way to work on this problem.

**Improvement made.** The weakness of our first synthetic data was that it contained almost no English words. Run F fixes part of this by replacing some Hindi words in the synthetic sentences with their English equivalents (Section 5.8). It is significantly better than run E.

### 9.6 The model gets the topic right but the meaning wrong, and fails on rare words. What can be done?

These are two different problems, and we measured and treated them separately.

| Problem | What we did | Result |
|---|---|---|
| Names and rare words are spelled wrongly | Copy mechanism (run E) | Unseen words kept: 67.6% to 82.0% |
| Meaning is often wrong, because the real data is small and the synthetic data was not mixed | Code-mixed synthetic data (run F) | chrF 51.6 to 52.2, BLEU-4 35.4 to 35.9 |

What would help further, in the order we would try it:

1. **Longer pre-training.** It was stopped at 10 passes while the score was still rising. This needs no new code.
2. **More real data,** for example the LinCE Hinglish set (about 8,000 pairs). Each new set must be checked so that no test tweet is included.
3. **Back-translation:** train a model in the other direction and use it to turn real English tweets into Hinglish. This gives synthetic data in the style of tweets.
4. **A pre-trained multilingual model.** In practice this gives the largest gain, but it is outside the rule of this project, which is to train from scratch.

With about 10,700 real pairs and a small model trained from scratch, some meaning errors will remain whatever is done. We state this as a limit of the project.

### 9.7 It is only translation. What is the use of the project?

Three points (details in Section 2.1):

1. **Translation opens the door to other tools.** Sentiment analysis, content filters, search and chatbots exist for English. Once Hinglish is translated, all of them can be used. Without translation, each tool would have to be built again for Hinglish, and each would face the same shortage of data. We tested this with sentiment analysis (Section 6.9): on tweets written in Hindi only, an English sentiment tool reaches a macro F1 of 19.9 on the tweet itself and 44.7 on our translation. Over all tweets the gain is small (45.1 to 46.7), because many tweets already contain enough English.
2. **The parts are useful alone.** The script converter, the spelling-noise generator, the language tagger and the code-mixed data generator can be used in other Hinglish tasks.
3. **The findings carry over.** The main result is not one translator but an answer to a general question: how to train a model when real data is very small. The answer (pre-train on synthetic data made from a corpus in the native script) can be applied to other mixed languages of India.

### 9.8 What is the VADER score?

VADER (Hutto and Gilbert, 2014) is an English sentiment tool made of a word list and a few rules. It is not a neural network and it is not trained by us.

| Step | What VADER does |
|---|---|
| 1 | Looks up every word of the sentence in its list. Each listed word has a fixed value, from about -4 (very negative) to +4 (very positive), given by human raters. `good` is positive, `bad` is negative |
| 2 | Changes the values with rules: `not` turns a value around, `very` makes it stronger, capital letters and `!` make it stronger, and the part after `but` counts more |
| 3 | Adds the values and presses the sum into the range -1 to +1. This number is the **compound score** |

The formula of step 3 is `compound = sum / sqrt(sum² + 15)`.

We turn the compound score into a label with the limits given by the authors of VADER (`sentiment.py`):

| Compound score | Label |
|---|---|
| 0.05 or more | positive |
| -0.05 or less | negative |
| Between the two | neutral |

In our project VADER reads the English translation, not the Hinglish tweet. On the tweet itself it knows none of the Hindi words, so the sum is 0 and the answer is neutral.

### 9.9 How does the system know which word is Hindi and which is English?

The translation model does not know, and does not need to. It reads word pieces and never receives a language label. It learns from examples that some input pieces are kept and others are translated.

The language of each word is found by a separate program, the rule-based tagger in `langid.py`. Its output is used for the data statistics, the Code-Mixing Index, the grouping of tweets (Sections 6.5 and 6.9) and the colours in the demo.

The tagger gives each word one of three tags: English, Hindi or other. The checks are made in this order, and the first one that fits decides:

| Order | Check | Tag | Example |
|---|---|---|---|
| 1 | Placeholder, number, symbol, laughter, link residue | Other | `<user>`, `123`, `hahaha`, `rt` |
| 2 | Single letter, looked up in a small table | English or Hindi | `u` is English, `h` and `k` are Hindi |
| 3 | One of 56 words that are valid in both languages | Decided by the neighbours (see below) | `to`, `me`, `is`, `do` |
| 4 | One of about 190 Romanised Hindi words that the English word list also contains | Hindi | `hai`, `ke`, `se` |
| 5 | Known to the English word list of the `wordfreq` library, with a frequency of about one in a million words or more | English | `office`, `tired` |
| 6 | Everything else | Hindi | `bahut`, `kharab` |

**Words valid in both languages.** The tagger looks at the two words before and the two words after. If more of them are Hindi, the word is Hindi; if more are English, it is English. If the count is equal, a default for that word is used.

| Sentence | Word | Tag |
|---|---|---|
| `hostel me banti thi` | `me` | Hindi |
| `please tell me the truth` | `me` | English |

Stretched spellings are reduced before the checks, so `jaldiii` is treated as `jaldi`.

The basic rule (English if the dictionary knows the word, Hindi otherwise) is the one used in the PHINC paper. Checks 3 and 4 are our additions for the cases where that rule fails. The tagger has no accuracy figure, because we have no hand-labelled data to test it on (Section 8).

### 9.10 How are unknown words (OOV) handled?

OOV means "out of vocabulary": a word the system has not seen before. It is handled differently in three places.

| Place | What happens to an unknown word | Effect |
|---|---|---|
| Language tagger | A word that the English dictionary does not know is tagged Hindi | Names and misspelled English words can be tagged Hindi by mistake |
| Translation model, run A (whole words) | The word list holds the words seen at least twice in the real training tweets. Any other word is replaced by one special unit, `<unk>` | The word is lost. 7.8% of validation words are unknown |
| Translation model, runs B to F (word pieces) | The word is split into smaller pieces that the model knows. In the worst case the pieces are single letters | Almost nothing is unknown: 0.04% of validation units |
| VADER | A word that is not in its list gets the value 0 and is ignored | If no word is known, the score is 0 and the label is neutral |

**Word pieces do not solve the whole problem.** The model can now read any word, but it must still build the output word from pieces. For a word it never saw, it often picks wrong pieces: `menopause` becomes `menops` (Section 5.8). Run D keeps 88.9% of frequent words but only 67.6% of unseen ones.

**The copy mechanism is the second step** (run E). The decoder can take a piece straight from the input tweet instead of generating it. Unseen words are then kept 82.0% of the time, and 84.4% in run F (Section 6.8).

**Effect on the sentiment application.** VADER's treatment of unknown words explains one result in Section 6.9. On an untranslated Hindi tweet every word is unknown, so the answer is neutral. This is right by accident for many neutral tweets, which is why F1 for the neutral label falls from 42.3 to 37.8 after translation.

## 10. Remaining work

All planned work is complete: the code is written, all six experiments are trained and scored, the sentiment application is measured and the manual error analysis is done (Section 6.10).

Possible further work, not started:

| Task | Expected benefit |
|---|---|
| Longer pre-training, more real data (Section 9.6) | Fewer meaning errors |
| A manual analysis of run F, and a second annotator | The manual analysis covers run D only and has one set of labels |
| Translate only tweets that are mostly Hindi before sentiment analysis (Section 6.9) | Keeps the gain on Hindi tweets without the loss on mixed tweets |

## 11. References

1. Srivastava, V. and Singh, M. (2020). PHINC: A Parallel Hinglish Social Media Code-Mixed Corpus for Machine Translation. Workshop on Noisy User-generated Text (W-NUT).
2. Kunchukuttan, A., Mehta, P. and Bhattacharyya, P. (2018). The IIT Bombay English-Hindi Parallel Corpus. LREC.
3. Bahdanau, D., Cho, K. and Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. ICLR.
4. Cho, K. et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. EMNLP.
5. Sennrich, R., Haddow, B. and Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. ACL.
6. Popović, M. (2015). chrF: character n-gram F-score for automatic MT evaluation. Workshop on Statistical Machine Translation.
7. Papineni, K. et al. (2002). BLEU: a Method for Automatic Evaluation of Machine Translation. ACL.
8. Post, M. (2018). A Call for Clarity in Reporting BLEU Scores. Conference on Machine Translation (WMT).
9. Das, A. and Gambäck, B. (2014). Identifying Languages at the Word Level in Code-Mixed Indian Social Media Text. ICON.
10. Koehn, P. (2004). Statistical Significance Tests for Machine Translation Evaluation. EMNLP.
11. See, A., Liu, P. J. and Manning, C. D. (2017). Get To The Point: Summarization with Pointer-Generator Networks. ACL.
12. Brown, P. F., Della Pietra, S. A., Della Pietra, V. J. and Mercer, R. L. (1993). The Mathematics of Statistical Machine Translation: Parameter Estimation. Computational Linguistics.
13. Patwa, P. et al. (2020). SemEval-2020 Task 9: Overview of Sentiment Analysis of Code-Mixed Tweets. International Workshop on Semantic Evaluation (SemEval).
14. Hutto, C. J. and Gilbert, E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. ICWSM.

## Appendix A: How to run the project

Setup (once):

```
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
```

Steps, in order:

| Command | What it does |
|---|---|
| `python -m hinglish_mt.splits` | Cleans PHINC and writes the train, validation and test files |
| `python -m hinglish_mt.analysis` | Writes the data statistics |
| `python -m hinglish_mt.synth` | Makes the clean and noisy synthetic files |
| `python -m hinglish_mt.tokenise` | Builds the word list and the word-piece model |
| `python -m hinglish_mt.train configs/run_a.yaml` | Trains run A (same for `run_b.yaml`) |
| `python -m hinglish_mt.train configs/pretrain_clean.yaml` | Pre-trains on clean synthetic data (same for `pretrain_noisy.yaml`) |
| `python -m hinglish_mt.train configs/run_c.yaml` | Fine-tunes run C (same for `run_d.yaml`) |
| `python -m hinglish_mt.train configs/pretrain_noisy_copy.yaml` | Pre-trains the model with the copy mechanism |
| `python -m hinglish_mt.train configs/run_e.yaml` | Fine-tunes run E |
| `python -m hinglish_mt.codemix` | Builds the word dictionary and the code-mixed synthetic files |
| `python -m hinglish_mt.train configs/pretrain_mixed_copy.yaml` | Pre-trains on the code-mixed synthetic data |
| `python -m hinglish_mt.train configs/run_f.yaml` | Fine-tunes run F |
| `python -m hinglish_mt.evaluate copy run_a run_b run_c run_d run_e run_f` | Scores the runs on the test part |
| `python -m hinglish_mt.ablation` | Writes the comparison table and significance tests |
| `python -m hinglish_mt.buckets` | Scores each mixing group separately |
| `python -m hinglish_mt.rare_words run_b run_d run_e run_f` | Measures how well rare words and names are kept |
| `python -m hinglish_mt.embeddings` | Lists the closest word vectors for ten words |
| `python -m hinglish_mt.sentiment` | Translates the SentiMix test tweets and scores the sentiment application (about 20 minutes; `--score-only` repeats the scoring in under a minute) |
| `streamlit run demo/app.py` | Starts the web demo |
| `python -m hinglish_mt.error_sheets` | Writes the sheets for manual error analysis. Existing sheets are kept unless `--force` is given |
| `python -m hinglish_mt.error_summary` | Counts the answers in the filled-in sheets |
| `.\scripts\run_all.ps1` | Runs all steps above in order (about 5 hours) |
| `python -m pytest` | Runs the automatic tests |

## Appendix B: Where things are

| Path | Contents |
|---|---|
| `data/` | The two original datasets |
| `data/processed/` | Cleaned PHINC parts and the synthetic files |
| `src/hinglish_mt/clean.py`, `splits.py` | Cleaning, placeholders, data split |
| `src/hinglish_mt/langid.py`, `cmi.py`, `analysis.py` | Language tagger and data statistics |
| `src/hinglish_mt/romanise.py`, `noise.py`, `synth.py` | Synthetic data |
| `src/hinglish_mt/codemix.py` | Word dictionary and code-mixed synthetic data |
| `src/hinglish_mt/tokenise.py`, `data.py` | Words and word pieces, batches |
| `src/hinglish_mt/model.py` | The translation model |
| `src/hinglish_mt/train.py`, `decode.py`, `evaluate.py` | Training, translating, scoring |
| `src/hinglish_mt/ablation.py` | Comparison table and significance tests |
| `src/hinglish_mt/buckets.py`, `embeddings.py` | Results by mixing level, word-vector check |
| `src/hinglish_mt/rare_words.py` | Check of rare words and names |
| `src/hinglish_mt/sentiment.py` | Sentiment application |
| `data/SentiMix/` | The SentiMix files |
| `demo/app.py` | Web demo |
| `src/hinglish_mt/error_sheets.py`, `error_summary.py` | Sheets for manual error analysis, and the count of the answers |
| `scripts/run_all.ps1` | Repeats the whole pipeline |
| `README.md` | Short guide: installation, data, how to run |
| `configs/` | Settings for each run |
| `tests/` | Automatic tests |
| `outputs/` | Statistics, trained models, scores and all test translations |
| `context.md` | Detailed progress notes per phase |

## Appendix C: Terms

| Term | Meaning |
|---|---|
| Hinglish | Hindi and English mixed, with Hindi typed in English letters |
| Code-mixing | Using two languages in one sentence |
| Devanagari | The script Hindi is normally written in |
| Romanise | Write Hindi in English letters |
| Synthetic data | Training data produced by a program, not written by people |
| Pre-training | First round of training, on the large synthetic data |
| Fine-tuning | Second round of training, on the small real data |
| BPE (word pieces) | Splitting rare words into smaller, common pieces |
| GRU | A neural network layer that reads a sentence one unit at a time |
| Attention | Lets the model focus on the relevant input words for each output word |
| Epoch | One full pass over the training data |
| Beam search | Keeping several candidate translations while translating, then choosing the best |
| Baseline | A simple system used as a point of comparison |
| Seed | A number that fixes the random choices, so a run can be repeated |
| Copy mechanism | Lets the model take a word piece straight from the input instead of generating it |
| Word alignment | Finding which word of a sentence is the translation of which word in the other language |
| Sentiment analysis | Deciding whether a text is positive, negative or neutral |
| Macro F1 | The mean of the F1 scores of all labels; F1 combines how many of the given labels are right and how many of the true ones are found |
| OOV (out of vocabulary) | A word the system has not seen before |
| Compound score | The sentiment score of VADER for one sentence, from -1 to +1 |

## Appendix D: Viva questions and answers

Short answers to questions that may be asked. Questions on the design choices are in Section 9; they are not repeated here.

### D.1 Model

**1. Why a GRU and not a Transformer?**
Our real data is very small (10,719 pairs) and we train on one laptop GPU with 6 GB of memory. A Transformer usually needs more data and more tuning to train well from random weights. A GRU with attention is small (about 5.8 million parameters) and trains in minutes per pass. The aim of the project is to compare kinds of training data, and for that the model only has to be the same in every run. A Transformer might score higher; we state this as a limit (Section 8).

**2. Why a GRU and not an LSTM?**
Both solve the same problem: keeping information over a long sentence. A GRU has two gates, an LSTM has three and an extra memory cell. The GRU therefore has fewer parameters and trains faster, and on small data the two usually give similar quality.

**3. What does attention do, and what happens without it?**
Without attention the encoder must press the whole tweet into one fixed vector, and the decoder writes the full translation from that vector alone. Long sentences lose detail. With attention, at every output step the decoder gives a weight to each input piece and reads a weighted mix of them. So when it writes `home` it can look at `ghar`. We use the additive attention of Bahdanau et al. (2015).

**4. Why is the encoder bidirectional?**
The meaning of a word depends on the words on both sides. Hindi puts the verb at the end of the sentence, so a reader going left to right only meets it last. One GRU reads the tweet from the left, a second from the right, and the two results are joined. Each input piece then has a vector that knows the whole sentence.

**5. Why do Hinglish and English share one vocabulary?**
Both sides are written in English letters, and many words are the same on both sides: English words, names and placeholders. With one shared list of 8,000 word pieces the same piece has the same vector on both sides. This also makes the copy mechanism simple, because a copied piece needs no extra lookup.

**6. What is the difference between greedy search and beam search?**
Greedy search takes the most likely piece at every step and can never go back. Beam search keeps the 4 best partial translations at every step and picks the best complete one at the end. It costs more time and usually gives a better sentence.

### D.2 Training

**7. What are word pieces (BPE), and how are they made?**
BPE starts with single letters. It counts which two neighbouring units occur together most often in the training text and joins them into one new unit. This is repeated until there are 8,000 units. Frequent words become one unit; rare words are split into several. We learn the pieces from the real training tweets and the noisy synthetic data, never from the validation or test part.

**8. What are pre-training and fine-tuning, and why in this order?**
Pre-training is the first round, on the 300,000 synthetic pairs. Fine-tuning is the second round, on the 10,719 real pairs, starting from the pre-trained weights. The large data teaches Hindi words and English sentence building. The real data comes last so that the model ends on the style of real tweets.

**9. What do dropout, label smoothing and gradient clipping do?**
All three make training more stable or reduce over-fitting, which matters with little data.

| Technique | What it does |
|---|---|
| Dropout 0.3 | Switches off 30% of the units at random during training, so the model cannot rely on a few of them |
| Label smoothing 0.1 | Gives 90% of the target to the correct piece and spreads 10% over all pieces, so the model does not become too sure |
| Gradient clipping | Limits the size of one update step, so one unusual batch cannot spoil the weights |

**10. How do you decide when to stop training?**
After each pass we translate the validation part and compute chrF. The model with the best validation chrF is saved. Training stops when this score has not improved for 5 passes. The test part plays no role in this.

**11. How do you make sure the test data did not leak into training?**

- PHINC was split once with a fixed seed, and no sentence is in more than one part.
- The word list and the word pieces are built from training text only.
- The rates of the spelling noise were counted in the training tweets only.
- The test part was used once, for the final scores.
- For the sentiment application we checked that none of the 3,000 SentiMix test tweets occurs in the translator's training data.

**12. Why are mentions, hashtags and links replaced by placeholders?**
They must not be translated, and each one is nearly unique, so the model could learn nothing from them. Each is replaced by `<user>`, `<hashtag>` or `<url>`. The model learns to copy the placeholder, and the original is put back after translation.

### D.3 Evaluation

**13. Why is chrF the main score and not BLEU?**
BLEU counts whole words: a word is right or wrong. chrF counts short letter sequences, so a nearly correct word gets part of the credit. Tweets have many spellings, names and short sentences, and for such text word-level scores are harsh and unstable. We report BLEU as well, so the results can be compared with other work.

**14. How is BLEU computed?**
It counts how many word sequences of length 1 to 4 in the output also occur in the reference translation, and combines the four shares. Outputs that are shorter than the reference get a penalty, so that a system cannot score well by writing only a few safe words. BLEU-1 uses single words only.

**15. Why does the copy baseline score so well without translating?**
Tweets contain English words, names and placeholders, and these are the same in the reference translation. Returning the tweet unchanged therefore already gives chrF 41.8. This is why a low score is not zero, and why every model must be compared with this baseline. Run A does not beat it on chrF.

**16. What is paired bootstrap resampling?**
A test of whether a difference between two systems could be chance. We draw 1,341 test tweets at random, with repeats, and score both systems on this sample. This is done 1,000 times. The p-value is the share of samples in which the system that looks better does not win. Below 0.05 we call the difference significant. It is "paired" because both systems are scored on the same tweets each time.

**17. Why train with three random seeds when there is already a significance test?**
They answer different questions. The bootstrap asks: would the result change with other test tweets? The seeds ask: would the result change if the same model were trained again? The starting weights and the order of the data are random, and run B alone moves by about 1.4 chrF between seeds. A gain must be larger than that to be believed.

**18. What is the Code-Mixing Index (CMI)?**
A number that says how mixed one sentence is (Das and Gambäck, 2014). Words without a language are left out. Then:

`CMI = 100 × (1 − words of the larger language / words with a language)`

A sentence in one language scores 0. A sentence that is half Hindi and half English scores 50, the highest value possible with two languages. Example: 6 Hindi words and 2 English words give `100 × (1 − 6/8) = 25`.

**19. Why macro F1 and not accuracy in the sentiment test?**
Accuracy can look good for a system that always gives the most frequent label. Always answering "neutral" reaches 36.7% accuracy but finds no positive and no negative tweet. Macro F1 is the mean of the F1 scores of the three labels, so each label counts equally; the same system gets only 17.9.

### D.4 Findings

**20. Spelling noise was part of the proposed method, but it gave no gain. Is that a failure?**
No. It is a result. The noise did what it was built for: it raised the share of Hindi words with more than one spelling in the synthetic data from 33.0% to 64.6%, close to the 60.8% of real tweets. But the translations did not improve significantly (chrF -0.07, p = 0.597). The likely reason is that word pieces already break unusual spellings into known parts, and fine-tuning on real tweets teaches the real spellings anyway. We did not test this reason. Reporting a step that did not help is part of honest reporting.

**21. What is IBM Model 1, and why was it used?**
A classic statistical method (Brown et al., 1993) that learns which word of one language goes with which word of the other, using only sentence pairs. It starts by treating all word pairs in a sentence pair as equally likely and improves the estimate round by round. We needed a Hindi to English word dictionary to make the code-mixed synthetic data of run F. Building it ourselves from our own data keeps the rule of the project: no pre-trained resources for translation.

**22. Why not use a pre-trained model such as mBART or an online translator?**
The rule of the project is to train the translator from scratch, so that every part of the pipeline is our own work and every gain can be traced to a step we made. A pre-trained multilingual model would very likely score higher (Section 9.6), but the experiments would then measure that model and not our method.

**23. If only one translation in five is fully correct, what has the project achieved?**
The contribution is the method and the measurements, not a finished product (Section 2.1). We show with controlled runs which steps help when real data is very small: word pieces (+5.2 chrF), pre-training on synthetic data (+5.0), a copy mechanism for rare words (67.6% to 82.0% kept) and code-mixed synthetic data (+0.6). We also show what did not help, and we state the limits.

**24. What would you do with more time?**
In this order (Section 9.6): longer pre-training, because the score was still rising at the tenth pass; more real data, for example the LinCE set; back-translation of real English tweets into Hinglish; and for the application, translating only the tweets that are mostly Hindi.
