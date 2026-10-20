# Corpus analysis: phinc

## Size

|  | Source (Hinglish) | Target (English) |
|---|---|---|
| Sentences | 13,399 | 13,399 |
| Tokens | 203,772 | 192,759 |
| Word tokens | 148,449 | 148,662 |
| Vocabulary (word types) | 19,514 | 13,982 |
| Types seen once | 56.4% | 54.8% |
| Mean length (tokens) | 15.2 | 14.4 |
| Median length | 14 | 13 |

## Sentence length (tokens)

| Bin | Source | Target |
|---|---|---|
| 1-4 | 236 | 443 |
| 5-9 | 3628 | 3798 |
| 10-14 | 3412 | 3541 |
| 15-19 | 2460 | 2461 |
| 20-29 | 2942 | 2568 |
| 30-40 | 688 | 559 |
| 41+ | 33 | 29 |

## Language tags and code-mixing (source)

| Measure | Value |
|---|---|
| Hindi / English / other tokens | 52.1% / 20.8% / 27.1% |
| Sentences with both languages | 69.1% |
| Mean CMI, all sentences | 15.6 |
| Mean CMI, mixed sentences only | 22.7 |
| CMI on pooled corpus counts | 28.5 |
| Mean switch points per sentence | 2.16 |
| Dominant language: Hindi / English / tie | 84.8% / 13.1% / 2.1% |
| Words that are valid in both languages | 7.2% |
| Words not in English dictionary (tokens / types) | 44.6% / 66.0% |

Sentence CMI distribution: 0: 4144, 0-10: 1473, 10-20: 3352, 20-30: 1871, 30-40: 1619, 40-50: 940

## Source words unseen in train (word-level vocabulary)

| Split | Tokens | Types | Sentences affected |
|---|---|---|---|
| val | 6.5% | 26.2% | 52.7% |
| test | 6.8% | 27.0% | 53.3% |

## Spelling variation (Hindi-tagged words)

Two keys bracket the true amount. Strict: spellings differ only by stretched letters, long vowels or a final n/h (misses vowel drops such as nhi). Loose: same first letter and consonants (also merges unrelated words such as bhi and bhai).

| Key | Groups with >1 spelling | Spellings per group | Hindi types covered | Hindi tokens covered |
|---|---|---|---|---|
| Strict (lower bound) | 1,385 | 2.3 | 24.5% | 60.8% |
| Loose (upper bound) | 1,952 | 4.4 | 65.3% | 67.4% |

Largest groups under the strict key:

| Key | Spellings (count) |
|---|---|
| hai | hai (5957), hain (497), haii (3), haiii (2), haiiiiiii (1), haiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii (1) |
| nahi | nahi (1876), nahin (71), naahi (10), nahiiiiinnn (1), nahiiiiiiiiiiii (1) |
| to | to (1344), toh (587) |
| ke | ke (1841), keh (23) |
| ki | ki (1823), kii (2), kki (1) |
| bhi | bhi (1744), bhii (1) |
| ka | ka (1382), kaan (11), kaa (9), kah (7) |
| bhai | bhai (1354), bhaii (3), bhaai (2), bhaiii (2), bhaaaaiiiiii (1), bhaaaiiiii (1), bhaiiii (1), bhaiiiii (1) |
| ye | ye (1095), yeh (205), yehh (1) |
| ko | ko (1271), kon (22) |
| kya | kya (1104), kyaa (7), kyaaa (2) |
| hi | hi (1014), hii (8), hee (6), hiii (4), hin (2), heeee (1), hiiiii (1) |
| kar | kar (996), kaar (2), karr (2), karrrrr (1) |
| mei | mein (748), mei (28) |
| na | na (593), naa (49), naah (5), naan (3), naaaa (1), nahh (1), naaa (1) |
| tha | tha (569), thaa (10), thha (2) |
| aj | aaj (547), aj (21), aaaj (3), ajj (1) |
| ab | ab (550), aab (8), abb (6) |
| kuch | kuch (500), kuchh (51), kucch (3), kuchhh (1), kooch (1), kucchh (1) |
| ap | aap (496), ap (51) |

## Most frequent words and bigrams

| Source unigram | Source bigram | Target unigram | Target bigram |
|---|---|---|---|
| hai (5957) | ke liye (276) | the (4138) | you are (415) |
| to (2123) | raha hai (271) | you (3918) | in the (389) |
| nahi (1876) | nahi hai (201) | is (3709) | it is (373) |
| ke (1841) | hota hai (143) | to (3145) | i am (361) |
| ki (1823) | hai kya (128) | i (2451) | this is (361) |
| bhi (1744) | rahe hai (126) | a (2196) | are you (308) |
| se (1520) | ho gaya (126) | of (2193) | is the (265) |
| ho (1476) | bhi nahi (120) | in (2187) | will be (253) |
| ka (1382) | rahi hai (113) | and (1904) | of the (237) |
| bhai (1354) | hai ye (105) | this (1751) | is a (237) |
| ko (1271) | hai aur (101) | it (1724) | if you (233) |
| kya (1104) | hai bhai (100) | will (1541) | i have (232) |
| ye (1095) | lagta hai (94) | are (1503) | you have (216) |
| aur (1092) | kar rahe (89) | have (1354) | is not (203) |
| me (1090) | rahe ho (87) | not (1269) | do you (197) |
| the (1075) | ke saath (85) | for (1206) | there is (190) |
| hi (1051) | hi nahi (81) | on (1129) | i will (178) |
| kar (996) | kar raha (78) | brother (1015) | on the (169) |
| mein (748) | kar diya (78) | do (931) | want to (167) |
| ek (666) | modi ji (77) | be (922) | going to (160) |
| na (593) | ki baat (77) | that (913) | do not (157) |
| is (588) | ke baad (73) | my (894) | he is (153) |
| toh (587) | kya hai (71) | me (871) | to be (144) |
| tha (569) | hai to (71) | your (857) | did you (144) |
| ab (550) | ko bhi (69) | what (778) | used to (144) |
