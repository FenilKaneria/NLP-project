# Corpus analysis: synth_noisy

## Size

|  | Source (Hinglish) | Target (English) |
|---|---|---|
| Sentences | 13,399 | 13,399 |
| Tokens | 251,309 | 237,389 |
| Word tokens | 222,522 | 205,858 |
| Vocabulary (word types) | 31,463 | 21,138 |
| Types seen once | 62.8% | 51.9% |
| Mean length (tokens) | 18.8 | 17.7 |
| Median length | 17 | 16 |

## Sentence length (tokens)

| Bin | Source | Target |
|---|---|---|
| 1-4 | 0 | 4 |
| 5-9 | 2249 | 2636 |
| 10-14 | 3214 | 3375 |
| 15-19 | 2626 | 2585 |
| 20-29 | 3239 | 3128 |
| 30-40 | 1760 | 1454 |
| 41+ | 311 | 217 |

## Language tags and code-mixing (source)

| Measure | Value |
|---|---|
| Hindi / English / other tokens | 85.7% / 2.9% / 11.5% |
| Sentences with both languages | 37.2% |
| Mean CMI, all sentences | 3.3 |
| Mean CMI, mixed sentences only | 8.9 |
| CMI on pooled corpus counts | 3.3 |
| Mean switch points per sentence | 0.94 |
| Dominant language: Hindi / English / tie | 99.9% / 0.0% / 0.0% |
| Words that are valid in both languages | 5.6% |
| Words not in English dictionary (tokens / types) | 65.1% / 96.4% |

Sentence CMI distribution: 0: 8418, 0-10: 3548, 10-20: 1231, 20-30: 144, 30-40: 50, 40-50: 8

## Spelling variation (Hindi-tagged words)

Two keys bracket the true amount. Strict: spellings differ only by stretched letters, long vowels or a final n/h (misses vowel drops such as nhi). Loose: same first letter and consonants (also merges unrelated words such as bhi and bhai).

| Key | Groups with >1 spelling | Spellings per group | Hindi types covered | Hindi tokens covered |
|---|---|---|---|---|
| Strict (lower bound) | 4,787 | 2.3 | 36.6% | 64.6% |
| Loose (upper bound) | 5,240 | 3.9 | 67.5% | 68.2% |

Largest groups under the strict key:

| Key | Spellings (count) |
|---|---|
| hai | hai (7646), hain (1215), haih (43), haiii (37), haiiii (29), haii (23), hainnn (7), hainnnn (4) |
| ki | ki (6365), kki (2), kih (1) |
| aur | aur (3845), aurrr (15), aurr (14), aurrrr (12), aaur (1) |
| ka | ka (3626), kah (45), kaan (8), kaa (1) |
| ko | ko (3547), kon (5) |
| mei | mein (2944), mei (111), meinnn (13), meinnnn (9), meinn (4) |
| ek | ek (2215), ekk (2) |
| liye | liye (2004), liyeh (2) |
| ye | ye (1486), yeh (217) |
| bhi | bhi (1255), bhinn (48), bhee (20), bhiiii (4), bhinnn (1), bhii (1), bhiii (1) |
| is | is (1269), ees (1) |
| nahi | nahi (1124), nahin (49), nahee (17), nahiii (4), nahiiii (3), nahii (2) |
| kiya | kiya (1151), kiyaa (23), kiyaaa (7), kiyaaaa (3) |
| par | par (1119), paar (26), parrr (5), parrrr (5) |
| jo | jo (1093), joh (1) |
| wo | wo (534), woh (527) |
| tha | tha (900), ttha (47), thaa (7), thaaa (4), thaaaa (4), thaah (2), tthaa (2), thaan (1) |
| hi | hi (959), heen (5), heeh (1) |
| kar | kar (916), kaar (10), karrrr (4), karrr (3), karr (2) |
| tatha | tatha (778), tathaa (14), tathaaa (6), tathaaaa (5) |

## Most frequent words and bigrams

| Source unigram | Source bigram | Target unigram | Target bigram |
|---|---|---|---|
| ke (7816) | ke liye (1465) | the (14450) | of the (2144) |
| hai (7646) | hai ki (469) | of (8982) | in the (1344) |
| ki (6365) | k liye (386) | and (6032) | to the (747) |
| aur (3845) | hai aur (369) | to (5385) | on the (491) |
| se (3636) | hota hai (355) | in (4811) | for the (432) |
| ka (3626) | karne ke (344) | a (4242) | and the (430) |
| ko (3547) | sakta hai (271) | is (3203) | is a (377) |
| me (3282) | hai jo (236) | for (2179) | it is (369) |
| mein (2944) | jata hai (236) | that (1761) | to be (369) |
| k (2651) | gaya hai (234) | it (1455) | by the (360) |
| ek (2215) | kiya gaya (232) | on (1436) | with the (281) |
| liye (2004) | karta hai (232) | this (1342) | of a (274) |
| ye (1486) | roop se (232) | be (1330) | is the (261) |
| is (1294) | ke saath (230) | was (1320) | in a (245) |
| bhi (1255) | hoti hai (200) | are (1294) | of india (240) |
| hain (1215) | jaata hai (183) | by (1279) | from the (234) |
| kiya (1151) | ke roop (183) | as (1248) | has been (220) |
| ho (1145) | ja sakta (176) | with (1160) | that the (206) |
| ne (1128) | nahi hai (175) | or (1053) | at the (205) |
| nahi (1124) | ka ek (167) | he (967) | as a (190) |
| par (1120) | raha hai (159) | not (960) | the president (174) |
| jo (1093) | roop me (153) | his (880) | which is (159) |
| karne (1055) | roop mein (144) | have (876) | this is (158) |
| hi (960) | karne ki (143) | from (870) | there is (155) |
| kar (916) | kaha ki (136) | which (859) | it was (149) |
