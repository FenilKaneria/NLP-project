# Corpus analysis: synth_clean

## Size

|  | Source (Hinglish) | Target (English) |
|---|---|---|
| Sentences | 13,399 | 13,399 |
| Tokens | 251,309 | 237,389 |
| Word tokens | 222,722 | 205,858 |
| Vocabulary (word types) | 22,582 | 21,138 |
| Types seen once | 56.2% | 51.9% |
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
| Hindi / English / other tokens | 85.8% / 2.8% / 11.4% |
| Sentences with both languages | 36.0% |
| Mean CMI, all sentences | 3.2 |
| Mean CMI, mixed sentences only | 8.9 |
| CMI on pooled corpus counts | 3.2 |
| Mean switch points per sentence | 0.92 |
| Dominant language: Hindi / English / tie | 99.9% / 0.0% / 0.0% |
| Words that are valid in both languages | 4.4% |
| Words not in English dictionary (tokens / types) | 65.5% / 95.6% |

Sentence CMI distribution: 0: 8575, 0-10: 3405, 10-20: 1231, 20-30: 132, 30-40: 47, 40-50: 9

## Spelling variation (Hindi-tagged words)

Two keys bracket the true amount. Strict: spellings differ only by stretched letters, long vowels or a final n/h (misses vowel drops such as nhi). Loose: same first letter and consonants (also merges unrelated words such as bhi and bhai).

| Key | Groups with >1 spelling | Spellings per group | Hindi types covered | Hindi tokens covered |
|---|---|---|---|---|
| Strict (lower bound) | 718 | 2.1 | 6.9% | 33.0% |
| Loose (upper bound) | 3,066 | 3.2 | 45.8% | 58.9% |

Largest groups under the strict key:

| Key | Spellings (count) |
|---|---|
| hai | hai (7315), hain (2525), haih (35) |
| ki | ki (7003), kih (1) |
| aur | aur (4693), aaur (1) |
| ka | ka (3626), kah (45), kaan (9) |
| ko | ko (3545), kon (6) |
| liye | liye (2246), liyeh (2) |
| par | par (2084), paar (26) |
| bhi | bhi (1388), bhinn (53) |
| is | is (1274), ees (1) |
| jo | jo (1093), joh (1) |
| kar | kar (988), kaar (10) |
| hi | hi (958), heen (6), heeh (1) |
| tha | tha (933), thaah (3), thaan (1) |
| gaya | gaya (815), gayan (2) |
| to | to (781), toh (1) |
| apne | apne (684), aapne (23) |
| rup | roop (653), rup (42) |
| sath | saath (663), sath (1) |
| kuch | kuch (545), kooch (2) |
| thi | thi (466), theen (74), theeh (1) |

## Most frequent words and bigrams

| Source unigram | Source bigram | Target unigram | Target bigram |
|---|---|---|---|
| ke (9822) | ke liye (2067) | the (14450) | of the (2144) |
| hai (7315) | hai ki (544) | of (8982) | in the (1344) |
| ki (7003) | jata hai (503) | and (6032) | to the (747) |
| mein (6491) | ke saath (462) | to (5385) | on the (491) |
| aur (4693) | karne ke (452) | in (4811) | for the (432) |
| se (3765) | hai aur (414) | a (4242) | and the (430) |
| ka (3626) | hota hai (409) | is (3203) | is a (377) |
| ko (3545) | roop mein (341) | for (2179) | it is (369) |
| hain (2525) | sakta hai (326) | that (1761) | to be (369) |
| liye (2246) | gaya hai (286) | it (1455) | by the (360) |
| ek (2215) | ke roop (279) | on (1436) | with the (281) |
| par (2087) | karta hai (272) | this (1342) | of a (274) |
| ye (1703) | kiya gaya (264) | be (1330) | is the (261) |
| nahi (1451) | roop se (259) | was (1320) | in a (245) |
| bhi (1388) | nahi hai (232) | are (1294) | of india (240) |
| is (1294) | hoti hai (222) | by (1279) | from the (234) |
| kiya (1251) | hai jo (214) | as (1248) | has been (220) |
| wo (1165) | bharat ke (208) | with (1160) | that the (206) |
| ne (1126) | karte hain (199) | or (1053) | at the (205) |
| karne (1125) | sakte hain (199) | he (967) | as a (190) |
| ho (1108) | bare mein (195) | not (960) | the president (174) |
| jo (1093) | raha hai (193) | his (880) | which is (159) |
| kar (988) | kiya jata (186) | have (876) | this is (158) |
| hi (959) | ja sakta (186) | from (870) | there is (155) |
| tha (933) | hote hain (179) | which (859) | it was (149) |
