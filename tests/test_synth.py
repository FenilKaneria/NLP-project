import random

import pytest

from hinglish_mt.noise import SHORTENINGS, NoiseConfig, noisy_romanise, noisy_word
from hinglish_mt.romanise import romanise, romanise_word
from hinglish_mt.synth import build_rows, detokenise, filter_pairs, reject_reason

NO_NOISE = NoiseConfig(nasal_drop=0, long_vowel_double=0, shortening_scale=0, vowel_length=0, vowel_drop=0, sound_swap=0, repeat=0)


@pytest.mark.parametrize(
    "word, expected",
    [
        # schwa deletion
        ("करना", "karna"), ("समझना", "samajhna"), ("सरकार", "sarkar"), ("मतलब", "matlab"),
        ("भारत", "bharat"), ("जनता", "janta"), ("समय", "samay"), ("मित्र", "mitra"),
        # long vowels
        ("काम", "kaam"), ("ठीक", "theek"), ("दूर", "door"), ("आप", "aap"), ("आदमी", "aadmi"),
        ("था", "tha"), ("की", "ki"), ("हमारा", "hamara"), ("पानी", "pani"),
        # nasalisation
        ("हैं", "hain"), ("में", "mein"), ("मैं", "main"), ("हूँ", "hoon"), ("संबंध", "sambandh"),
        ("अंदर", "andar"), ("हिंदी", "hindi"), ("उन्हें", "unhein"),
        # conjuncts and special consonants
        ("जल्दी", "jaldi"), ("क्या", "kya"), ("अच्छा", "accha"), ("पत्थर", "patthar"),
        ("ज्ञान", "gyaan"), ("शिक्षा", "shiksha"), ("लड़का", "ladka"), ("पढ़ाई", "padhai"),
        ("ज़रूर", "zarur"), ("फ़िल्म", "film"), ("कृपया", "kripya"), ("अतः", "atah"), ("दुःख", "duhkh"),
        # v and w
        ("वाला", "wala"), ("विकास", "vikas"), ("विश्वास", "vishwas"), ("स्वागत", "swagat"), ("सार्वजनिक", "sarvajnik"),
        # vowel sequences
        ("गए", "gaye"), ("लिए", "liye"), ("चाहिए", "chahiye"), ("गई", "gayi"), ("भाई", "bhai"),
        ("हुआ", "hua"), ("कोई", "koi"),
        # "ah" before a consonant
        ("पहले", "pehle"), ("कहना", "kehna"), ("रहते", "rehte"), ("बहुत", "bahut"),
        # exceptions
        ("नहीं", "nahi"), ("वह", "wo"), ("यह", "ye"), ("हम", "hum"),
    ],
)  # fmt: skip
def test_romanise_word(word, expected):
    assert romanise_word(word) == expected


def test_precomposed_and_combining_nukta_agree():
    assert romanise_word("ज़िंदगी") == romanise_word("ज़िंदगी") == "zindagi"


def test_final_nasal_can_be_dropped():
    assert romanise_word("हैं", final_nasal=False) == "hai"
    assert romanise_word("में", final_nasal=False) == "me"
    assert romanise_word("हूँ", final_nasal=False) == "hu"
    assert romanise_word("संबंध", final_nasal=False) == "sambandh"
    assert romanise_word("स्वयं", final_nasal=False) == "swayam"


def test_long_vowels_can_be_doubled_except_at_the_end():
    assert romanise_word("शादी", double_long=True) == "shaadi"
    assert romanise_word("ज़रूरी", double_long=True) == "zaroori"
    assert romanise_word("इंसान", double_long=True) == "insaan"
    assert romanise_word("करता", double_long=True) == "karta"


def test_romanise_sentence_keeps_latin_digits_and_punctuation():
    assert romanise("वह 3 जून को GNOME में १२ बार गया।") == "wo 3 joon ko GNOME mein 12 baar gaya."


def test_noise_is_deterministic_under_a_seed():
    text = "मैं आज बहुत खुश हूँ क्योंकि वह घर आ रहा है और हम साथ में खाना खाएंगे"
    first = noisy_romanise(text, random.Random(7))
    assert first == noisy_romanise(text, random.Random(7))
    assert any(noisy_romanise(text, random.Random(seed)) != first for seed in range(8, 20))


def test_zero_noise_equals_clean_romanisation():
    text = "मैं आज बहुत खुश हूँ क्योंकि वह घर आ रहा है"
    assert noisy_romanise(text, random.Random(1), NO_NOISE) == romanise(text)


def test_noise_leaves_latin_words_alone():
    config = NoiseConfig(nasal_drop=1, long_vowel_double=1, vowel_length=1, vowel_drop=1, sound_swap=1, repeat=1)
    out = noisy_romanise("यह computer बहुत अच्छा है", random.Random(3), config).split()
    assert out[1] == "computer"


def test_each_noise_operation():
    rng = random.Random(0)
    only = lambda **kw: NoiseConfig(**{**vars(NO_NOISE), **kw})  # noqa: E731
    assert noisy_word("karna", rng, only(vowel_drop=1)) == "krna"
    assert noisy_word("saath", rng, only(vowel_length=1)) == "sath"
    assert noisy_word("zarur", rng, only(sound_swap=1)) == "jarur"
    assert noisy_word("jaldi", rng, only(repeat=1)).startswith("jaldii")
    assert noisy_word("nahi", rng, only(shortening_scale=100)) in SHORTENINGS["nahi"][1]
    assert noisy_word("ki", rng, only(vowel_drop=1, repeat=1)) == "ki"


@pytest.mark.parametrize(
    "en, hi, reason",
    [
        ("He arrived in Benaras on June 3 .", "वह 3 जून को बनारस पहुंचा ।", None),
        ("Flag of Turkey", "तुर्की का ध्वज", "length"),
        ("one two three four five", "एक दो तीन चार पांच छह सात आठ नौ दस ग्यारह", "length_ratio"),
        ("Enable the _ word wrapping option now", "शब्द व्रेपिंग को अभी सक्षम करें _", "markup"),
        ("This is the keep alive option", "This is the keep alive option", "hindi_script"),
        ("यह वाक्य हिंदी में लिखा है", "यह वाक्य हिंदी में लिखा है", "english_script"),
        ("see the site http://a.com/x for more", "अधिक जानकारी के लिए साइट देखें", "mask_mismatch"),
    ],
)
def test_reject_reason(en, hi, reason):
    assert reject_reason(en, hi) == reason


def test_detokenise():
    assert detokenise("the writer 's fancy , is n't it ?") == "the writer's fancy, isn't it?"
    assert detokenise("rate of 3 .5 and a : b") == "rate of 3 .5 and a: b"


def test_filter_drops_duplicate_hindi_and_build_rows_align():
    pairs = [
        ("He arrived in Benaras on June 3 .", "वह 3 जून को बनारस पहुंचा ।"),
        ("He reached Benaras on the third of June .", "वह 3 जून को बनारस  पहुंचा ।"),
        ("The people of India were pleased .", "भारत की जनता को प्रसन्नता हुई थी ।"),
    ]
    kept, report = filter_pairs(pairs)
    assert [i for i, _, _ in kept] == [0, 2]
    assert report["dropped_duplicate"] == 1

    clean, noisy = build_rows(kept, seed=42, config=NoiseConfig())
    assert clean[0]["src"] == "wo 3 joon ko banaras pahuncha."
    assert clean[0]["tgt"] == "he arrived in benaras on june 3."
    assert [r["id"] for r in clean] == [r["id"] for r in noisy] == [0, 2]
    assert noisy == build_rows(kept, seed=42, config=NoiseConfig())[1]
