import random

from hinglish_mt.codemix import MixConfig, build_dictionary, hindi_words, mixed_romanise
from hinglish_mt.noise import NoiseConfig, noisy_romanise

PAIRS = [
    ("घर बड़ा है", "house big"),
    ("घर छोटा है", "house small"),
    ("पानी ठंडा है", "water cold"),
    ("पानी बड़ा है", "water big"),
    ("छोटा पानी", "small water"),
    ("ठंडा घर", "cold house"),
]
DICTIONARY = {"घर": ["home", "house"], "पानी": ["water"]}
TEXT = "मेरा घर बड़ा है और पानी ठंडा है।"


def mix(text, target_words, sentence_share=1.0, word_chance=1.0, seed=1):
    config = MixConfig(sentence_share=sentence_share, word_chance=word_chance)
    return mixed_romanise(
        text, target_words, DICTIONARY, random.Random(seed), random.Random(seed + 100), NoiseConfig(), config
    )


def test_dictionary_learns_word_pairs():
    pairs = [(hindi_words(hindi), english.split()) for hindi, english in PAIRS]
    dictionary = build_dictionary(pairs, filter_words=False)
    assert dictionary["घर"] == ["house"]
    assert dictionary["पानी"] == ["water"]
    assert dictionary["ठंडा"] == ["cold"]


def test_no_mixing_gives_the_noisy_spelling():
    expected = noisy_romanise(TEXT, random.Random(1), NoiseConfig())
    assert mix(TEXT, {"house", "water"}, word_chance=0.0) == (expected, 0)
    assert mix(TEXT, {"house", "water"}, sentence_share=0.0) == (expected, 0)


def test_word_is_replaced_only_if_its_english_word_is_in_the_target():
    text, replaced = mix(TEXT, {"house"})
    assert replaced == 1
    assert "house" in text.split() and "water" not in text.split()

    text, replaced = mix(TEXT, set())
    assert replaced == 0 and "house" not in text


def test_other_words_keep_their_noisy_spelling():
    noisy = noisy_romanise(TEXT, random.Random(1), NoiseConfig()).split()
    mixed = mix(TEXT, {"house", "water"})[0].split()
    assert len(noisy) == len(mixed)
    different = [i for i in range(len(noisy)) if noisy[i] != mixed[i]]
    assert [mixed[i] for i in different] == ["house", "water"]


def test_same_seed_gives_the_same_text():
    assert mix(TEXT, {"house", "water"}, word_chance=0.5, seed=7) == mix(TEXT, {"house", "water"}, word_chance=0.5, seed=7)
