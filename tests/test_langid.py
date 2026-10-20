import pytest

from hinglish_mt.clean import tokenize
from hinglish_mt.cmi import cmi, corpus_cmi, switch_points
from hinglish_mt.langid import EN, HI, OTHER, is_ambiguous, squeezed_forms, tag


def test_tokenize_keeps_placeholders_and_contractions():
    assert tokenize("<user> don't worry, 10:30 pe milte hain!! <url>") == [
        "<user>", "don't", "worry", ",", "10:30", "pe", "milte", "hain", "!", "!", "<url>",
    ]  # fmt: skip


def test_basic_tags():
    tokens = tokenize("<user> cake kaha hai ?? 2 minute")
    assert tag(tokens) == [OTHER, EN, HI, HI, OTHER, OTHER, OTHER, EN]


def test_hindi_words_known_to_the_english_dictionary_stay_hindi():
    assert tag(["hai", "ke", "se", "ko", "aur", "nahi"]) == [HI] * 6


@pytest.mark.parametrize(
    "sentence, word, expected",
    [
        ("yehi to hum samjhane ki koshish kar rahe hain", "to", HI),
        ("i want to go home", "to", EN),
        ("hostel me banti thi", "me", HI),
        ("please tell me the truth", "me", EN),
        ("kaam is tarah nahi chalega", "is", HI),
        ("this is very good", "is", EN),
        ("mujhe do rupaye chahiye", "do", HI),
    ],
)
def test_ambiguous_words_follow_their_neighbours(sentence, word, expected):
    tokens = tokenize(sentence)
    assert is_ambiguous(word)
    assert tag(tokens)[tokens.index(word)] == expected


def test_ambiguous_word_alone_takes_its_default():
    assert tag(["to"]) == [EN]
    assert tag(["me"]) == [HI]


def test_stretched_words_and_laughter():
    assert squeezed_forms("jaldiii") == ("jaldiii", "jaldii", "jaldi")
    assert tag(["sooooo", "goooood", "nahiii", "hahahaha", "lolll"]) == [EN, EN, HI, OTHER, OTHER]


def test_cmi_values():
    assert cmi([HI, HI, HI, HI]) == 0
    assert cmi([OTHER, OTHER]) == 0
    assert cmi([HI, EN, OTHER]) == 50
    assert cmi([HI, HI, HI, EN, OTHER, OTHER]) == 25


def test_switch_points_ignore_other_tokens():
    assert switch_points([HI, OTHER, HI, EN, OTHER, EN, HI]) == 2


def test_corpus_cmi_summaries():
    summary = corpus_cmi([[HI, HI], [HI, EN], [EN, EN, EN, HI]])
    assert summary["mean_all"] == pytest.approx(25.0)
    assert summary["mean_mixed"] == pytest.approx(37.5)
    assert summary["mixed_share"] == pytest.approx(2 / 3)
    assert summary["pooled"] == pytest.approx(50.0)


def test_spelling_keys():
    from hinglish_mt.analysis import skeleton, strict_key

    assert {skeleton(w) for w in ("nahi", "nhi", "nahii")} == {"nh"}
    assert {strict_key(w) for w in ("nahi", "nahii", "nahin", "naahi")} == {"nahi"}
    assert strict_key("toh") == "to"
    assert strict_key("aaj") == "aj"
    assert strict_key("nhi") != strict_key("nahi")
