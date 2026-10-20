import pytest

from hinglish_mt.clean import fix_text, has_devanagari, mask, preprocess, restore


@pytest.mark.parametrize(
    "text",
    [
        "@LoKarDi_RT uske liye toh bahot kuch karna padega #ForTheSakeOfHumanity",
        "Jeet ka jashn https://t.co/ix2dB9b1Iv",
        "@MehrTarar bhabhi India me aapke saare dewar aapko yaad kar rahe hai..",
        "#IndvsSA India waalo , #patakhe footenge ? @a @b http://x.co/1 www.y.in/2",
        "no masks here at all",
    ],
)
def test_mask_restore_round_trip(text):
    masked, originals = mask(text)
    assert restore(masked, originals) == text


def test_mask_replaces_every_kind():
    masked, originals = mask("@user1 dekho #mast video http://t.co/abc")
    assert masked == "<user> dekho <hashtag> video <url>"
    assert originals == [("user", "@user1"), ("hashtag", "#mast"), ("url", "http://t.co/abc")]


def test_url_glued_to_word_and_spaced_scheme():
    masked, originals = mask("dena bhaiyapic.twitter.com/FvLEv3onxZ")
    assert masked == "dena bhaiya <url>"
    assert originals == [("url", "pic.twitter.com/FvLEv3onxZ")]

    masked, _ = mask("see http: //www.youtube.com/watch?v=fwf2 sample")
    assert masked == "see <url> sample"


def test_mask_glued_to_punctuation_is_spaced_out():
    masked, originals = mask(".@MehrTarar bhabhi")
    assert masked == ". <user> bhabhi"
    assert restore(masked, originals) == ". @MehrTarar bhabhi"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("dekho #Masthttps://twitter.com/a/status/1", "dekho <hashtag> <url>"),
        ("chodenge... @Somepic.twitter.com/beas9vyohq", "chodenge... <user> <url>"),
        ("aya #Tagpic.twitter.com/hk7aha63ez", "aya <hashtag> <url>"),
        ("my lord?https://twitter.com/x/status/1", "my lord? <url>"),
        ("dekho twitter.com/p9jdhw0jrs abhi", "dekho <url> abhi"),
        ("truth https:// aur kya", "truth <url> aur kya"),
        ("mr.sharma ne kaha", "mr.sharma ne kaha"),
    ],
)
def test_links_glued_to_handles_hashtags_and_words(text, expected):
    assert mask(text)[0] == expected


def test_hash_and_at_inside_url_are_not_masked_separately():
    masked, originals = mask("link https://a.com/p#frag@x end")
    assert masked == "link <url> end"
    assert len(originals) == 1


def test_email_is_not_a_mention():
    masked, originals = mask("mail me at name@example.com")
    assert "<user>" not in masked
    assert originals == []


def test_restore_reorders_by_kind_and_drops_extras():
    originals = [("user", "@a"), ("hashtag", "#h"), ("user", "@b")]
    assert restore("<hashtag> hello <user> and <user> <user> <url>", originals) == "#h hello @a and @b"


def test_fix_text_repairs_mojibake_and_whitespace():
    assert fix_text("shuruat â€” eating\n\n bread � ") == "shuruat — eating bread"


def test_preprocess_lowercases_text_but_keeps_originals():
    text, originals = preprocess("Chalo @Mohit_Sharma ko 10 ADS")
    assert text == "chalo <user> ko 10 ads"
    assert originals == [("user", "@Mohit_Sharma")]


def test_has_devanagari():
    assert has_devanagari("yeh नहीं hai")
    assert not has_devanagari("yeh nahi hai")


def test_anonymised_and_truncated_links():
    assert mask("interview in 20 years http/URL #tag")[0] == "interview in 20 years <url> <hashtag>"
    assert mask("unki party ko https")[0] == "unki party ko <url>"
    assert mask("httpserver chal raha hai")[0] == "httpserver chal raha hai"
