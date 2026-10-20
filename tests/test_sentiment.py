from hinglish_mt.sentiment import (
    confusion,
    join_words,
    label_of,
    paired_bootstrap,
    read_sentimix,
    remove_seen,
    scores_of,
    vader_labels,
)
from hinglish_mt.splits import norm_key

SAMPLE = (
    "meta\t11\tpositive\n"
    "@\tO\n"
    "rahul\tEng\n"
    "ye\tHin\n"
    "movie\tEng\n"
    "acchi\tHin\n"
    "hai\tHin\n"
    "#\tO\n"
    "fun\tEng\n"
    "\n"
    "meta\t12\n"
    "dekho\tHin\n"
    "https\tEng\n"
    "//\tO\n"
    "t\tEng\n"
    ".\tO\n"
    "co\tEng\n"
    "/\tO\n"
    "Ab12\tEng\n"
)


def test_read_sentimix_gives_text_and_label(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text(SAMPLE, encoding="utf-8")
    rows = read_sentimix(path)
    assert rows == [
        {"id": "11", "text": "@rahul ye movie acchi hai #fun", "label": "positive"},
        {"id": "12", "text": "dekho https://t.co/link", "label": ""},
    ]


def test_join_words_keeps_other_words_apart():
    assert join_words(["a", "@", "b", "c"]) == "a @b c"


def test_label_thresholds():
    assert label_of(0.05) == "positive"
    assert label_of(0.04) == "neutral"
    assert label_of(-0.04) == "neutral"
    assert label_of(-0.05) == "negative"


def test_vader_labels_clear_sentences():
    texts = ["i love this movie", "this is the worst day", "he is going home"]
    assert vader_labels(texts) == ["positive", "negative", "neutral"]


def test_remove_seen_drops_only_known_tweets():
    rows = [
        {"id": "1", "text": "Yaar ye kya hai @rahul", "label": "neutral"},
        {"id": "2", "text": "mujhe nahi pata", "label": "neutral"},
    ]
    seen = {norm_key("yaar ye kya hai <user>")}
    kept, dropped = remove_seen(rows, seen)
    assert dropped == 1
    assert kept == [rows[1]]


def test_scores_and_confusion():
    true_labels = ["positive", "negative", "neutral", "positive"]
    predicted = ["positive", "negative", "positive", "positive"]
    scores = scores_of(true_labels, predicted)
    assert scores["accuracy"] == 75.0
    assert scores["f1_negative"] == 100.0
    assert scores["f1_neutral"] == 0.0
    table = confusion(true_labels, predicted)
    assert table["neutral"]["positive"] == 1
    assert table["positive"]["positive"] == 2


def test_bootstrap_sees_a_clearly_better_system():
    true_labels = ["positive", "negative", "neutral"] * 20
    always_positive = ["positive"] * 60
    test = paired_bootstrap(true_labels, true_labels, always_positive, "accuracy")
    assert test["difference"] > 60
    assert test["p_value"] == 0
