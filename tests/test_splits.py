import pytest

from hinglish_mt import PROCESSED_DIR
from hinglish_mt.splits import SPLIT_NAMES, clean_rows, norm_key, read_tsv, split_rows


def _rows(pairs):
    return [{"id": i, "src_raw": s, "tgt_raw": t} for i, (s, t) in enumerate(pairs)]


def test_clean_rows_drops_devanagari_empty_and_duplicates():
    rows = _rows(
        [
            ("Kya haal hai", "how are you"),
            ("kya haal hai !!", "how are you doing"),  # duplicate after normalisation
            ("yeh नहीं hai", "this is not"),
            ("...", "nothing"),
            ("theek hai", " "),
            ("@someone kya HAAL hai", "@someone how are you"),  # differs only by a mention: kept
        ]
    )
    kept, report = clean_rows(rows)
    assert [r["id"] for r in kept] == [0, 5]
    assert report["dropped_duplicate"] == 1
    assert report["dropped_devanagari"] == 1
    assert report["dropped_empty"] == 2


def test_split_is_seeded_complete_and_disjoint():
    rows = [{"id": i} for i in range(1000)]
    a = split_rows(rows, seed=42)
    b = split_rows(rows, seed=42)
    assert a == b
    assert [len(a[name]) for name in SPLIT_NAMES] == [800, 100, 100]
    ids = [r["id"] for name in SPLIT_NAMES for r in a[name]]
    assert sorted(ids) == list(range(1000))
    assert split_rows(rows, seed=1) != a


def test_no_source_overlap_between_written_splits():
    paths = [PROCESSED_DIR / f"phinc_{name}.tsv" for name in SPLIT_NAMES]
    if not all(p.exists() for p in paths):
        pytest.skip("run `python -m hinglish_mt.splits` first")
    keys = [{norm_key(r["src"]) for r in read_tsv(p)} for p in paths]
    train, val, test = keys
    assert not train & val
    assert not train & test
    assert not val & test
