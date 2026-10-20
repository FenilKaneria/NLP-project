import pytest

from hinglish_mt.error_summary import count_column, sign_test

ROWS = [
    {"id": "1", "better": "D"},
    {"id": "2", "better": " D "},
    {"id": "3", "better": "same"},
]


def test_count_column_counts_answers():
    counts = count_column(ROWS, "better", ["D", "B", "same"])
    assert counts["D"] == 2
    assert counts["same"] == 1
    assert counts["B"] == 0


def test_count_column_rejects_unknown_answer():
    rows = ROWS + [{"id": "4", "better": "both"}]
    with pytest.raises(ValueError):
        count_column(rows, "better", ["D", "B", "same"])


def test_sign_test_values():
    assert sign_test(5, 5) == 1.0
    assert sign_test(10, 0) == pytest.approx(2 / 1024)
    assert sign_test(3, 7) == sign_test(7, 3)
