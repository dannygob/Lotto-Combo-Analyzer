from datetime import date

import pandas as pd

from src.domain import NumberPool, PoolKind, Ruleset
from src.preprocessing import detect_ruleset_change, load_and_validate_history
from src.validation import validate_lottery_frame


def test_load_and_validate_history_accepts_valid_draw_file(tmp_path) -> None:
    csv_path = tmp_path / "valid_history.csv"
    pd.DataFrame(
        {
            "draw_date": ["2024-01-01", "2024-01-02"],
            "numero 1": [1, 12],
            "numero 2": [10, 40],
            "numero 3": [23, 18],
            "numero 4": [45, 7],
            "numero 5": [30, 36],
        }
    ).to_csv(csv_path, index=False)

    loaded = load_and_validate_history(
        csv_path,
        date_column="draw_date",
        minimum=1,
        maximum=50,
        expected_count=5,
    )
    assert list(loaded.columns) == ["draw_date", "numero 1", "numero 2", "numero 3", "numero 4", "numero 5"]
    assert len(loaded) == 2


def test_detect_ruleset_change_flags_count_and_range_drift() -> None:
    ruleset = Ruleset(
        lottery_id="lottery-a",
        version="v2",
        effective_from=date(2024, 1, 1),
        pools=(
            NumberPool(
                pool_id="main",
                kind=PoolKind.PRIMARY,
                minimum=1,
                maximum=50,
                numbers_drawn=5,
            ),
        ),
    )
    frame = pd.DataFrame({"main": [1, 2, 3, 4, 5, 12]})
    drift = detect_ruleset_change(ruleset, frame, number_columns=["main"])
    assert drift is not None
    assert drift["changes"][0]["reason"] == "draw count differs"


def test_validate_lottery_frame_returns_clean_records_and_issue_summary() -> None:
    ruleset = Ruleset(
        lottery_id="lottery-a",
        version="v3",
        effective_from=date(2024, 1, 1),
        pools=(
            NumberPool(
                pool_id="main",
                kind=PoolKind.PRIMARY,
                minimum=1,
                maximum=50,
                numbers_drawn=5,
            ),
        ),
    )
    frame = pd.DataFrame({"n1": [1], "n2": [2], "n3": [3], "n4": [4], "n5": [5]})
    result = validate_lottery_frame(frame, ruleset, number_columns=["n1", "n2", "n3", "n4", "n5"], expected_count=5)
    assert result.is_valid is True
    assert len(result.cleaned_rows) == 1
    assert result.ruleset_change is None

    invalid = pd.DataFrame({"n1": [0], "n2": [2], "n3": [3], "n4": [4], "n5": [5]})
    invalid_result = validate_lottery_frame(invalid, ruleset, number_columns=["n1", "n2", "n3", "n4", "n5"], expected_count=5)
    assert invalid_result.is_valid is False
    assert invalid_result.issues[0].code == "validation-error"
