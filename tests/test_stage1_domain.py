from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from src.domain import (
    Draw,
    ExecutionLog,
    ExecutionStatus,
    Lottery,
    NumberPool,
    PoolKind,
    Ruleset,
    SourceFile,
)


def test_ruleset_uses_exclusive_end_date_and_unique_pools() -> None:
    ruleset = Ruleset(
        lottery_id="test-lottery",
        version="v1",
        effective_from=date(2020, 1, 1),
        effective_to=date(2021, 1, 1),
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
    assert ruleset.effective_to == date(2021, 1, 1)

    with pytest.raises(ValidationError):
        Ruleset(
            lottery_id="test-lottery",
            version="v2",
            effective_from=date(2020, 1, 1),
            pools=(
                NumberPool(
                    pool_id="main",
                    kind=PoolKind.PRIMARY,
                    minimum=1,
                    maximum=50,
                    numbers_drawn=5,
                ),
                NumberPool(
                    pool_id="main",
                    kind=PoolKind.BONUS,
                    minimum=1,
                    maximum=10,
                    numbers_drawn=1,
                ),
            ),
        )


def test_draw_rejects_empty_pools_and_normalizes_digest() -> None:
    with pytest.raises(ValidationError):
        Draw(
            lottery_id="test-lottery",
            ruleset_version="v1",
            draw_date=date(2024, 1, 1),
            numbers={"main": ()},
            source_sha256="A" * 64,
            source_record_number=1,
        )
    draw = Draw(
        lottery_id="test-lottery",
        ruleset_version="v1",
        draw_date=date(2024, 1, 1),
        numbers={"main": (1, 3, 5)},
        source_sha256="A" * 64,
        source_record_number=1,
    )
    assert draw.source_sha256 == "a" * 64


def test_provenance_models_validate_required_fields() -> None:
    source = SourceFile(
        sha256="a" * 64,
        original_name="draws.csv",
        storage_uri="data/raw/draws.csv",
        media_type="text/csv",
        size_bytes=100,
    )
    assert source.sha256 == "a" * 64

    finished = ExecutionLog(
        execution_id="run-1",
        pipeline_name="import",
        status=ExecutionStatus.SUCCEEDED,
        started_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        finished_at=datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc),
    )
    assert finished.status is ExecutionStatus.SUCCEEDED
    with pytest.raises(ValidationError):
        ExecutionLog(
            execution_id="run-2",
            pipeline_name="import",
            status=ExecutionStatus.FAILED,
        )


def test_number_pool_rejects_impossible_without_replacement() -> None:
    with pytest.raises(ValidationError):
        NumberPool(
            pool_id="main",
            kind=PoolKind.PRIMARY,
            minimum=1,
            maximum=5,
            numbers_drawn=6,
        )


def test_lottery_ids_are_isolated_by_contract() -> None:
    lottery = Lottery(
        lottery_id="regional-lottery",
        name="Regional Lottery",
    )
    assert lottery.lottery_id == "regional-lottery"
    with pytest.raises(ValidationError):
        Lottery(lottery_id="Regional Lottery", name="Invalid")
