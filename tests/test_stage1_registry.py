import random
import sqlite3
from datetime import date, datetime, timedelta, timezone

import pytest

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
from src.registry import SQLiteRegistry


def create_registry(tmp_path):
    registry = SQLiteRegistry(tmp_path / "registry.sqlite3")
    registry.initialize()
    registry.register_lottery(
        Lottery(lottery_id="lottery-a", name="Lottery A")
    )
    registry.register_lottery(
        Lottery(lottery_id="lottery-b", name="Lottery B")
    )
    registry.register_ruleset(
        Ruleset(
            lottery_id="lottery-a",
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
    )
    registry.register_ruleset(
        Ruleset(
            lottery_id="lottery-a",
            version="v2",
            effective_from=date(2021, 1, 1),
            pools=(
                NumberPool(
                    pool_id="main",
                    kind=PoolKind.PRIMARY,
                    minimum=1,
                    maximum=60,
                    numbers_drawn=5,
                ),
            ),
        )
    )
    registry.register_source_file(
        SourceFile(
            sha256="a" * 64,
            original_name="uniform.csv",
            storage_uri="data/raw/uniform.csv",
            media_type="text/csv",
            size_bytes=5000,
        )
    )
    return registry


def test_ruleset_ranges_are_exclusive_and_cannot_overlap(tmp_path) -> None:
    registry = create_registry(tmp_path)
    with pytest.raises(sqlite3.IntegrityError, match="ruleset date ranges overlap"):
        registry.register_ruleset(
            Ruleset(
                lottery_id="lottery-a",
                version="overlap",
                effective_from=date(2020, 12, 31),
                pools=(
                    NumberPool(
                        pool_id="main",
                        kind=PoolKind.PRIMARY,
                        minimum=1,
                        maximum=10,
                        numbers_drawn=3,
                    ),
                ),
            )
        )
    with pytest.raises(ValueError, match="outside the bound ruleset range"):
        registry.add_draw(
            Draw(
                lottery_id="lottery-a",
                ruleset_version="v1",
                draw_date=date(2021, 1, 1),
                numbers={"main": (1, 2, 3, 4, 5)},
                source_sha256="a" * 64,
                source_record_number=1,
            )
        )


def test_draw_pool_validation_and_lottery_isolation(tmp_path) -> None:
    registry = create_registry(tmp_path)
    with pytest.raises(ValueError, match="unregistered lottery ruleset"):
        registry.add_draw(
            Draw(
                lottery_id="lottery-b",
                ruleset_version="v1",
                draw_date=date(2020, 6, 1),
                numbers={"main": (1, 2, 3, 4, 5)},
                source_sha256="a" * 64,
                source_record_number=1,
            )
        )
    with pytest.raises(ValueError, match="exactly match"):
        registry.add_draw(
            Draw(
                lottery_id="lottery-a",
                ruleset_version="v1",
                draw_date=date(2020, 6, 1),
                numbers={"bonus": (1,)},
                source_sha256="a" * 64,
                source_record_number=1,
            )
        )
    with pytest.raises(ValueError, match="requires 5 numbers"):
        registry.add_draw(
            Draw(
                lottery_id="lottery-a",
                ruleset_version="v1",
                draw_date=date(2020, 6, 1),
                numbers={"main": (1, 2, 3)},
                source_sha256="a" * 64,
                source_record_number=1,
            )
        )


def test_uniform_null_draws_persist_with_valid_provenance(tmp_path) -> None:
    registry = create_registry(tmp_path)
    rng = random.Random(20261031)
    draw_ids = []
    for record_number in range(1, 101):
        draw_ids.append(
            registry.add_draw(
                Draw(
                    lottery_id="lottery-a",
                    ruleset_version="v1",
                    draw_date=date(2020, 1, 1) + timedelta(days=record_number),
                    numbers={"main": tuple(rng.sample(range(1, 51), 5))},
                    source_sha256="a" * 64,
                    source_record_number=record_number,
                )
            )
        )
    assert len(set(draw_ids)) == 100
    persisted = registry.get_draw(draw_ids[0])
    assert persisted is not None
    assert persisted["lottery_id"] == "lottery-a"
    assert persisted["ruleset_version"] == "v1"
    assert len(persisted["numbers"]["main"]) == 5
    assert all(1 <= number <= 50 for number in persisted["numbers"]["main"])


def test_execution_provenance_records_terminal_status_once(tmp_path) -> None:
    registry = create_registry(tmp_path)
    registry.record_execution(
        ExecutionLog(
            execution_id="import-run-1",
            lottery_id="lottery-a",
            pipeline_name="historical-import",
        )
    )
    registry.finish_execution(
        execution_id="import-run-1",
        status=ExecutionStatus.SUCCEEDED,
        finished_at=datetime.now(timezone.utc),
    )
    with pytest.raises(ValueError, match="already complete"):
        registry.finish_execution(
            execution_id="import-run-1",
            status=ExecutionStatus.SUCCEEDED,
            finished_at=datetime.now(timezone.utc),
        )


def test_model_promotion_requires_gates_and_archives_previous_active_model(tmp_path) -> None:
    registry = create_registry(tmp_path)
    passing_metrics = {
        "validation_passed": True,
        "leakage_free": True,
        "calibration_passed": True,
        "beats_null": True,
        "drift_within_tolerance": True,
    }
    registry.register_model_version(
        model_version="v1",
        lottery_id="lottery-a",
        dataset_sha256="a" * 64,
        artifact_uri="models/lottery-a/v1",
        metrics={**passing_metrics, "beats_null": False},
    )
    with pytest.raises(ValueError, match="beats_null"):
        registry.promote_model_version("v1")

    registry.register_model_version(
        model_version="v2",
        lottery_id="lottery-a",
        dataset_sha256="a" * 64,
        artifact_uri="models/lottery-a/v2",
        metrics=passing_metrics,
    )
    registry.promote_model_version("v2")
    registry.register_model_version(
        model_version="v3",
        lottery_id="lottery-a",
        dataset_sha256="a" * 64,
        artifact_uri="models/lottery-a/v3",
        metrics=passing_metrics,
    )
    registry.promote_model_version("v3")

    models = {model["model_version"]: model for model in registry.list_model_versions("lottery-a")}
    assert models["v2"]["status"] == "archived"
    assert models["v3"]["status"] == "active"
