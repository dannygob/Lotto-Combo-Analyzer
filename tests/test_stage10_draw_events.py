from __future__ import annotations

import json
import math
from datetime import datetime

from fastapi.testclient import TestClient

from src.api import create_app
from src.domain import SourceFile
from src.draw_schedule import LotterySchedule, next_scheduled_event


def _register_lottery_and_ruleset(client: TestClient, lottery_id: str, pools: list[dict]) -> None:
    response = client.post(
        "/lotteries",
        json={
            "lottery_id": lottery_id,
            "name": lottery_id,
            "jurisdiction": "Test",
        },
    )
    assert response.status_code == 201
    response = client.post(
        "/rulesets",
        json={
            "lottery_id": lottery_id,
            "version": "v1",
            "effective_from": "2020-01-01",
            "pools": pools,
        },
    )
    assert response.status_code == 201


def test_schema_preview_maps_distinct_number_pools_and_api_imports_repeated_dates(tmp_path) -> None:
    app = create_app(
        tmp_path / "registry.db",
        dataset_directory=tmp_path / "datasets",
        model_directory=tmp_path / "models",
    )
    client = TestClient(app)
    _register_lottery_and_ruleset(
        client,
        "two-pool",
        [
            {
                "pool_id": "main",
                "kind": "primary",
                "minimum": 1,
                "maximum": 50,
                "numbers_drawn": 5,
                "allow_replacement": False,
            },
            {
                "pool_id": "bonus",
                "kind": "bonus",
                "minimum": 1,
                "maximum": 9,
                "numbers_drawn": 2,
                "allow_replacement": False,
            },
        ],
    )
    schedule = {
        "lottery_id": "two-pool",
        "timezone": "America/New_York",
        "effective_from": "2024-01-01",
        "sessions": [
            {"session_id": "early", "local_time": "09:00", "weekdays": [0, 1, 2, 3, 4, 5, 6]},
            {"session_id": "late", "local_time": "19:00", "weekdays": [0, 1, 2, 3, 4, 5, 6]},
        ],
    }
    assert client.put("/lotteries/two-pool/schedule", json=schedule).status_code == 200

    rows = [
        "draw_id,draw_date,session,draw_note,numero 1,numero 2,numero 3,"
        "numero 4,numero 5,numero 6,numero 7"
    ]
    for index in range(40):
        day = index // 2 + 1
        session = "early" if index % 2 == 0 else "late"
        main_start = 10 + index % 36
        main = [main_start + offset for offset in range(5)]
        bonus_start = index % 8 + 1
        bonus = [bonus_start, (bonus_start % 9) + 1]
        rows.append(
            f"official-{index + 1},2024-01-{day:02d},{session},verified,"
            + ",".join(str(number) for number in main + bonus)
        )
    csv_bytes = ("\n".join(rows) + "\n").encode()

    preview = client.post(
        "/datasets/two-pool/preview",
        data={"ruleset_version": "v1"},
        files={"file": ("history.csv", csv_bytes, "text/csv")},
    )
    assert preview.status_code == 200, preview.text
    preview_payload = preview.json()
    assert preview_payload["mapping_unambiguous"] is True
    assert preview_payload["suggested_pool_columns"] == {
        "main": [f"numero {index}" for index in range(1, 6)],
        "bonus": ["numero 6", "numero 7"],
    }
    ambiguous_preview = client.post(
        "/datasets/two-pool/preview",
        data={"ruleset_version": "v1"},
        files={
            "file": (
                "ambiguous.csv",
                b"numero 1,numero 2,numero 3,numero 4,numero 5,numero 6,numero 7\n1,2,3,4,5,6,7\n",
                "text/csv",
            )
        },
    )
    assert ambiguous_preview.status_code == 200
    assert ambiguous_preview.json()["confirmation_required"] is True
    assert ambiguous_preview.json()["suggested_pool_columns"] == {}
    profile = {
        "lottery_id": "two-pool",
        "ruleset_version": "v1",
        "pool_columns": {
            "main": [f"numero {index}" for index in range(1, 6)],
            "bonus": ["numero 6", "numero 7"],
        },
        "date_column": "draw_date",
        "event_id_column": "draw_id",
        "session_column": "session",
        "draw_time_column": None,
        "allowed_attribute_columns": ["draw_note"],
        "ignored_columns": [],
        "chronological_order_confirmed": False,
    }
    saved_profile = client.put(
        "/lotteries/two-pool/csv-schema-profiles/v1",
        json=profile,
    )
    assert saved_profile.status_code == 200, saved_profile.text
    assert client.get("/lotteries/two-pool/csv-schema-profiles/v1").json() == profile

    imported = client.post(
        "/datasets/two-pool/import",
        data={"ruleset_version": "v1"},
        files={"file": ("history.csv", csv_bytes, "text/csv")},
    )
    assert imported.status_code == 200, imported.text
    result = imported.json()
    assert result["row_count"] == 40
    assert result["metrics"]["leakage_free"] is True
    assert set(result["metrics"]["temporal_feature_tests"]) == {
        "weekday",
        "month",
        "quarter_of_year",
        "inter_draw_gap",
    }
    profile["allowed_attribute_columns"] = []
    profile["ignored_columns"] = ["draw_note"]
    profile_update = client.put(
        "/lotteries/two-pool/csv-schema-profiles/v1",
        json=profile,
    )
    assert profile_update.status_code == 200
    retrained_for_schema = client.post(
        "/datasets/two-pool/import",
        data={"ruleset_version": "v1"},
        files={"file": ("history.csv", csv_bytes, "text/csv")},
    )
    assert retrained_for_schema.status_code == 200, retrained_for_schema.text
    assert retrained_for_schema.json()["retrained"] is True
    assert retrained_for_schema.json()["dataset_version"] == 1
    assert retrained_for_schema.json()["model_version"] != result["model_version"]
    assert len(client.get("/lotteries/two-pool/training-runs").json()) == 2
    identical_schema_repeat = client.post(
        "/datasets/two-pool/import",
        data={"ruleset_version": "v1"},
        files={"file": ("history.csv", csv_bytes, "text/csv")},
    )
    assert identical_schema_repeat.status_code == 200
    assert identical_schema_repeat.json()["retrained"] is False

    changed_rows = csv_bytes.decode().splitlines()
    changed_rows = [changed_rows[0] + ",unreviewed"] + [
        f"{line},extra" for line in changed_rows[1:]
    ]
    unclassified = ("\n".join(changed_rows) + "\n").encode()
    rejected = client.post(
        "/datasets/two-pool/import",
        data={"ruleset_version": "v1"},
        files={"file": ("new-column.csv", unclassified, "text/csv")},
    )
    assert rejected.status_code == 422
    assert "unclassified CSV columns" in rejected.json()["detail"]


def test_draw_time_distinguishes_same_day_events_without_session_column(tmp_path) -> None:
    app = create_app(
        tmp_path / "registry.db",
        dataset_directory=tmp_path / "datasets",
        model_directory=tmp_path / "models",
    )
    client = TestClient(app)
    _register_lottery_and_ruleset(
        client,
        "timed-events",
        [
            {
                "pool_id": "main",
                "kind": "primary",
                "minimum": 1,
                "maximum": 10,
                "numbers_drawn": 2,
                "allow_replacement": False,
            }
        ],
    )
    csv_bytes = (
        "draw_date,draw_time,n1,n2\n"
        "2024-02-01,18:00,1,2\n"
        "2024-02-01,09:00,2,3\n"
        "2024-02-01,15:00,3,4\n"
        "2024-02-01,12:00,4,5\n"
    ).encode()
    imported = client.post(
        "/datasets/timed-events/import",
        data={"ruleset_version": "v1", "date_column": "draw_date", "draw_time_column": "draw_time"},
        files={"file": ("events.csv", csv_bytes, "text/csv")},
    )
    assert imported.status_code == 200, imported.text
    assert imported.json()["row_count"] == 4

    duplicate_time = csv_bytes.replace(b"12:00,4,5", b"09:00,4,5")
    rejected = client.post(
        "/datasets/timed-events/import",
        data={"ruleset_version": "v1", "date_column": "draw_date", "draw_time_column": "draw_time"},
        files={"file": ("duplicate.csv", duplicate_time, "text/csv")},
    )
    assert rejected.status_code == 422
    assert "combination must be unique" in rejected.json()["detail"]


def test_next_draw_uses_timezone_and_schedule_exceptions() -> None:
    schedule = LotterySchedule.model_validate(
        {
            "lottery_id": "dst-lottery",
            "timezone": "America/New_York",
            "effective_from": "2026-01-01",
            "sessions": [
                {"session_id": "evening", "local_time": "19:00", "weekdays": [0, 1, 2, 3, 4, 5, 6]},
                {"session_id": "morning", "local_time": "09:00", "weekdays": [0, 1, 2, 3, 4, 5, 6]},
            ],
            "exceptions": [
                {
                    "event_date": "2026-11-01",
                    "session_id": "morning",
                    "action": "cancel",
                }
            ],
        }
    )
    first_event = next_scheduled_event(
        schedule,
        datetime.fromisoformat("2026-10-31T00:00:00-04:00"),
    )
    assert first_event["event_id"] == "dst-lottery:2026-10-31:morning"
    after = datetime.fromisoformat("2026-10-31T20:00:00-04:00")
    event = next_scheduled_event(schedule, after)
    assert event["event_id"] == "dst-lottery:2026-11-01:evening"
    assert event["scheduled_at"] == "2026-11-01T19:00:00-05:00"


def test_next_draw_resolves_fall_back_fold_and_every_other_day_addition() -> None:
    folded_schedule = LotterySchedule.model_validate(
        {
            "lottery_id": "fold-lottery",
            "timezone": "America/New_York",
            "effective_from": "2026-01-01",
            "sessions": [
                {
                    "session_id": "night",
                    "local_time": "01:15",
                    "weekdays": [6],
                    "fold": 1,
                }
            ],
        }
    )
    after_first_fold = datetime.fromisoformat("2026-11-01T01:30:00-04:00")
    folded_event = next_scheduled_event(folded_schedule, after_first_fold)
    assert folded_event["scheduled_at"] == "2026-11-01T01:15:00-05:00"

    alternating_schedule = LotterySchedule.model_validate(
        {
            "lottery_id": "alternating",
            "timezone": "America/New_York",
            "effective_from": "2026-11-01",
            "sessions": [
                {
                    "session_id": "main",
                    "local_time": "10:00",
                    "every_n_days": 2,
                    "anchor_date": "2026-11-01",
                }
            ],
            "exceptions": [
                {
                    "event_date": "2026-11-02",
                    "session_id": "main",
                    "action": "add",
                    "local_time": "15:00",
                }
            ],
        }
    )
    added_event = next_scheduled_event(
        alternating_schedule,
        datetime.fromisoformat("2026-11-01T12:00:00-05:00"),
    )
    assert added_event["event_date"] == "2026-11-02"
    assert added_event["scheduled_at"] == "2026-11-02T15:00:00-05:00"
    recurring_event = next_scheduled_event(
        alternating_schedule,
        datetime.fromisoformat("2026-11-02T16:00:00-05:00"),
    )
    assert recurring_event["event_date"] == "2026-11-03"


def test_schedule_configuration_and_next_draw_api(tmp_path) -> None:
    app = create_app(tmp_path / "registry.db")
    client = TestClient(app)
    assert client.post(
        "/lotteries",
        json={"lottery_id": "scheduled", "name": "Scheduled", "jurisdiction": "Test"},
    ).status_code == 201
    schedule = {
        "lottery_id": "scheduled",
        "timezone": "America/New_York",
        "effective_from": "2026-01-01",
        "sessions": [
            {"session_id": "daily", "local_time": "12:00", "weekdays": [0, 1, 2, 3, 4, 5, 6]},
        ],
    }
    saved = client.put("/lotteries/scheduled/schedule", json=schedule)
    assert saved.status_code == 200
    assert client.get("/lotteries/scheduled/schedule").json()["timezone"] == "America/New_York"

    next_draw = client.get(
        "/lotteries/scheduled/next-draw",
        params={"after": "2026-07-04T11:00:00-04:00"},
    )
    assert next_draw.status_code == 200
    assert next_draw.json()["event_id"] == "scheduled:2026-07-04:daily"
    assert next_draw.json()["scheduled_at"] == "2026-07-04T12:00:00-04:00"
    invalid_after = client.get(
        "/lotteries/scheduled/next-draw",
        params={"after": "2026-07-04T11:00:00"},
    )
    assert invalid_after.status_code == 422


def test_prediction_api_binds_active_model_to_next_or_requested_event(tmp_path) -> None:
    app = create_app(
        tmp_path / "registry.db",
        model_directory=tmp_path / "models",
    )
    client = TestClient(app)
    _register_lottery_and_ruleset(
        client,
        "predictable-contract",
        [
            {
                "pool_id": "main",
                "kind": "primary",
                "minimum": 1,
                "maximum": 10,
                "numbers_drawn": 2,
                "allow_replacement": False,
            }
        ],
    )
    schedule = {
        "lottery_id": "predictable-contract",
        "timezone": "America/New_York",
        "effective_from": "2026-01-01",
        "sessions": [
            {"session_id": "day", "local_time": "12:00", "weekdays": [0, 1, 2, 3, 4, 5, 6]},
        ],
    }
    assert client.put("/lotteries/predictable-contract/schedule", json=schedule).status_code == 200
    no_model = client.get(
        "/lotteries/predictable-contract/prediction",
        params={"after": "2026-07-04T11:00:00-04:00"},
    )
    assert no_model.status_code == 409
    assert "no active model" in no_model.json()["detail"]

    registry = app.state.registry
    source_sha256 = "a" * 64
    registry.register_source_file(
        SourceFile(
            sha256=source_sha256,
            original_name="training.csv",
            storage_uri=str(tmp_path / "training.csv"),
            media_type="text/csv",
            size_bytes=1,
        )
    )
    artifact_path = tmp_path / "models" / "predictable-contract" / "model.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(
        json.dumps(
            {
                "model_type": "smoothed-frequency-inclusion",
                "model_version": "active-model",
                "lottery_ruleset_version": "v1",
                "pool_models": {
                    "main": {
                        "minimum": 1,
                        "maximum": 10,
                        "numbers_drawn": 2,
                        "probabilities": [0.8, 0.7, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    }
                },
                "evaluation": {"leakage_free": True},
            }
        ),
        encoding="utf-8",
    )
    model_payload = {
        "model_version": "active-model",
        "lottery_id": "predictable-contract",
        "dataset_sha256": source_sha256,
        "artifact_uri": str(artifact_path),
        "metrics": {
            "validation_passed": True,
            "leakage_free": True,
            "calibration_passed": True,
            "beats_null": True,
            "drift_within_tolerance": True,
        },
    }
    assert client.post("/models", json=model_payload).status_code == 201
    assert client.post("/models/active-model/promote").status_code == 200

    prediction = client.get(
        "/lotteries/predictable-contract/prediction",
        params={"after": "2026-07-04T11:00:00-04:00", "candidate_limit": 3},
    )
    assert prediction.status_code == 200, prediction.text
    result = prediction.json()
    assert result["event"]["event_id"] == "predictable-contract:2026-07-04:day"
    assert result["event"]["scheduled_at"] == "2026-07-04T12:00:00-04:00"
    assert result["model_version"] == "active-model"
    assert result["pools"]["main"]["number_inclusion_probabilities"]["1"] == 0.8
    assert len(result["pools"]["main"]["candidates"]) == 3
    assert math.isclose(
        sum(
            candidate["model_implied_combination_probability"]
            for candidate in result["pools"]["main"]["candidates"]
        ),
        1.0,
        abs_tol=1e-12,
    )
    assert "not empirically calibrated forecasts" in result["scientific_note"]

    by_session = client.get(
        "/lotteries/predictable-contract/prediction",
        params={"event_date": "2026-07-05", "session_id": "day"},
    )
    assert by_session.status_code == 200
    assert by_session.json()["event"]["event_id"] == "predictable-contract:2026-07-05:day"
    by_event_id = client.get(
        "/lotteries/predictable-contract/prediction",
        params={"event_id": "predictable-contract:2026-07-05:day"},
    )
    assert by_event_id.status_code == 200
    assert by_event_id.json()["event"] == by_session.json()["event"]
    ambiguous_target = client.get(
        "/lotteries/predictable-contract/prediction",
        params={"event_date": "2026-07-05"},
    )
    assert ambiguous_target.status_code == 422


def test_prediction_api_rejects_artifact_from_another_ruleset(tmp_path) -> None:
    app = create_app(tmp_path / "registry.db")
    client = TestClient(app)
    _register_lottery_and_ruleset(
        client,
        "wrong-ruleset",
        [
            {
                "pool_id": "main",
                "kind": "primary",
                "minimum": 1,
                "maximum": 10,
                "numbers_drawn": 2,
                "allow_replacement": False,
            }
        ],
    )
    schedule = {
        "lottery_id": "wrong-ruleset",
        "timezone": "UTC",
        "effective_from": "2026-01-01",
        "sessions": [
            {"session_id": "daily", "local_time": "12:00", "weekdays": [0, 1, 2, 3, 4, 5, 6]},
        ],
    }
    assert client.put("/lotteries/wrong-ruleset/schedule", json=schedule).status_code == 200
    source_sha256 = "b" * 64
    app.state.registry.register_source_file(
        SourceFile(
            sha256=source_sha256,
            original_name="training.csv",
            storage_uri=str(tmp_path / "training.csv"),
            media_type="text/csv",
            size_bytes=1,
        )
    )
    artifact_path = tmp_path / "wrong-ruleset-model.json"
    artifact_path.write_text(
        json.dumps(
            {
                "lottery_ruleset_version": "old-rules",
                "pool_models": {},
            }
        ),
        encoding="utf-8",
    )
    model = {
        "model_version": "old-model",
        "lottery_id": "wrong-ruleset",
        "dataset_sha256": source_sha256,
        "artifact_uri": str(artifact_path),
        "metrics": {
            "validation_passed": True,
            "leakage_free": True,
            "calibration_passed": True,
            "beats_null": True,
            "drift_within_tolerance": True,
        },
    }
    assert client.post("/models", json=model).status_code == 201
    assert client.post("/models/old-model/promote").status_code == 200
    response = client.get(
        "/lotteries/wrong-ruleset/prediction",
        params={"after": "2026-07-04T11:00:00+00:00"},
    )
    assert response.status_code == 409
    assert "ruleset does not match" in response.json()["detail"]


def test_schedule_rejects_nonexistent_local_time_during_dst_transition() -> None:
    schedule = LotterySchedule.model_validate(
        {
            "lottery_id": "dst-gap",
            "timezone": "America/New_York",
            "effective_from": "2026-01-01",
            "sessions": [
                {"session_id": "night", "local_time": "02:30", "weekdays": [6]},
            ],
        }
    )
    try:
        next_scheduled_event(schedule, datetime.fromisoformat("2026-03-07T23:00:00-05:00"))
    except ValueError as exc:
        assert "does not exist" in str(exc)
    else:
        raise AssertionError("a nonexistent local DST time must be rejected")
