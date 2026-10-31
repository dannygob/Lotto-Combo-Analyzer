from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from src.api import create_app


def test_health_endpoint(tmp_path) -> None:
    app = create_app(tmp_path / "registry.db")
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "lotto-combo-analyzer"


def test_frontend_origin_can_access_api_and_submit_preflighted_requests(tmp_path) -> None:
    app = create_app(tmp_path / "registry.db")
    client = TestClient(app)
    origin = "http://localhost:8080"

    health = client.get("/health", headers={"Origin": origin})
    assert health.status_code == 200
    assert health.headers["access-control-allow-origin"] == origin

    preflight = client.options(
        "/lotteries",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == origin
    assert "POST" in preflight.headers["access-control-allow-methods"]


def test_lottery_can_be_edited_without_changing_its_id(tmp_path) -> None:
    app = create_app(tmp_path / "registry.db")
    client = TestClient(app)
    assert client.post(
        "/lotteries",
        json={"name": "Old name"},
    ).status_code == 201
    lottery_id = client.get("/lotteries").json()[0]["lottery_id"]

    updated = client.put(
        f"/lotteries/{lottery_id}",
        json={"name": "New name"},
    )
    assert updated.status_code == 200
    assert client.get("/lotteries").json() == [
        {"lottery_id": lottery_id, "name": "New name"}
    ]
    assert client.put("/lotteries/missing", json={"name": "New name"}).status_code == 404


def test_lottery_id_is_generated_from_name_and_made_unique(tmp_path) -> None:
    app = create_app(tmp_path / "registry.db")
    client = TestClient(app)

    first = client.post("/lotteries", json={"name": "Lotería Súper"})
    second = client.post("/lotteries", json={"name": "Lotería Súper"})

    assert first.status_code == 201
    assert first.json()["lottery_id"] == "loteria-super"
    assert second.status_code == 201
    assert second.json()["lottery_id"] == "loteria-super-2"
    long_name = client.post("/lotteries", json={"name": "A" * 200})
    assert long_name.status_code == 201
    assert len(long_name.json()["lottery_id"]) <= 64
    assert client.get("/lotteries").json() == [
        {"lottery_id": "a" * 64, "name": "A" * 200},
        {"lottery_id": "loteria-super", "name": "Lotería Súper"},
        {"lottery_id": "loteria-super-2", "name": "Lotería Súper"},
    ]


def test_csv_inspection_suggests_observed_format_and_ruleset_is_readable(tmp_path) -> None:
    app = create_app(tmp_path / "registry.db")
    client = TestClient(app)
    csv_content = (
        "Sorteo,Fecha,numero 1,numero 2,numero 3,Plus\n"
        "1,2024-01-01,2,4,8,A\n"
        "2,2024-01-02,1,3,6,B\n"
    )

    inspection = client.post(
        "/datasets/inspect",
        files={"file": ("history.csv", csv_content.encode(), "text/csv")},
    )
    assert inspection.status_code == 200
    assert inspection.json()["columns"] == [
        "Sorteo", "Fecha", "numero 1", "numero 2", "numero 3", "Plus"
    ]
    assert inspection.json()["number_columns"] == ["numero 1", "numero 2", "numero 3"]
    assert inspection.json()["date_columns"] == ["Fecha"]
    assert inspection.json()["event_id_columns"] == ["Sorteo"]
    assert inspection.json()["earliest_draw_date"] == "2024-01-01"
    assert inspection.json()["suggested_format"] == {
        "minimum": 1,
        "maximum": 8,
        "numbers_drawn": 3,
    }
    assert "official rules" in inspection.json()["warning"]

    assert client.post("/lotteries", json={"name": "Example"}).status_code == 201
    created = client.post(
        "/rulesets",
        json={
            "lottery_id": "example",
            "version": "v1",
            "effective_from": "2020-01-01",
            "pools": [
                {
                    "pool_id": "main",
                    "kind": "primary",
                    "minimum": 1,
                    "maximum": 8,
                    "numbers_drawn": 3,
                }
            ],
        },
    )
    assert created.status_code == 201
    details = client.get("/lotteries/example/rulesets/v1")
    assert details.status_code == 200
    assert details.json()["pools"][0]["numbers_drawn"] == 3


def test_registering_new_ruleset_closes_previous_open_ended_format(tmp_path) -> None:
    app = create_app(tmp_path / "registry.db")
    client = TestClient(app)
    assert client.post("/lotteries", json={"name": "Example"}).status_code == 201
    base_ruleset = {
        "lottery_id": "example",
        "effective_from": "2020-01-01",
        "pools": [
            {
                "pool_id": "main",
                "kind": "primary",
                "minimum": 1,
                "maximum": 40,
                "numbers_drawn": 5,
            }
        ],
    }
    first = client.post("/rulesets", json={**base_ruleset, "version": "v1"})
    second = client.post(
        "/rulesets",
        json={
            **base_ruleset,
            "version": "v2",
            "effective_from": "2025-01-01",
            "pools": [{**base_ruleset["pools"][0], "maximum": 50}],
        },
    )

    assert first.status_code == 201
    assert second.status_code == 201
    formats = client.get("/lotteries/example/rulesets").json()
    assert {item["version"]: item["effective_to"] for item in formats} == {
        "v1": "2025-01-01",
        "v2": None,
    }


def test_registry_migrates_existing_lotteries_without_losing_names_or_ids(tmp_path) -> None:
    database_path = tmp_path / "legacy.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE lotteries (
                lottery_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                jurisdiction TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO lotteries VALUES (?, ?, ?)",
            ("existing-lottery", "Existing Lottery", "Legacy Region"),
        )

    app = create_app(database_path)
    client = TestClient(app)

    assert client.get("/lotteries").json() == [
        {"lottery_id": "existing-lottery", "name": "Existing Lottery"}
    ]
    with sqlite3.connect(database_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(lotteries)")}
    assert "jurisdiction" not in columns


def test_cascade_delete_preserves_source_files_used_by_another_lottery(tmp_path) -> None:
    app = create_app(
        tmp_path / "registry.db",
        dataset_directory=tmp_path / "datasets",
        model_directory=tmp_path / "models",
    )
    client = TestClient(app)
    shared_sha = "c" * 64

    for lottery_id, draw_date, record_number in (
        ("lottery-a", "2024-01-01", 1),
        ("lottery-b", "2024-01-02", 2),
    ):
        assert client.post(
            "/lotteries",
            json={"name": lottery_id},
        ).status_code == 201
        assert client.post(
            "/rulesets",
            json={
                "lottery_id": lottery_id,
                "version": "v1",
                "effective_from": "2020-01-01",
                "pools": [
                    {
                        "pool_id": "main",
                        "kind": "primary",
                        "minimum": 1,
                        "maximum": 10,
                        "numbers_drawn": 2,
                    }
                ],
            },
        ).status_code == 201
        assert client.post(
            "/draws",
            json={
                "lottery_id": lottery_id,
                "ruleset_version": "v1",
                "draw_date": draw_date,
                "numbers": {"main": [1, 2]},
                "source_sha256": shared_sha,
                "source_record_number": record_number,
            },
        ).status_code == 201

    deleted = client.delete("/lotteries/lottery-a")
    assert deleted.status_code == 200
    assert deleted.json()["deleted_counts"].get("source_files", 0) == 0
    assert client.get("/lotteries/lottery-b/draws").status_code == 200


def test_registry_round_trip_and_validation(tmp_path) -> None:
    app = create_app(tmp_path / "registry.db")
    client = TestClient(app)

    lottery = {"name": "Mega Millions"}
    response = client.post("/lotteries", json=lottery)
    assert response.status_code == 201
    assert response.json()["lottery_id"] == "mega-millions"

    ruleset = {
        "lottery_id": "mega-millions",
        "version": "v1",
        "effective_from": "2024-01-01",
        "effective_to": "2025-01-01",
        "pools": [
            {
                "pool_id": "main",
                "kind": "primary",
                "minimum": 1,
                "maximum": 70,
                "numbers_drawn": 5,
                "allow_replacement": False,
            },
            {
                "pool_id": "bonus",
                "kind": "bonus",
                "minimum": 1,
                "maximum": 25,
                "numbers_drawn": 1,
                "allow_replacement": False,
            },
        ],
    }
    response = client.post("/rulesets", json=ruleset)
    assert response.status_code == 201

    response = client.get("/lotteries")
    assert response.status_code == 200
    ids = [entry["lottery_id"] for entry in response.json()]
    assert "mega-millions" in ids

    validation = client.post(
        "/validate",
        json={
            "records": [
                {"n1": 1, "n2": 2, "n3": 3, "n4": 4, "n5": 5},
                {"n1": 8, "n2": 9, "n3": 11, "n4": 14, "n5": 22},
            ],
            "number_columns": ["n1", "n2", "n3", "n4", "n5"],
            "minimum": 1,
            "maximum": 99,
            "expected_count": 5,
            "ruleset": {
                "lottery_id": "mega-millions",
                "version": "v1",
                "effective_from": "2024-01-01",
                "effective_to": "2025-01-01",
                "pools": [{
                    "pool_id": "main",
                    "kind": "primary",
                    "minimum": 1,
                    "maximum": 70,
                    "numbers_drawn": 5,
                    "allow_replacement": False,
                }],
            },
        },
    )
    assert validation.status_code == 200
    payload = validation.json()
    assert payload["is_valid"] is True
    assert payload["ruleset_change"] is None

    draw = {
        "lottery_id": "mega-millions",
        "ruleset_version": "v1",
        "draw_date": "2024-12-15",
        "numbers": {"main": [1, 2, 3, 4, 5], "bonus": [12]},
        "source_sha256": "a" * 64,
        "source_record_number": 1,
        "source_reference": "manual-api-seed",
    }
    response = client.post("/draws", json=draw)
    assert response.status_code == 201
    assert response.json()["draw_id"] > 0

    draws = client.get("/lotteries/mega-millions/draws", params={"limit": 10})
    assert draws.status_code == 200
    assert any(item["draw_date"] == "2024-12-15" for item in draws.json())

    confidence = client.post(
        "/confidence/evaluate",
        json={
            "probabilities": [[0.8, 0.1, 0.1], [0.8, 0.1, 0.1]],
            "outcomes": [0, 0],
            "bootstrap_samples": 100,
        },
    )
    assert confidence.status_code == 200
    assert confidence.json()["beats_uniform_at_95pct"] is True

    gates = {
        "validation_passed": True,
        "leakage_free": True,
        "calibration_passed": True,
        "beats_null": True,
        "drift_within_tolerance": True,
    }
    model = {
        "model_version": "v1",
        "lottery_id": "mega-millions",
        "dataset_sha256": "a" * 64,
        "artifact_uri": "models/mega-millions/v1",
        "metrics": gates,
    }
    registration = client.post("/models", json=model)
    assert registration.status_code == 201
    promotion = client.post("/models/v1/promote")
    assert promotion.status_code == 200
    model_list = client.get("/lotteries/mega-millions/models")
    assert model_list.status_code == 200
    assert model_list.json()[0]["status"] == "active"


def test_csv_import_trains_first_snapshot_then_retrains_only_on_changed_file(tmp_path) -> None:
    app = create_app(
        tmp_path / "registry.db",
        dataset_directory=tmp_path / "datasets",
        model_directory=tmp_path / "models",
    )
    client = TestClient(app)
    assert client.post(
        "/lotteries",
        json={"name": "Lottery A"},
    ).status_code == 201
    lottery_id = client.get("/lotteries").json()[0]["lottery_id"]
    ruleset = {
        "lottery_id": lottery_id,
        "version": "v1",
        "effective_from": "2020-01-01",
        "pools": [
            {
                "pool_id": "main",
                "kind": "primary",
                "minimum": 1,
                "maximum": 10,
                "numbers_drawn": 5,
                "allow_replacement": False,
            }
        ],
    }
    assert client.post("/rulesets", json=ruleset).status_code == 201

    draw = client.post(
        "/draws",
        json={
            "lottery_id": "lottery-a",
            "ruleset_version": "v1",
            "draw_date": "2024-01-01",
            "numbers": {"main": [1, 2, 3, 4, 5]},
            "source_sha256": "b" * 64,
            "source_record_number": 1,
            "source_reference": "cascade-delete-test",
        },
    )
    assert draw.status_code == 201, draw.text

    csv_text = "Sorteo,numero 1,numero 2,numero 3,numero 4,numero 5\n,,,,,\n0,,,,,\n"
    csv_text += "".join(
        f"{draw},1,2,3,4,5\n" for draw in range(1, 41)
    )
    form = {
        "ruleset_version": "v1",
        "chronological_order_confirmed": "true",
    }
    first = client.post(
        f"/datasets/{lottery_id}/import",
        data=form,
        files={"file": ("history.csv", csv_text.encode(), "text/csv")},
    )
    assert first.status_code == 200, first.text
    first_payload = first.json()
    assert first_payload["retrained"] is True
    assert first_payload["first_dataset"] is True
    assert first_payload["previous_active_model"] is None
    assert first_payload["dataset_version"] == 1
    assert first_payload["row_count"] == 40
    assert first_payload["blank_rows_skipped"] == 1
    assert first_payload["rows_without_draw_numbers_skipped"] == 1
    assert first_payload["metrics"]["leakage_free"] is True

    repeated = client.post(
        "/datasets/lottery-a/import",
        data=form,
        files={"file": ("history.csv", csv_text.encode(), "text/csv")},
    )
    assert repeated.status_code == 200
    assert repeated.json()["retrained"] is False
    assert len(client.get(f"/lotteries/{lottery_id}/datasets").json()) == 1

    updated_csv = csv_text + "41,1,2,3,4,6\n"
    updated = client.post(
        f"/datasets/{lottery_id}/import",
        data=form,
        files={"file": ("history.csv", updated_csv.encode(), "text/csv")},
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["retrained"] is True
    assert updated.json()["first_dataset"] is False
    assert updated.json()["dataset_version"] == 2
    assert updated.json()["dataset_sha256"] != first_payload["dataset_sha256"]
    assert len(client.get(f"/lotteries/{lottery_id}/datasets").json()) == 2
    runs = client.get(f"/lotteries/{lottery_id}/training-runs")
    assert runs.status_code == 200
    assert len(runs.json()) == 2

    snapshot_path = Path(updated.json()["snapshot_uri"])
    artifact_path = Path(updated.json()["artifact_uri"])
    assert snapshot_path.is_file()
    assert artifact_path.is_file()

    deleted = client.delete(f"/lotteries/{lottery_id}")
    assert deleted.status_code == 200, deleted.text
    deleted_payload = deleted.json()
    assert deleted_payload["status"] == "deleted"
    assert deleted_payload["deleted_counts"]["lotteries"] == 1
    assert deleted_payload["deleted_counts"]["draws"] == 1
    assert deleted_payload["deleted_counts"]["draw_numbers"] == 5
    assert deleted_payload["deleted_counts"]["rulesets"] == 1
    assert deleted_payload["deleted_counts"]["dataset_versions"] == 2
    assert deleted_payload["deleted_counts"]["model_versions"] == 2
    assert deleted_payload["deleted_counts"]["training_runs"] == 2
    assert deleted_payload["deleted_counts"]["source_files"] == 3
    assert deleted_payload["cleanup_errors"] == []
    assert not snapshot_path.exists()
    assert not artifact_path.exists()
    assert client.get("/lotteries").json() == []
    assert client.delete(f"/lotteries/{lottery_id}").status_code == 404


def test_csv_import_requires_chronology_and_rejects_bad_draws(tmp_path) -> None:
    app = create_app(
        tmp_path / "registry.db",
        dataset_directory=tmp_path / "datasets",
        model_directory=tmp_path / "models",
    )
    client = TestClient(app)
    client.post(
        "/lotteries",
        json={"name": "lottery-a"},
    )
    client.post(
        "/rulesets",
        json={
            "lottery_id": "lottery-a",
            "version": "v1",
            "effective_from": "2020-01-01",
            "pools": [
                {
                    "pool_id": "main",
                    "kind": "primary",
                    "minimum": 1,
                    "maximum": 10,
                    "numbers_drawn": 2,
                    "allow_replacement": False,
                }
            ],
        },
    )
    csv_text = "numero 1,numero 2\n1,1\n2,3\n4,5\n"
    response = client.post(
        "/datasets/lottery-a/import",
        data={"ruleset_version": "v1"},
        files={"file": ("history.csv", csv_text.encode(), "text/csv")},
    )
    assert response.status_code == 422
    assert "chronological" in response.json()["detail"]
