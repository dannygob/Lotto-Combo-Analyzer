from __future__ import annotations

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


def test_registry_round_trip_and_validation(tmp_path) -> None:
    app = create_app(tmp_path / "registry.db")
    client = TestClient(app)

    lottery = {
        "lottery_id": "mega-millions",
        "name": "Mega Millions",
        "jurisdiction": "US",
    }
    response = client.post("/lotteries", json=lottery)
    assert response.status_code == 201

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
        json={"lottery_id": "lottery-a", "name": "Lottery A", "jurisdiction": "Test"},
    ).status_code == 201
    ruleset = {
        "lottery_id": "lottery-a",
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

    csv_text = "Sorteo,numero 1,numero 2,numero 3,numero 4,numero 5\n,,,,,\n0,,,,,\n"
    csv_text += "".join(
        f"{draw},1,2,3,4,5\n" for draw in range(1, 41)
    )
    form = {
        "ruleset_version": "v1",
        "chronological_order_confirmed": "true",
    }
    first = client.post(
        "/datasets/lottery-a/import",
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
    assert len(client.get("/lotteries/lottery-a/datasets").json()) == 1

    updated_csv = csv_text + "41,1,2,3,4,6\n"
    updated = client.post(
        "/datasets/lottery-a/import",
        data=form,
        files={"file": ("history.csv", updated_csv.encode(), "text/csv")},
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["retrained"] is True
    assert updated.json()["first_dataset"] is False
    assert updated.json()["dataset_version"] == 2
    assert updated.json()["dataset_sha256"] != first_payload["dataset_sha256"]
    assert len(client.get("/lotteries/lottery-a/datasets").json()) == 2
    runs = client.get("/lotteries/lottery-a/training-runs")
    assert runs.status_code == 200
    assert len(runs.json()) == 2


def test_csv_import_requires_chronology_and_rejects_bad_draws(tmp_path) -> None:
    app = create_app(
        tmp_path / "registry.db",
        dataset_directory=tmp_path / "datasets",
        model_directory=tmp_path / "models",
    )
    client = TestClient(app)
    client.post(
        "/lotteries",
        json={"lottery_id": "lottery-a", "name": "Lottery A", "jurisdiction": "Test"},
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
