from __future__ import annotations

import sqlite3
import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, ConfigDict, Field

from src.candidate_engine import generate_fixed_size_candidates
from src.confidence_engine import probabilistic_diagnostics
from src.csv_learning import import_and_retrain_csv, preview_csv_schema
from src.domain import CsvSchemaProfile, Draw, Lottery, Ruleset, SourceFile
from src.draw_schedule import (
    LotterySchedule,
    next_scheduled_event,
    scheduled_session_datetime,
)
from src.registry import SQLiteRegistry
from src.validation import validate_lottery_frame

DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parents[1] / "data" / "lotto_registry.db"
DEFAULT_DATASET_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "datasets"
DEFAULT_MODEL_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "models"
MAX_CSV_BYTES = 25 * 1024 * 1024


class ValidationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[dict[str, Any]] = Field(default_factory=list)
    ruleset: dict[str, Any] | None = None
    number_columns: list[str] | None = None
    minimum: int = 1
    maximum: int = 99
    expected_count: int | None = None


class ConfidenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    probabilities: list[list[float]]
    outcomes: list[int]
    regimes: list[str] | None = None
    calibration_bins: int = Field(default=10, ge=1)
    bootstrap_samples: int = Field(default=1000, ge=1)
    seed: int = 42


class ModelRegistrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_version: str = Field(min_length=1, max_length=128)
    lottery_id: str = Field(min_length=1, max_length=64)
    dataset_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    artifact_uri: str = Field(min_length=1, max_length=2000)
    metrics: dict[str, Any]


class ApiResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    message: str
    data: dict[str, Any] | None = None


def create_app(
    registry_path: str | Path | None = None,
    *,
    dataset_directory: str | Path | None = None,
    model_directory: str | Path | None = None,
) -> FastAPI:
    """Create a FastAPI application bound to a SQLite registry for lottery metadata and draws."""
    resolved_path = Path(registry_path) if registry_path is not None else DEFAULT_REGISTRY_PATH
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_dataset_directory = (
        Path(dataset_directory) if dataset_directory is not None else DEFAULT_DATASET_DIRECTORY
    )
    resolved_model_directory = (
        Path(model_directory) if model_directory is not None else DEFAULT_MODEL_DIRECTORY
    )

    registry = SQLiteRegistry(resolved_path)
    registry.initialize()

    app = FastAPI(
        title="Lotto-Combo-Analyzer",
        version="1.1.0",
        description="Scientific lottery analysis backend for data validation, provenance, and draw ingestion.",
    )
    app.state.registry = registry
    app.state.registry_path = str(resolved_path)
    app.state.dataset_directory = str(resolved_dataset_directory)
    app.state.model_directory = str(resolved_model_directory)

    @app.get("/health", tags=["system"])
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "lotto-combo-analyzer",
            "database": str(resolved_path),
        }

    @app.get("/lotteries", tags=["registry"])
    def list_lotteries() -> list[dict[str, str]]:
        return registry.list_lotteries()

    @app.get("/lotteries/{lottery_id}/rulesets", tags=["registry"])
    def list_rulesets(lottery_id: str) -> list[dict[str, Any]]:
        rows = registry.list_rulesets(lottery_id)
        if not rows:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"lottery {lottery_id!r} was not found")
        return rows

    @app.get("/lotteries/{lottery_id}/draws", tags=["registry"])
    def list_draws(lottery_id: str, limit: int = Query(100, ge=1, le=5000)) -> list[dict[str, Any]]:
        rows = registry.list_draws(lottery_id, limit=limit)
        if not rows:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"no draws were found for lottery {lottery_id!r}")
        return rows

    @app.post("/lotteries", tags=["registry"], status_code=status.HTTP_201_CREATED)
    def create_lottery(payload: Lottery) -> dict[str, str]:
        registry.register_lottery(payload)
        return {"lottery_id": payload.lottery_id, "status": "created"}

    @app.post("/rulesets", tags=["registry"], status_code=status.HTTP_201_CREATED)
    def create_ruleset(payload: Ruleset) -> dict[str, str]:
        registry.register_ruleset(payload)
        return {"lottery_id": payload.lottery_id, "version": payload.version, "status": "created"}

    @app.post("/draws", tags=["registry"], status_code=status.HTTP_201_CREATED)
    def create_draw(payload: Draw) -> dict[str, Any]:
        source_file = SourceFile(
            sha256=payload.source_sha256,
            original_name=f"{payload.lottery_id}-{payload.draw_date.isoformat()}.csv",
            storage_uri=f"memory://{payload.lottery_id}/{payload.draw_date.isoformat()}",
            media_type="application/octet-stream",
            size_bytes=0,
            source_url=payload.source_reference,
        )
        try:
            registry.register_source_file(source_file)
        except sqlite3.IntegrityError:
            pass
        draw_id = registry.add_draw(payload)
        return {"draw_id": draw_id, "lottery_id": payload.lottery_id, "ruleset_version": payload.ruleset_version, "status": "created"}

    @app.post("/validate", tags=["validation"])
    def validate_payload(payload: ValidationRequest) -> dict[str, Any]:
        frame = pd.DataFrame(payload.records)
        ruleset_model = Ruleset.model_validate(payload.ruleset) if payload.ruleset is not None else None
        result = validate_lottery_frame(
            frame,
            ruleset=ruleset_model,
            number_columns=payload.number_columns,
            minimum=payload.minimum,
            maximum=payload.maximum,
            expected_count=payload.expected_count,
        )
        return result.model_dump(mode="json")

    @app.post("/confidence/evaluate", tags=["confidence"])
    def evaluate_confidence(payload: ConfidenceRequest) -> dict[str, object]:
        try:
            return probabilistic_diagnostics(
                payload.probabilities,
                payload.outcomes,
                regimes=payload.regimes,
                calibration_bins=payload.calibration_bins,
                bootstrap_samples=payload.bootstrap_samples,
                seed=payload.seed,
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    @app.post("/models", tags=["models"], status_code=status.HTTP_201_CREATED)
    def register_model(payload: ModelRegistrationRequest) -> dict[str, str]:
        try:
            registry.register_model_version(
                model_version=payload.model_version,
                lottery_id=payload.lottery_id,
                dataset_sha256=payload.dataset_sha256,
                artifact_uri=payload.artifact_uri,
                metrics=payload.metrics,
            )
        except sqlite3.IntegrityError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
        return {"model_version": payload.model_version, "status": "candidate"}

    @app.get("/lotteries/{lottery_id}/models", tags=["models"])
    def list_models(lottery_id: str) -> list[dict[str, object]]:
        return registry.list_model_versions(lottery_id)

    @app.get("/lotteries/{lottery_id}/datasets", tags=["datasets"])
    def list_datasets(lottery_id: str) -> list[dict[str, object]]:
        return registry.list_dataset_versions(lottery_id)

    @app.get("/lotteries/{lottery_id}/training-runs", tags=["datasets"])
    def list_training_runs(lottery_id: str) -> list[dict[str, object]]:
        return registry.list_training_runs(lottery_id)

    @app.post("/datasets/{lottery_id}/preview", tags=["datasets"])
    async def preview_csv(
        lottery_id: str,
        file: UploadFile = File(...),
        ruleset_version: str = Form(...),
    ) -> dict[str, object]:
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="uploaded CSV must have a filename",
            )
        try:
            raw_csv = await file.read(MAX_CSV_BYTES + 1)
            if len(raw_csv) > MAX_CSV_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"CSV exceeds the {MAX_CSV_BYTES}-byte upload limit",
                )
            ruleset = registry.get_ruleset(lottery_id, ruleset_version)
            if ruleset is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="registered ruleset was not found")
            preview = preview_csv_schema(
                raw_csv,
                filename=file.filename,
                lottery_id=lottery_id,
                ruleset=ruleset,
            )
            profile = registry.get_csv_schema_profile(lottery_id, ruleset_version)
            preview["saved_schema_profile"] = (
                profile.model_dump(mode="json") if profile is not None else None
            )
            return preview
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        finally:
            await file.close()

    @app.put(
        "/lotteries/{lottery_id}/csv-schema-profiles/{ruleset_version}",
        tags=["datasets"],
    )
    def save_csv_schema_profile(
        lottery_id: str,
        ruleset_version: str,
        payload: CsvSchemaProfile,
    ) -> dict[str, str]:
        if payload.lottery_id != lottery_id or payload.ruleset_version != ruleset_version:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="schema profile lottery_id and ruleset_version must match the URL",
            )
        ruleset = registry.get_ruleset(lottery_id, ruleset_version)
        if ruleset is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="registered ruleset was not found")
        expected_pools = {pool.pool_id: pool.numbers_drawn for pool in ruleset.pools}
        if set(payload.pool_columns) != set(expected_pools):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="schema profile pool_columns must exactly match the registered ruleset pools",
            )
        for pool_id, expected_count in expected_pools.items():
            if len(payload.pool_columns[pool_id]) != expected_count:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"pool {pool_id!r} requires exactly {expected_count} CSV number columns",
                )
        try:
            registry.save_csv_schema_profile(payload)
        except sqlite3.IntegrityError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        return {
            "lottery_id": lottery_id,
            "ruleset_version": ruleset_version,
            "status": "saved",
        }

    @app.get(
        "/lotteries/{lottery_id}/csv-schema-profiles/{ruleset_version}",
        tags=["datasets"],
    )
    def get_csv_schema_profile(lottery_id: str, ruleset_version: str) -> dict[str, object]:
        profile = registry.get_csv_schema_profile(lottery_id, ruleset_version)
        if profile is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="CSV schema profile was not configured")
        return profile.model_dump(mode="json")

    @app.put("/lotteries/{lottery_id}/schedule", tags=["schedule"])
    def save_schedule(lottery_id: str, payload: LotterySchedule) -> dict[str, str]:
        if payload.lottery_id != lottery_id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="schedule lottery_id must match the URL lottery_id",
            )
        try:
            registry.save_lottery_schedule(payload)
        except sqlite3.IntegrityError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        return {"lottery_id": lottery_id, "status": "saved"}

    @app.get("/lotteries/{lottery_id}/schedule", tags=["schedule"])
    def get_schedule(lottery_id: str) -> dict[str, object]:
        schedule = registry.get_lottery_schedule(lottery_id)
        if schedule is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="lottery schedule was not configured")
        return schedule.model_dump(mode="json")

    @app.get("/lotteries/{lottery_id}/next-draw", tags=["schedule"])
    def get_next_draw(lottery_id: str, after: str | None = Query(default=None)) -> dict[str, str]:
        schedule = registry.get_lottery_schedule(lottery_id)
        if schedule is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="lottery schedule was not configured")
        try:
            after_datetime = datetime.fromisoformat(after) if after is not None else None
            return next_scheduled_event(schedule, after_datetime)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

    @app.get("/lotteries/{lottery_id}/prediction", tags=["predictions"])
    def predict_scheduled_event(
        lottery_id: str,
        event_id: str | None = Query(default=None, min_length=1, max_length=256),
        event_date: date | None = Query(default=None),
        session_id: str | None = Query(default=None, min_length=1, max_length=64),
        after: str | None = Query(default=None),
        candidate_limit: int = Query(default=5, ge=1, le=20),
    ) -> dict[str, object]:
        schedule = registry.get_lottery_schedule(lottery_id)
        if schedule is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="lottery schedule was not configured",
            )
        if event_id is not None:
            if event_date is not None or session_id is not None or after is not None:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="event_id cannot be combined with event_date, session_id, or after",
                )
            parts = event_id.split(":")
            if len(parts) != 3 or parts[0] != lottery_id:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="event_id must use the scheduled event format lottery_id:YYYY-MM-DD:session_id",
                )
            try:
                target_date = date.fromisoformat(parts[1])
                target_session = parts[2]
                target_time = scheduled_session_datetime(
                    schedule,
                    target_date,
                    target_session,
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=str(exc),
                ) from exc
            if target_date < schedule.effective_from or (
                schedule.effective_to is not None and target_date >= schedule.effective_to
            ):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="target event date is outside the configured schedule effective range",
                )
            target_event = {
                "event_id": f"{lottery_id}:{target_date.isoformat()}:{target_session}",
                "lottery_id": lottery_id,
                "event_date": target_date.isoformat(),
                "session_id": target_session,
                "scheduled_at": target_time.isoformat(),
                "timezone": schedule.timezone,
            }
        elif event_date is not None or session_id is not None:
            if event_date is None or session_id is None or after is not None:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="specify event_date and session_id together, without after or event_id",
                )
            if event_date < schedule.effective_from or (
                schedule.effective_to is not None and event_date >= schedule.effective_to
            ):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="target event date is outside the configured schedule effective range",
                )
            try:
                target_time = scheduled_session_datetime(
                    schedule,
                    event_date,
                    session_id,
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=str(exc),
                ) from exc
            target_event = {
                "event_id": f"{lottery_id}:{event_date.isoformat()}:{session_id}",
                "lottery_id": lottery_id,
                "event_date": event_date.isoformat(),
                "session_id": session_id,
                "scheduled_at": target_time.isoformat(),
                "timezone": schedule.timezone,
            }
        else:
            try:
                after_datetime = datetime.fromisoformat(after) if after is not None else None
                target_event = next_scheduled_event(schedule, after_datetime)
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=str(exc),
                ) from exc

        applicable_rulesets = [
            item
            for item in registry.list_rulesets(lottery_id)
            if item["effective_from"] <= target_event["event_date"]
            and (
                item["effective_to"] is None
                or target_event["event_date"] < item["effective_to"]
            )
        ]
        if len(applicable_rulesets) != 1:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="no unique registered ruleset applies to the target event date",
            )
        target_ruleset_version = str(applicable_rulesets[0]["version"])

        active_models = [
            model
            for model in registry.list_model_versions(lottery_id)
            if model["status"] == "active"
        ]
        if not active_models:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="no active model is available for this lottery",
            )
        active_model = active_models[0]
        model_path = Path(str(active_model["artifact_uri"]))
        try:
            artifact = json.loads(model_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"active model artifact is unavailable or invalid: {exc}",
            ) from exc
        if not isinstance(artifact, dict):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="active model artifact must contain a JSON object",
            )
        if artifact.get("lottery_ruleset_version") != target_ruleset_version:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "active model ruleset does not match the target event; "
                    "train and promote a model for the applicable ruleset"
                ),
            )
        ruleset = registry.get_ruleset(lottery_id, target_ruleset_version)
        pool_models = artifact.get("pool_models")
        if ruleset is None or not isinstance(pool_models, dict):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="active model artifact does not contain valid pool models",
            )
        expected_pool_ids = {pool.pool_id for pool in ruleset.pools}
        if set(pool_models) != expected_pool_ids:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="active model pools do not match the applicable ruleset",
            )

        pool_predictions: dict[str, object] = {}
        for pool in ruleset.pools:
            pool_model = pool_models[pool.pool_id]
            if not isinstance(pool_model, dict):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"active model pool {pool.pool_id!r} is malformed",
                )
            probabilities = pool_model.get("probabilities")
            expected_size = pool.maximum - pool.minimum + 1
            if (
                pool_model.get("minimum") != pool.minimum
                or pool_model.get("maximum") != pool.maximum
                or pool_model.get("numbers_drawn") != pool.numbers_drawn
                or not isinstance(probabilities, list)
                or len(probabilities) != expected_size
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0.0
                    or value > 1.0
                    for value in probabilities
                )
                or not math.isclose(
                    sum(probabilities),
                    pool.numbers_drawn,
                    rel_tol=1e-8,
                    abs_tol=1e-8,
                )
            ):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"active model probabilities for pool {pool.pool_id!r} are invalid",
                )
            candidates = generate_fixed_size_candidates(
                probabilities,
                number_pool=range(pool.minimum, pool.maximum + 1),
                combination_size=pool.numbers_drawn,
                limit=candidate_limit,
            )
            pool_predictions[pool.pool_id] = {
                "number_inclusion_probabilities": {
                    str(number): float(probability)
                    for number, probability in zip(
                        range(pool.minimum, pool.maximum + 1),
                        probabilities,
                        strict=True,
                    )
                },
                "candidates": [
                    {
                        "numbers": list(candidate.numbers),
                        "rank": candidate.rank,
                        "score": candidate.score,
                        "model_implied_combination_probability": candidate.model_probability,
                        "score_definition": (
                            "sum of marginal inclusion probabilities; ranking only, "
                            "not a joint combination probability"
                        ),
                    }
                    for candidate in candidates
                ],
            }

        return {
            "status": "research_prediction",
            "event": target_event,
            "model_version": active_model["model_version"],
            "dataset_sha256": active_model["dataset_sha256"],
            "ruleset_version": target_ruleset_version,
            "model_metrics": active_model["metrics"],
            "pools": pool_predictions,
            "scientific_note": (
                "Per-number values are marginal inclusion estimates. Candidate combination "
                "probabilities are conditional-Bernoulli model outputs formed by treating "
                "the marginal estimates as independent inclusion propensities and "
                "conditioning on the pool's fixed draw count. They are model-implied, "
                "not empirically calibrated forecasts or evidence of predictability."
            ),
        }

    @app.post("/datasets/{lottery_id}/import", tags=["datasets"])
    async def import_csv(
        lottery_id: str,
        file: UploadFile = File(...),
        ruleset_version: str = Form(...),
        date_column: str | None = Form(default=None),
        event_id_column: str | None = Form(default=None),
        session_column: str | None = Form(default=None),
        draw_time_column: str | None = Form(default=None),
        chronological_order_confirmed: bool | None = Form(default=None),
        pool_columns: str | None = Form(default=None),
    ) -> dict[str, object]:
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="uploaded CSV must have a filename",
            )
        try:
            raw_csv = await file.read(MAX_CSV_BYTES + 1)
            if len(raw_csv) > MAX_CSV_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"CSV exceeds the {MAX_CSV_BYTES}-byte upload limit",
                )
            mapping = None
            if pool_columns is not None:
                parsed_mapping = json.loads(pool_columns)
                if not isinstance(parsed_mapping, dict) or any(
                    not isinstance(key, str)
                    or not isinstance(columns, list)
                    or any(not isinstance(column, str) for column in columns)
                    for key, columns in parsed_mapping.items()
                ):
                    raise ValueError("pool_columns must be a JSON object of pool IDs to column-name arrays")
                mapping = parsed_mapping
            return import_and_retrain_csv(
                registry,
                raw_csv=raw_csv,
                filename=file.filename,
                lottery_id=lottery_id,
                ruleset_version=ruleset_version,
                data_directory=resolved_dataset_directory,
                model_directory=resolved_model_directory,
                pool_columns=mapping,
                date_column=date_column,
                event_id_column=event_id_column,
                session_column=session_column,
                draw_time_column=draw_time_column,
                chronological_order_confirmed=chronological_order_confirmed,
            )
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="pool_columns must be valid JSON",
            ) from exc
        except sqlite3.IntegrityError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        finally:
            await file.close()

    @app.post("/models/{model_version}/promote", tags=["models"])
    def promote_model(model_version: str) -> dict[str, str]:
        try:
            registry.promote_model_version(model_version)
        except ValueError as exc:
            code = status.HTTP_404_NOT_FOUND if str(exc) == "model version does not exist" else status.HTTP_409_CONFLICT
            raise HTTPException(status_code=code, detail=str(exc)) from exc
        return {"model_version": model_version, "status": "active"}

    return app


app = create_app()
