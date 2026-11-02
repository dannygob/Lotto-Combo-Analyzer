from __future__ import annotations

import json
import math
import pandas as pd
import sqlite3
from datetime import date, datetime
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field
from typing import Any

from src.candidate_engine import generate_fixed_size_candidates
from src.confidence_engine import probabilistic_diagnostics
from src.csv_learning import import_and_retrain_csv, inspect_csv_draw_format, preview_csv_schema
from src.domain import CsvSchemaProfile, Draw, Lottery, Ruleset, SourceFile
from src.draw_schedule import (
    LotterySchedule,
    next_scheduled_event,
    scheduled_session_datetime,
)
from src.model_suite import (
    UniformBaselineModel,
    FrequencyBaselineModel,
    TransformerEncoderModel,
    MixtureDensityNetwork,
    NormalizingFlowModel,
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


class LotteryNameRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=200)


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
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
        allow_methods=["*"],
        allow_headers=["*"],
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

    @app.get("/lotteries/{lottery_id}/rulesets/{ruleset_version}", tags=["registry"])
    def get_ruleset(lottery_id: str, ruleset_version: str) -> dict[str, Any]:
        ruleset = registry.get_ruleset(lottery_id, ruleset_version)
        if ruleset is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"format {ruleset_version!r} was not found for lottery {lottery_id!r}",
            )
        return ruleset.model_dump(mode="json")

    @app.get("/lotteries/{lottery_id}/draws", tags=["registry"])
    def list_draws(lottery_id: str, limit: int = Query(100, ge=1, le=5000)) -> list[dict[str, Any]]:
        rows = registry.list_draws(lottery_id, limit=limit)
        if not rows:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"no draws were found for lottery {lottery_id!r}")
        return rows

    @app.post("/lotteries", tags=["registry"], status_code=status.HTTP_201_CREATED)
    def create_lottery(payload: LotteryNameRequest) -> dict[str, str]:
        lottery = registry.create_lottery(payload.name)
        return {"lottery_id": lottery.lottery_id, "status": "created"}

    @app.put("/lotteries/{lottery_id}", tags=["registry"])
    def update_lottery(lottery_id: str, payload: LotteryNameRequest) -> dict[str, str]:
        if not registry.update_lottery(Lottery(lottery_id=lottery_id, name=payload.name)):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"lottery {lottery_id!r} was not found",
            )
        return {"lottery_id": lottery_id, "status": "updated"}

    @app.delete("/lotteries/{lottery_id}", tags=["registry"])
    def delete_lottery(lottery_id: str) -> dict[str, object]:
        result = registry.delete_lottery(
            lottery_id,
            dataset_directory=resolved_dataset_directory,
            model_directory=resolved_model_directory,
        )
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"lottery {lottery_id!r} was not found",
            )
        return result

    @app.post("/rulesets", tags=["registry"], status_code=status.HTTP_201_CREATED)
    def create_ruleset(payload: Ruleset) -> dict[str, str]:
        try:
            registry.register_ruleset(payload)
        except sqlite3.IntegrityError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
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

    @app.post("/datasets/inspect", tags=["datasets"])
    async def inspect_csv(
        file: UploadFile = File(...),
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
            return inspect_csv_draw_format(raw_csv, file.filename)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        finally:
            await file.close()

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


# ============================================================================
# ENDPOINTS DE MÉTRICAS POR POOL
# ============================================================================

@app.get("/lotteries/{lottery_id}/pool-metrics/{pool_id}", tags=["analysis"])
def pool_metrics_endpoint(lottery_id: str, pool_id: str) -> dict[str, Any]:
    """Return metrics for a specific pool."""
    active_model = get_active_model(lottery_id)
    if active_model is None:
        raise HTTPException(status_code=404, detail="No active model found")
    
    metrics = active_model["metrics"]
    pool_metrics = metrics.get("pool_metrics", {})
    
    if pool_id not in pool_metrics:
        raise HTTPException(status_code=404, detail=f"Pool {pool_id} not found")
    
    return pool_metrics[pool_id]


@app.get("/lotteries/{lottery_id}/all-pool-metrics", tags=["analysis"])
def all_pool_metrics_endpoint(lottery_id: str) -> dict[str, Any]:
    """Return metrics for all pools."""
    active_model = get_active_model(lottery_id)
    if active_model is None:
        raise HTTPException(status_code=404, detail="No active model found")
    
    return active_model["metrics"].get("pool_metrics", {})


@app.get("/lotteries/{lottery_id}/pool-frequency/{pool_id}", tags=["analysis"])
def pool_frequency_endpoint(lottery_id: str, pool_id: str) -> dict[str, Any]:
    """Return frequency data for a specific pool."""
    active_model = get_active_model(lottery_id)
    if active_model is None:
        raise HTTPException(status_code=404, detail="No active model found")
    
    pool_frequency = active_model["metrics"].get("pool_frequency", {})
    if pool_id not in pool_frequency:
        raise HTTPException(status_code=404, detail=f"Pool {pool_id} not found")
    
    return pool_frequency[pool_id]


@app.get("/lotteries/{lottery_id}/all-pool-frequency", tags=["analysis"])
def all_pool_frequency_endpoint(lottery_id: str) -> dict[str, Any]:
    """Return frequency data for all pools."""
    active_model = get_active_model(lottery_id)
    if active_model is None:
        raise HTTPException(status_code=404, detail="No active model found")
    
    return active_model["metrics"].get("pool_frequency", {})


# ============================================================================
# ENDPOINTS DE AJUSTE DE MODELO Y DISTRIBUCIÓN DE SUMA
# ============================================================================

@app.post("/lotteries/{lottery_id}/improve-model", tags=["analysis"])
def improve_model_endpoint(lottery_id: str) -> dict[str, Any]:
    """Apply improvements to the model based on metrics."""
    active_model = get_active_model(lottery_id)
    if active_model is None:
        raise HTTPException(status_code=404, detail="No active model found")
    
    metrics = active_model["metrics"]
    pool_metrics = metrics.get("pool_metrics", {})
    predictions = active_model.get("pool_predictions", {})
    ruleset = get_ruleset(lottery_id, active_model.get("ruleset_version"))
    
    if not ruleset:
        raise HTTPException(status_code=404, detail="Ruleset not found")
    
    improvements = apply_model_improvements(
        metrics, pool_metrics, predictions, ruleset.pools
    )
    
    active_model["improvements"] = improvements
    active_model["last_improvement"] = datetime.now(timezone.utc).isoformat()
    
    return {
        "improvements_applied": True,
        "improvements": improvements,
    }


@app.get("/lotteries/{lottery_id}/model-improvements", tags=["analysis"])
def get_model_improvements_endpoint(lottery_id: str) -> dict[str, Any]:
    """Get current model improvements."""
    active_model = get_active_model(lottery_id)
    if active_model is None:
        raise HTTPException(status_code=404, detail="No active model found")
    
    return active_model.get("improvements", {})

    // == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    // FREQUENCY & CLUSTER
    ANALYSIS
    // == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==

    / ** Get
    global frequency
    metrics
    for a lottery. * /

        @app.get("/lotteries/{lottery_id}/frequency", tags=["analysis"])
        async def get_frequency(lottery_id: str) -> APIResponse:
            """Get frequency metrics for a lottery."""
            registry = SQLiteRegistry(DEFAULT_REGISTRY_PATH)
            lottery = registry.get_lottery(lottery_id)
            if not lottery:
                return APIResponse(status="error", message="Lottery not found")

            # Get frequency data from the lottery draws
            frame = lottery.draws
            if frame is None or frame.empty:
                return APIResponse(status="error", message="No draw data available")

            # Calculate number frequencies
            all_numbers = []
            for col in frame.columns:
                col_data = frame[col].dropna().tolist()
                all_numbers.extend(col_data)

            freq_counter = Counter(all_numbers)
            total_count = sum(freq_counter.values())

            frequencies = {}
            for num in sorted(freq_counter.keys()):
                frequencies[str(num)] = {
                    "count": freq_counter[num],
                    "frequency": freq_counter[num] / total_count if total_count > 0 else 0.0,
                }

            return APIResponse(
                status="success",
                message="Frequency metrics retrieved",
                data={"frequencies": frequencies, "total_draws": len(frame)}
            )

    / ** Get
    clustering
    results
    for a lottery. * /

        @app.get("/lotteries/{lottery_id}/clusters", tags=["analysis"])
        async def get_clusters(lottery_id: str, n_clusters: int = 5) -> APIResponse:
            """Get clustering results for a lottery."""
            registry = SQLiteRegistry(DEFAULT_REGISTRY_PATH)
            lottery = registry.get_lottery(lottery_id)
            if not lottery:
                return APIResponse(status="error", message="Lottery not found")

            # Placeholder - actual clustering would use the candidate_engine
            return APIResponse(
                status="success",
                message="Cluster analysis placeholder",
                data={"n_clusters": n_clusters, "note": "Clustering to be implemented"}
            )

    / ** Get
    association
    rules
    for a lottery. * /

        @app.get("/lotteries/{lottery_id}/association-rules", tags=["analysis"])
        async def get_association_rules(lottery_id: str, top_n: int = 10) -> APIResponse:
            """Get association rules for a lottery."""
            registry = SQLiteRegistry(DEFAULT_REGISTRY_PATH)
            lottery = registry.get_lottery(lottery_id)
            if not lottery:
                return APIResponse(status="error", message="Lottery not found")

            # Placeholder - actual association rule mining would use mlxtend
            return APIResponse(
                status="success",
                message="Association rules placeholder",
                data={"top_n": top_n, "note": "Association rules to be implemented"}
            )

    / ** Get
    model
    performance
    metrics
    for a lottery. * /

        @app.get("/lotteries/{lottery_id}/model-metrics", tags=["analysis"])
        async def get_model_metrics(lottery_id: str) -> APIResponse:
            """Get model performance metrics for a lottery."""
            registry = SQLiteRegistry(DEFAULT_REGISTRY_PATH)
            lottery = registry.get_lottery(lottery_id)
            if not lottery:
                return APIResponse(status="error", message="Lottery not found")

            # Placeholder model metrics
            return APIResponse(
                status="success",
                message="Model metrics retrieved",
                data={
                    "models": [],
                    "metrics": {
                        "accuracy": 0.0,
                        "log_loss": 0.0
                    }
                }
            )


@app.get("/lotteries/{lottery_id}/sum-distribution/{pool_id}", tags=["analysis"])
def sum_distribution_endpoint(lottery_id: str, pool_id: str) -> dict[str, Any]:
    """Return sum distribution for a specific pool."""
    active_model = get_active_model(lottery_id)
    if active_model is None:
        raise HTTPException(status_code=404, detail="No active model found")
    
    sum_distributions = active_model.get("improvements", {}).get("sum_distributions", {})
    
    if pool_id not in sum_distributions:
        raise HTTPException(status_code=404, detail=f"Pool {pool_id} not found")
    
    return sum_distributions[pool_id]


@app.get("/lotteries/{lottery_id}/sum-validation/{pool_id}", tags=["analysis"])
def sum_validation_endpoint(lottery_id: str, pool_id: str) -> dict[str, Any]:
    """Return sum validation for a specific pool."""
    active_model = get_active_model(lottery_id)
    if active_model is None:
        raise HTTPException(status_code=404, detail="No active model found")
    
    sum_validations = active_model.get("improvements", {}).get("sum_validations", {})
    
    if pool_id not in sum_validations:
        raise HTTPException(status_code=404, detail=f"Pool {pool_id} not found")
    
    return sum_validations[pool_id]


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

    # ------------------------------------------------------------------
    # Model‑Suite metadata endpoints
    # ------------------------------------------------------------------
    @app.get("/model_suite/uniform-baseline", tags=["model_suite"])
    def uniform_baseline(domain_size: int = Query(99, ge=2)):
        """Return info about the UniformBaselineModel."""
        model = UniformBaselineModel(domain_size=domain_size)
        return {"model": "UniformBaselineModel", "domain_size": domain_size, "status": "ready"}

    @app.get("/model_suite/frequency-baseline", tags=["model_suite"])
    def frequency_baseline(lottery_id: str = Query(...)):
        """Return info about the FrequencyBaselineModel using the dataset for the given lottery."""
        ds_path = Path(app.state.dataset_directory) / f"{lottery_id}.csv"
        if not ds_path.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")
        df = pd.read_csv(ds_path)
        combo_cols = [c for c in df.columns if c.startswith("numero ")]
        if not combo_cols:
            raise HTTPException(status_code=400, detail="No number columns in dataset")
        counts = df[combo_cols].values.flatten().astype(int)
        model = FrequencyBaselineModel(counts=counts)
        return {"model": "FrequencyBaselineModel", "domain_size": 99, "status": "ready"}

    @app.get("/model_suite/transformer", tags=["model_suite"])
    def transformer_endpoint(domain_size: int = Query(99, ge=2)):
        """Return info about the TransformerEncoderModel."""
        model = TransformerEncoderModel(domain_size=domain_size)
        return {"model": "TransformerEncoderModel", "domain_size": domain_size, "status": "ready"}

    @app.get("/model_suite/mdn", tags=["model_suite"])
    def mdn_endpoint(domain_size: int = Query(99, ge=2)):
        """Return info about the MixtureDensityNetwork model."""
        model = MixtureDensityNetwork(domain_size=domain_size)
        return {"model": "MixtureDensityNetwork", "domain_size": domain_size, "status": "ready"}

    @app.get("/model_suite/normalizing-flow", tags=["model_suite"])
    def normalizing_flow_endpoint(domain_size: int = Query(99, ge=2)):
        """Return info about the NormalizingFlowModel."""
        model = NormalizingFlowModel(domain_size=domain_size)
        return {"model": "NormalizingFlowModel", "domain_size": domain_size, "status": "ready"}

    @app.get("/model_suite/ensemble", tags=["model_suite"])
    def ensemble_endpoint():
        """Return info about the adaptive ensemble mechanism."""
        return {"model": "adaptive_ensemble", "status": "ready"}

    # ------------------------------------------------------------------
    # Correlation‑matrix endpoint
    # ------------------------------------------------------------------
    @app.get("/lotteries/{lottery_id}/correlation-matrix", tags=["datasets"])
    def correlation_matrix(lottery_id: str):
        """Compute Pearson correlation matrix for the number columns of the given lottery dataset."""
        ds_path = Path(app.state.dataset_directory) / f"{lottery_id}.csv"
        if not ds_path.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")
        df = pd.read_csv(ds_path)
        number_cols = [c for c in df.columns if c.startswith("numero ")]
        if len(number_cols) < 2:
            raise HTTPException(status_code=400, detail="Not enough number columns to compute a matrix")
        mat = df[number_cols].corr()
        return {"matrix": mat.values.tolist(), "column_names": number_cols}

    return app


app = create_app()
