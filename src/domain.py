"""Validated domain contracts for lottery rules, draws, and provenance."""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PoolKind(StrEnum):
    """Kinds of independently drawn number pools."""

    PRIMARY = "primary"
    BONUS = "bonus"
    SUPPLEMENTARY = "supplementary"


class ExecutionStatus(StrEnum):
    """Lifecycle state for a recorded pipeline execution."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


Slug = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")]
Version = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")]
Sha256 = Annotated[str, Field(pattern=r"^[a-fA-F0-9]{64}$")]


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class Lottery(DomainModel):
    lottery_id: Slug
    name: Annotated[str, Field(min_length=1, max_length=200)]


class NumberPool(DomainModel):
    pool_id: Slug
    kind: PoolKind
    minimum: Annotated[int, Field(ge=0)]
    maximum: Annotated[int, Field(gt=0)]
    numbers_drawn: Annotated[int, Field(gt=0)]
    allow_replacement: bool = False

    @model_validator(mode="after")
    def validate_bounds(self) -> NumberPool:
        if self.minimum > self.maximum:
            raise ValueError("minimum must not exceed maximum")
        available = self.maximum - self.minimum + 1
        if self.numbers_drawn > available and not self.allow_replacement:
            raise ValueError("numbers_drawn exceeds pool size without replacement")
        return self


class Ruleset(DomainModel):
    lottery_id: Slug
    version: Version
    effective_from: date
    effective_to: date | None = None
    pools: tuple[NumberPool, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_ruleset(self) -> Ruleset:
        if self.effective_to is not None and self.effective_to <= self.effective_from:
            raise ValueError("effective_to is exclusive and must be after effective_from")
        pool_ids = [pool.pool_id for pool in self.pools]
        if len(pool_ids) != len(set(pool_ids)):
            raise ValueError("pool_id values must be unique within a ruleset")
        return self


class CsvSchemaProfile(DomainModel):
    """Confirmed CSV column roles scoped to one lottery ruleset."""

    lottery_id: Slug
    ruleset_version: Version
    pool_columns: dict[Slug, list[Annotated[str, Field(min_length=1, max_length=128)]]]
    date_column: Annotated[str, Field(min_length=1, max_length=128)] | None = None
    event_id_column: Annotated[str, Field(min_length=1, max_length=128)] | None = None
    session_column: Annotated[str, Field(min_length=1, max_length=128)] | None = None
    draw_time_column: Annotated[str, Field(min_length=1, max_length=128)] | None = None
    allowed_attribute_columns: list[Annotated[str, Field(min_length=1, max_length=128)]] = Field(
        default_factory=list
    )
    ignored_columns: list[Annotated[str, Field(min_length=1, max_length=128)]] = Field(
        default_factory=list
    )
    chronological_order_confirmed: bool = False

    @model_validator(mode="after")
    def validate_column_roles(self) -> CsvSchemaProfile:
        flattened_pool_columns = [
            column
            for pool_columns in self.pool_columns.values()
            for column in pool_columns
        ]
        if any(not columns for columns in self.pool_columns.values()):
            raise ValueError("each configured number pool must map at least one CSV column")
        if len(flattened_pool_columns) != len(set(flattened_pool_columns)):
            raise ValueError("a CSV number column cannot be assigned to multiple pools")
        if any(
            len(columns) != len(set(columns))
            for columns in (*self.pool_columns.values(), self.allowed_attribute_columns, self.ignored_columns)
        ):
            raise ValueError("CSV column lists must not contain duplicates")

        role_columns = flattened_pool_columns + [
            column
            for column in (
                self.date_column,
                self.event_id_column,
                self.session_column,
                self.draw_time_column,
            )
            if column is not None
        ]
        classified_columns = (
            role_columns + self.allowed_attribute_columns + self.ignored_columns
        )
        if len(classified_columns) != len(set(classified_columns)):
            raise ValueError("each CSV column must have exactly one configured role")
        return self


class Draw(DomainModel):
    lottery_id: Slug
    ruleset_version: Version
    draw_date: date
    numbers: dict[Slug, tuple[Annotated[int, Field(ge=0)], ...]]
    source_sha256: Sha256
    source_record_number: Annotated[int, Field(gt=0)]
    source_reference: Annotated[str, Field(min_length=1, max_length=500)] | None = None

    @field_validator("source_sha256")
    @classmethod
    def normalize_sha256(cls, value: str) -> str:
        return value.lower()

    @model_validator(mode="after")
    def validate_number_groups(self) -> Draw:
        if not self.numbers:
            raise ValueError("a draw must contain at least one number pool")
        for pool_id, numbers in self.numbers.items():
            if not numbers:
                raise ValueError(f"number pool {pool_id!r} must not be empty")
        return self


class SourceFile(DomainModel):
    sha256: Sha256
    original_name: Annotated[str, Field(min_length=1, max_length=500)]
    storage_uri: Annotated[str, Field(min_length=1, max_length=2000)]
    media_type: Annotated[str, Field(min_length=1, max_length=100)]
    size_bytes: Annotated[int, Field(ge=0)]
    source_url: Annotated[str, Field(max_length=2000)] | None = None
    registered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("sha256")
    @classmethod
    def normalize_sha256(cls, value: str) -> str:
        return value.lower()

    @field_validator("registered_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("registered_at must be timezone-aware")
        return value.astimezone(timezone.utc)


class ExecutionLog(DomainModel):
    execution_id: Slug
    lottery_id: Slug | None = None
    pipeline_name: Annotated[str, Field(min_length=1, max_length=200)]
    status: ExecutionStatus = ExecutionStatus.RUNNING
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    code_version: Annotated[str, Field(max_length=128)] | None = None
    configuration_json: str = "{}"
    error_message: Annotated[str, Field(max_length=4000)] | None = None

    @field_validator("started_at", "finished_at")
    @classmethod
    def normalize_timestamps(cls, value: datetime | None) -> datetime | None:
        if value is not None:
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("execution timestamps must be timezone-aware")
            return value.astimezone(timezone.utc)
        return value

    @field_validator("configuration_json")
    @classmethod
    def validate_configuration_json(cls, value: str) -> str:
        import json

        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("configuration_json must encode a JSON object")
        return value

    @model_validator(mode="after")
    def validate_execution_state(self) -> ExecutionLog:
        if self.status == ExecutionStatus.RUNNING and self.finished_at is not None:
            raise ValueError("running executions must not have finished_at")
        if self.status != ExecutionStatus.RUNNING and self.finished_at is None:
            raise ValueError("completed executions require finished_at")
        if self.finished_at is not None and self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        if self.status == ExecutionStatus.FAILED and not self.error_message:
            raise ValueError("failed executions require an error_message")
        if self.status != ExecutionStatus.FAILED and self.error_message is not None:
            raise ValueError("error_message is only valid for failed executions")
        return self
