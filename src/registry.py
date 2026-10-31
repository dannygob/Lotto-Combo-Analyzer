"""SQLite persistence for isolated lottery rulesets, draws, and provenance."""

from __future__ import annotations

import sqlite3
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Mapping

from src.domain import (
    CsvSchemaProfile,
    Draw,
    ExecutionLog,
    ExecutionStatus,
    Lottery,
    Ruleset,
    SourceFile,
)
from src.draw_schedule import LotterySchedule


_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS lotteries (
    lottery_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    jurisdiction TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rulesets (
    lottery_id TEXT NOT NULL,
    version TEXT NOT NULL,
    effective_from TEXT NOT NULL,
    effective_to TEXT,
    PRIMARY KEY (lottery_id, version),
    FOREIGN KEY (lottery_id) REFERENCES lotteries(lottery_id) ON DELETE RESTRICT,
    CHECK (effective_to IS NULL OR effective_to > effective_from)
);

CREATE TABLE IF NOT EXISTS ruleset_pools (
    lottery_id TEXT NOT NULL,
    ruleset_version TEXT NOT NULL,
    pool_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('primary', 'bonus', 'supplementary')),
    minimum INTEGER NOT NULL CHECK (minimum >= 0),
    maximum INTEGER NOT NULL CHECK (maximum >= minimum),
    numbers_drawn INTEGER NOT NULL CHECK (numbers_drawn > 0),
    allow_replacement INTEGER NOT NULL CHECK (allow_replacement IN (0, 1)),
    PRIMARY KEY (lottery_id, ruleset_version, pool_id),
    FOREIGN KEY (lottery_id, ruleset_version)
        REFERENCES rulesets(lottery_id, version) ON DELETE RESTRICT,
    CHECK (allow_replacement = 1 OR numbers_drawn <= maximum - minimum + 1)
);

CREATE TABLE IF NOT EXISTS source_files (
    sha256 TEXT PRIMARY KEY CHECK (length(sha256) = 64),
    original_name TEXT NOT NULL,
    storage_uri TEXT NOT NULL,
    media_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    source_url TEXT,
    registered_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS draws (
    draw_id INTEGER PRIMARY KEY,
    lottery_id TEXT NOT NULL,
    ruleset_version TEXT NOT NULL,
    draw_date TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    source_record_number INTEGER NOT NULL CHECK (source_record_number > 0),
    source_reference TEXT,
    UNIQUE (lottery_id, draw_date),
    UNIQUE (draw_id, lottery_id, ruleset_version),
    UNIQUE (source_sha256, source_record_number),
    FOREIGN KEY (lottery_id, ruleset_version)
        REFERENCES rulesets(lottery_id, version) ON DELETE RESTRICT,
    FOREIGN KEY (source_sha256) REFERENCES source_files(sha256) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS draw_numbers (
    draw_id INTEGER NOT NULL,
    lottery_id TEXT NOT NULL,
    ruleset_version TEXT NOT NULL,
    pool_id TEXT NOT NULL,
    position INTEGER NOT NULL CHECK (position >= 0),
    number INTEGER NOT NULL CHECK (number >= 0),
    PRIMARY KEY (draw_id, pool_id, position),
    UNIQUE (draw_id, pool_id, number),
    FOREIGN KEY (draw_id, lottery_id, ruleset_version)
        REFERENCES draws(draw_id, lottery_id, ruleset_version) ON DELETE CASCADE,
    FOREIGN KEY (lottery_id, ruleset_version, pool_id)
        REFERENCES ruleset_pools(lottery_id, ruleset_version, pool_id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS execution_logs (
    execution_id TEXT PRIMARY KEY,
    lottery_id TEXT,
    pipeline_name TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
    started_at TEXT NOT NULL,
    finished_at TEXT,
    code_version TEXT,
    configuration_json TEXT NOT NULL CHECK (json_valid(configuration_json)),
    error_message TEXT,
    FOREIGN KEY (lottery_id) REFERENCES lotteries(lottery_id) ON DELETE RESTRICT,
    CHECK (
        (status = 'running' AND finished_at IS NULL AND error_message IS NULL)
        OR (status = 'succeeded' AND finished_at IS NOT NULL AND error_message IS NULL)
        OR (status = 'failed' AND finished_at IS NOT NULL AND error_message IS NOT NULL)
    ),
    CHECK (finished_at IS NULL OR finished_at >= started_at)
);

CREATE TABLE IF NOT EXISTS model_versions (
    model_version TEXT PRIMARY KEY,
    lottery_id TEXT NOT NULL,
    dataset_sha256 TEXT NOT NULL,
    artifact_uri TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('candidate', 'active', 'archived', 'rejected')),
    metrics_json TEXT NOT NULL CHECK (json_valid(metrics_json)),
    created_at TEXT NOT NULL,
    activated_at TEXT,
    rejection_reason TEXT,
    FOREIGN KEY (lottery_id) REFERENCES lotteries(lottery_id) ON DELETE RESTRICT,
    FOREIGN KEY (dataset_sha256) REFERENCES source_files(sha256) ON DELETE RESTRICT,
    CHECK (
        (status = 'rejected' AND rejection_reason IS NOT NULL)
        OR (status != 'rejected' AND rejection_reason IS NULL)
    ),
    CHECK (status != 'active' OR activated_at IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS model_versions_by_lottery
    ON model_versions(lottery_id, created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS one_active_model_per_lottery
    ON model_versions(lottery_id) WHERE status = 'active';

CREATE TABLE IF NOT EXISTS dataset_versions (
    dataset_id INTEGER PRIMARY KEY,
    lottery_id TEXT NOT NULL,
    dataset_key TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    version_number INTEGER NOT NULL CHECK (version_number > 0),
    row_count INTEGER NOT NULL CHECK (row_count > 0),
    columns_json TEXT NOT NULL CHECK (json_valid(columns_json)),
    created_at TEXT NOT NULL,
    UNIQUE (lottery_id, dataset_key, source_sha256),
    UNIQUE (lottery_id, dataset_key, version_number),
    FOREIGN KEY (lottery_id) REFERENCES lotteries(lottery_id) ON DELETE RESTRICT,
    FOREIGN KEY (source_sha256) REFERENCES source_files(sha256) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS training_runs (
    run_id TEXT PRIMARY KEY,
    dataset_id INTEGER NOT NULL,
    model_version TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL CHECK (status IN ('candidate', 'active', 'archived')),
    metrics_json TEXT NOT NULL CHECK (json_valid(metrics_json)),
    created_at TEXT NOT NULL,
    FOREIGN KEY (dataset_id) REFERENCES dataset_versions(dataset_id) ON DELETE RESTRICT,
    FOREIGN KEY (model_version) REFERENCES model_versions(model_version) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS lottery_schedules (
    lottery_id TEXT PRIMARY KEY,
    schedule_json TEXT NOT NULL CHECK (json_valid(schedule_json)),
    updated_at TEXT NOT NULL,
    FOREIGN KEY (lottery_id) REFERENCES lotteries(lottery_id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS csv_schema_profiles (
    lottery_id TEXT NOT NULL,
    ruleset_version TEXT NOT NULL,
    profile_json TEXT NOT NULL CHECK (json_valid(profile_json)),
    updated_at TEXT NOT NULL,
    PRIMARY KEY (lottery_id, ruleset_version),
    FOREIGN KEY (lottery_id, ruleset_version)
        REFERENCES rulesets(lottery_id, version) ON DELETE RESTRICT
);

CREATE TRIGGER IF NOT EXISTS rulesets_no_overlapping_insert
BEFORE INSERT ON rulesets
WHEN EXISTS (
    SELECT 1 FROM rulesets existing
    WHERE existing.lottery_id = NEW.lottery_id
      AND COALESCE(existing.effective_to, '9999-12-31') > NEW.effective_from
      AND COALESCE(NEW.effective_to, '9999-12-31') > existing.effective_from
)
BEGIN
    SELECT RAISE(ABORT, 'ruleset date ranges overlap for this lottery');
END;

CREATE TRIGGER IF NOT EXISTS rulesets_no_overlapping_update
BEFORE UPDATE OF lottery_id, effective_from, effective_to ON rulesets
WHEN EXISTS (
    SELECT 1 FROM rulesets existing
    WHERE existing.lottery_id = NEW.lottery_id
      AND existing.version != OLD.version
      AND COALESCE(existing.effective_to, '9999-12-31') > NEW.effective_from
      AND COALESCE(NEW.effective_to, '9999-12-31') > existing.effective_from
)
BEGIN
    SELECT RAISE(ABORT, 'ruleset date ranges overlap for this lottery');
END;

CREATE TRIGGER IF NOT EXISTS draws_within_ruleset_range
BEFORE INSERT ON draws
WHEN NOT EXISTS (
    SELECT 1 FROM rulesets r
    WHERE r.lottery_id = NEW.lottery_id
      AND r.version = NEW.ruleset_version
      AND r.effective_from <= NEW.draw_date
      AND (r.effective_to IS NULL OR NEW.draw_date < r.effective_to)
)
BEGIN
    SELECT RAISE(ABORT, 'draw date is outside the bound ruleset range');
END;
"""


class SQLiteRegistry:
    """Transactional SQLite registry; each operation opens and closes a connection."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = str(database_path)
        if self.database_path == ":memory:":
            raise ValueError("use a filesystem database path; registry operations use separate connections")
        if self.database_path != ":memory:":
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        """Create registry tables and integrity triggers if they do not exist."""
        with self._connection() as connection:
            connection.executescript(_SCHEMA)
            self._migrate_training_runs_for_schema_versions(connection)

    @staticmethod
    def _migrate_training_runs_for_schema_versions(connection: sqlite3.Connection) -> None:
        indexes = connection.execute("PRAGMA index_list(training_runs)").fetchall()
        for index in indexes:
            if not index["unique"]:
                continue
            indexed_columns = [
                row["name"]
                for row in connection.execute(f"PRAGMA index_info({index['name']})")
            ]
            if indexed_columns != ["dataset_id"]:
                continue
            connection.execute("ALTER TABLE training_runs RENAME TO training_runs_legacy")
            connection.execute(
                """
                CREATE TABLE training_runs (
                    run_id TEXT PRIMARY KEY,
                    dataset_id INTEGER NOT NULL,
                    model_version TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL CHECK (status IN ('candidate', 'active', 'archived')),
                    metrics_json TEXT NOT NULL CHECK (json_valid(metrics_json)),
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (dataset_id) REFERENCES dataset_versions(dataset_id) ON DELETE RESTRICT,
                    FOREIGN KEY (model_version) REFERENCES model_versions(model_version) ON DELETE RESTRICT
                )
                """
            )
            connection.execute(
                """
                INSERT INTO training_runs (
                    run_id, dataset_id, model_version, status, metrics_json, created_at
                )
                SELECT run_id, dataset_id, model_version, status, metrics_json, created_at
                FROM training_runs_legacy
                """
            )
            connection.execute("DROP TABLE training_runs_legacy")
            break

    def register_lottery(self, lottery: Lottery) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO lotteries (lottery_id, name, jurisdiction) VALUES (?, ?, ?)",
                (lottery.lottery_id, lottery.name, lottery.jurisdiction),
            )

    def register_ruleset(self, ruleset: Ruleset) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO rulesets (lottery_id, version, effective_from, effective_to)
                VALUES (?, ?, ?, ?)
                """,
                (
                    ruleset.lottery_id,
                    ruleset.version,
                    ruleset.effective_from.isoformat(),
                    ruleset.effective_to.isoformat() if ruleset.effective_to else None,
                ),
            )
            connection.executemany(
                """
                INSERT INTO ruleset_pools (
                    lottery_id, ruleset_version, pool_id, kind, minimum, maximum,
                    numbers_drawn, allow_replacement
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        ruleset.lottery_id,
                        ruleset.version,
                        pool.pool_id,
                        pool.kind.value,
                        pool.minimum,
                        pool.maximum,
                        pool.numbers_drawn,
                        int(pool.allow_replacement),
                    )
                    for pool in ruleset.pools
                ],
            )

    def save_lottery_schedule(self, schedule: LotterySchedule) -> None:
        payload = json.dumps(
            schedule.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO lottery_schedules (lottery_id, schedule_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(lottery_id) DO UPDATE SET
                    schedule_json = excluded.schedule_json,
                    updated_at = excluded.updated_at
                """,
                (schedule.lottery_id, payload, datetime.now(timezone.utc).isoformat()),
            )

    def get_lottery_schedule(self, lottery_id: str) -> LotterySchedule | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT schedule_json FROM lottery_schedules WHERE lottery_id = ?",
                (lottery_id,),
            ).fetchone()
        if row is None:
            return None
        return LotterySchedule.model_validate_json(row["schedule_json"])

    def save_csv_schema_profile(self, profile: CsvSchemaProfile) -> None:
        payload = json.dumps(
            profile.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO csv_schema_profiles (
                    lottery_id, ruleset_version, profile_json, updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(lottery_id, ruleset_version) DO UPDATE SET
                    profile_json = excluded.profile_json,
                    updated_at = excluded.updated_at
                """,
                (
                    profile.lottery_id,
                    profile.ruleset_version,
                    payload,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def get_csv_schema_profile(
        self,
        lottery_id: str,
        ruleset_version: str,
    ) -> CsvSchemaProfile | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT profile_json FROM csv_schema_profiles
                WHERE lottery_id = ? AND ruleset_version = ?
                """,
                (lottery_id, ruleset_version),
            ).fetchone()
        if row is None:
            return None
        return CsvSchemaProfile.model_validate_json(row["profile_json"])

    def register_source_file(self, source_file: SourceFile) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO source_files (
                    sha256, original_name, storage_uri, media_type, size_bytes,
                    source_url, registered_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_file.sha256,
                    source_file.original_name,
                    source_file.storage_uri,
                    source_file.media_type,
                    source_file.size_bytes,
                    source_file.source_url,
                    source_file.registered_at.isoformat(),
                ),
            )

    def register_source_file_if_missing(self, source_file: SourceFile) -> None:
        with self._connection() as connection:
            existing = connection.execute(
                "SELECT 1 FROM source_files WHERE sha256 = ?",
                (source_file.sha256,),
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO source_files (
                        sha256, original_name, storage_uri, media_type, size_bytes,
                        source_url, registered_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source_file.sha256,
                        source_file.original_name,
                        source_file.storage_uri,
                        source_file.media_type,
                        source_file.size_bytes,
                        source_file.source_url,
                        source_file.registered_at.isoformat(),
                    ),
                )

    def get_dataset_version(
        self, *, lottery_id: str, dataset_key: str, source_sha256: str
    ) -> dict[str, object] | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT dataset_id, lottery_id, dataset_key, source_sha256,
                       version_number, row_count, columns_json, created_at
                FROM dataset_versions
                WHERE lottery_id = ? AND dataset_key = ? AND source_sha256 = ?
                """,
                (lottery_id, dataset_key, source_sha256.lower()),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["columns"] = json.loads(result.pop("columns_json"))
        return result

    def register_dataset_version(
        self,
        *,
        lottery_id: str,
        dataset_key: str,
        source_sha256: str,
        row_count: int,
        columns: list[str],
    ) -> dict[str, object]:
        if not dataset_key.strip():
            raise ValueError("dataset_key must not be empty")
        if row_count <= 0:
            raise ValueError("row_count must be positive")
        columns_json = json.dumps(columns, separators=(",", ":"), allow_nan=False)
        with self._connection() as connection:
            existing = connection.execute(
                """
                SELECT dataset_id, lottery_id, dataset_key, source_sha256,
                       version_number, row_count, columns_json, created_at
                FROM dataset_versions
                WHERE lottery_id = ? AND dataset_key = ? AND source_sha256 = ?
                """,
                (lottery_id, dataset_key, source_sha256.lower()),
            ).fetchone()
            if existing is not None:
                raise ValueError("dataset version is already registered")
            version_row = connection.execute(
                """
                SELECT COALESCE(MAX(version_number), 0) + 1 AS next_version
                FROM dataset_versions
                WHERE lottery_id = ? AND dataset_key = ?
                """,
                (lottery_id, dataset_key),
            ).fetchone()
            version_number = int(version_row["next_version"])
            created_at = datetime.now(timezone.utc).isoformat()
            cursor = connection.execute(
                """
                INSERT INTO dataset_versions (
                    lottery_id, dataset_key, source_sha256, version_number,
                    row_count, columns_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    lottery_id,
                    dataset_key,
                    source_sha256.lower(),
                    version_number,
                    row_count,
                    columns_json,
                    created_at,
                ),
            )
            dataset_id = int(cursor.lastrowid)
        return {
            "dataset_id": dataset_id,
            "lottery_id": lottery_id,
            "dataset_key": dataset_key,
            "source_sha256": source_sha256.lower(),
            "version_number": version_number,
            "row_count": row_count,
            "columns": columns,
            "created_at": created_at,
        }

    def record_training_run(
        self,
        *,
        run_id: str,
        dataset_id: int,
        model_version: str,
        status: str,
        metrics: Mapping[str, object],
    ) -> None:
        if status not in {"candidate", "active", "archived"}:
            raise ValueError("training run status must be candidate, active, or archived")
        metrics_json = json.dumps(dict(metrics), sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO training_runs (
                    run_id, dataset_id, model_version, status, metrics_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    dataset_id,
                    model_version,
                    status,
                    metrics_json,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def list_dataset_versions(self, lottery_id: str) -> list[dict[str, object]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT dataset_id, lottery_id, dataset_key, source_sha256,
                       version_number, row_count, columns_json, created_at
                FROM dataset_versions
                WHERE lottery_id = ?
                ORDER BY dataset_key, version_number DESC
                """,
                (lottery_id,),
            ).fetchall()
        return [
            {**dict(row), "columns": json.loads(row["columns_json"])}
            for row in rows
        ]

    def list_training_runs(self, lottery_id: str) -> list[dict[str, object]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT tr.run_id, tr.dataset_id, dv.dataset_key, dv.version_number,
                       tr.model_version, tr.status, tr.metrics_json, tr.created_at
                FROM training_runs tr
                JOIN dataset_versions dv ON dv.dataset_id = tr.dataset_id
                WHERE dv.lottery_id = ?
                ORDER BY tr.created_at DESC
                """,
                (lottery_id,),
            ).fetchall()
        return [
            {**dict(row), "metrics": json.loads(row["metrics_json"])}
            for row in rows
        ]

    def add_draw(self, draw: Draw) -> int:
        """Validate a draw against its registered, date-bound ruleset and persist it."""
        with self._connection() as connection:
            ruleset = connection.execute(
                """
                SELECT r.effective_from, r.effective_to
                FROM rulesets r
                WHERE r.lottery_id = ? AND r.version = ?
                """,
                (draw.lottery_id, draw.ruleset_version),
            ).fetchone()
            if ruleset is None:
                raise ValueError("draw references an unregistered lottery ruleset")
            draw_date = draw.draw_date.isoformat()
            if draw_date < ruleset["effective_from"] or (
                ruleset["effective_to"] is not None and draw_date >= ruleset["effective_to"]
            ):
                raise ValueError("draw date is outside the bound ruleset range")

            pool_rows = connection.execute(
                """
                SELECT pool_id, minimum, maximum, numbers_drawn, allow_replacement
                FROM ruleset_pools
                WHERE lottery_id = ? AND ruleset_version = ?
                """,
                (draw.lottery_id, draw.ruleset_version),
            ).fetchall()
            pools = {row["pool_id"]: row for row in pool_rows}
            if set(draw.numbers) != set(pools):
                raise ValueError("draw pools must exactly match the registered ruleset pools")

            for pool_id, numbers in draw.numbers.items():
                pool = pools[pool_id]
                if len(numbers) != pool["numbers_drawn"]:
                    raise ValueError(f"pool {pool_id!r} requires {pool['numbers_drawn']} numbers")
                if any(number < pool["minimum"] or number > pool["maximum"] for number in numbers):
                    raise ValueError(f"pool {pool_id!r} contains a number outside its allowed range")
                if not pool["allow_replacement"] and len(numbers) != len(set(numbers)):
                    raise ValueError(f"pool {pool_id!r} does not allow replacement")

            cursor = connection.execute(
                """
                INSERT INTO draws (
                    lottery_id, ruleset_version, draw_date, source_sha256,
                    source_record_number, source_reference
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    draw.lottery_id,
                    draw.ruleset_version,
                    draw_date,
                    draw.source_sha256,
                    draw.source_record_number,
                    draw.source_reference,
                ),
            )
            draw_id = int(cursor.lastrowid)
            connection.executemany(
                """
                INSERT INTO draw_numbers (
                    draw_id, lottery_id, ruleset_version, pool_id, position, number
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (draw_id, draw.lottery_id, draw.ruleset_version, pool_id, position, number)
                    for pool_id, numbers in draw.numbers.items()
                    for position, number in enumerate(numbers)
                ],
            )
            return draw_id

    def record_execution(self, execution: ExecutionLog) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO execution_logs (
                    execution_id, lottery_id, pipeline_name, status, started_at,
                    finished_at, code_version, configuration_json, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    execution.execution_id,
                    execution.lottery_id,
                    execution.pipeline_name,
                    execution.status.value,
                    execution.started_at.isoformat(),
                    execution.finished_at.isoformat() if execution.finished_at else None,
                    execution.code_version,
                    execution.configuration_json,
                    execution.error_message,
                ),
            )

    def finish_execution(
        self,
        execution_id: str,
        status: ExecutionStatus,
        finished_at: datetime,
        error_message: str | None = None,
    ) -> None:
        """Complete a running execution exactly once with validated terminal state."""
        if status == ExecutionStatus.RUNNING:
            raise ValueError("finish_execution requires a terminal execution status")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM execution_logs WHERE execution_id = ?",
                (execution_id,),
            ).fetchone()
            if row is None:
                raise ValueError("execution does not exist")
            if row["status"] != ExecutionStatus.RUNNING.value:
                raise ValueError("execution is already complete")

            completed = ExecutionLog(
                execution_id=row["execution_id"],
                lottery_id=row["lottery_id"],
                pipeline_name=row["pipeline_name"],
                status=status,
                started_at=datetime.fromisoformat(row["started_at"]),
                finished_at=finished_at,
                code_version=row["code_version"],
                configuration_json=row["configuration_json"],
                error_message=error_message,
            )
            result = connection.execute(
                """
                UPDATE execution_logs
                SET status = ?, finished_at = ?, error_message = ?
                WHERE execution_id = ? AND status = 'running'
                """,
                (
                    completed.status.value,
                    completed.finished_at.isoformat(),
                    completed.error_message,
                    completed.execution_id,
                ),
            )
            if result.rowcount != 1:
                raise ValueError("execution was concurrently completed")

    def list_lotteries(self) -> list[dict[str, str]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT lottery_id, name, jurisdiction FROM lotteries ORDER BY lottery_id"
            ).fetchall()
            return [dict(row) for row in rows]

    def list_rulesets(self, lottery_id: str) -> list[dict[str, object]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT lottery_id, version, effective_from, effective_to
                FROM rulesets
                WHERE lottery_id = ?
                ORDER BY effective_from DESC, version DESC
                """,
                (lottery_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_ruleset(self, lottery_id: str, version: str) -> Ruleset | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT lottery_id, version, effective_from, effective_to
                FROM rulesets
                WHERE lottery_id = ? AND version = ?
                """,
                (lottery_id, version),
            ).fetchone()
            if row is None:
                return None
            pool_rows = connection.execute(
                """
                SELECT pool_id, kind, minimum, maximum, numbers_drawn, allow_replacement
                FROM ruleset_pools
                WHERE lottery_id = ? AND ruleset_version = ?
                ORDER BY pool_id
                """,
                (lottery_id, version),
            ).fetchall()
        return Ruleset.model_validate(
            {
                **dict(row),
                "pools": [
                    {
                        **dict(pool),
                        "allow_replacement": bool(pool["allow_replacement"]),
                    }
                    for pool in pool_rows
                ],
            }
        )

    def list_draws(self, lottery_id: str, limit: int = 100) -> list[dict[str, object]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT draw_id, lottery_id, ruleset_version, draw_date, source_sha256,
                       source_record_number, source_reference
                FROM draws
                WHERE lottery_id = ?
                ORDER BY draw_date DESC, draw_id DESC
                LIMIT ?
                """,
                (lottery_id, limit),
            ).fetchall()
            return [dict(row) for row in rows]

    def register_model_version(
        self,
        *,
        model_version: str,
        lottery_id: str,
        dataset_sha256: str,
        artifact_uri: str,
        metrics: Mapping[str, object],
    ) -> None:
        """Register a model artifact as a candidate; promotion is a separate gated operation."""
        if not model_version.strip() or not artifact_uri.strip():
            raise ValueError("model_version and artifact_uri must not be empty")
        if not isinstance(metrics, Mapping):
            raise ValueError("metrics must be a mapping")
        metrics_json = json.dumps(dict(metrics), sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO model_versions (
                    model_version, lottery_id, dataset_sha256, artifact_uri, status,
                    metrics_json, created_at
                ) VALUES (?, ?, ?, ?, 'candidate', ?, ?)
                """,
                (
                    model_version,
                    lottery_id,
                    dataset_sha256.lower(),
                    artifact_uri,
                    metrics_json,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def promote_model_version(self, model_version: str) -> None:
        """Atomically promote a candidate only when all scientific release gates pass."""
        required_gates = (
            "validation_passed",
            "leakage_free",
            "calibration_passed",
            "beats_null",
            "drift_within_tolerance",
        )
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT lottery_id, status, metrics_json FROM model_versions WHERE model_version = ?",
                (model_version,),
            ).fetchone()
            if row is None:
                raise ValueError("model version does not exist")
            if row["status"] != "candidate":
                raise ValueError("only candidate model versions can be promoted")
            metrics = json.loads(row["metrics_json"])
            failed_gates = [gate for gate in required_gates if metrics.get(gate) is not True]
            if failed_gates:
                raise ValueError(f"model promotion gates failed: {', '.join(failed_gates)}")

            activated_at = datetime.now(timezone.utc).isoformat()
            connection.execute(
                """
                UPDATE model_versions
                SET status = 'archived'
                WHERE lottery_id = ? AND status = 'active'
                """,
                (row["lottery_id"],),
            )
            connection.execute(
                """
                UPDATE training_runs
                SET status = 'archived'
                WHERE model_version IN (
                    SELECT model_version FROM model_versions
                    WHERE lottery_id = ? AND status = 'archived'
                ) AND status = 'active'
                """,
                (row["lottery_id"],),
            )
            result = connection.execute(
                """
                UPDATE model_versions
                SET status = 'active', activated_at = ?
                WHERE model_version = ? AND status = 'candidate'
                """,
                (activated_at, model_version),
            )
            if result.rowcount != 1:
                raise ValueError("candidate model version was concurrently changed")

    def reject_model_version(self, model_version: str, reason: str) -> None:
        if not reason.strip():
            raise ValueError("rejection reason must not be empty")
        with self._connection() as connection:
            result = connection.execute(
                """
                UPDATE model_versions
                SET status = 'rejected', rejection_reason = ?
                WHERE model_version = ? AND status = 'candidate'
                """,
                (reason, model_version),
            )
            if result.rowcount != 1:
                raise ValueError("model version does not exist or is not a candidate")

    def list_model_versions(self, lottery_id: str) -> list[dict[str, object]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT model_version, lottery_id, dataset_sha256, artifact_uri, status,
                       metrics_json, created_at, activated_at, rejection_reason
                FROM model_versions
                WHERE lottery_id = ?
                ORDER BY created_at DESC, model_version DESC
                """,
                (lottery_id,),
            ).fetchall()
        return [
            {**dict(row), "metrics": json.loads(row["metrics_json"])}
            for row in rows
        ]

    def get_draw(self, draw_id: int) -> dict[str, object] | None:
        """Return persisted draw metadata and its ordered number groups."""
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT draw_id, lottery_id, ruleset_version, draw_date, source_sha256,
                       source_record_number, source_reference
                FROM draws WHERE draw_id = ?
                """,
                (draw_id,),
            ).fetchone()
            if row is None:
                return None
            number_rows = connection.execute(
                """
                SELECT pool_id, number FROM draw_numbers
                WHERE draw_id = ? ORDER BY pool_id, position
                """,
                (draw_id,),
            ).fetchall()

        grouped: dict[str, list[int]] = {}
        for number_row in number_rows:
            grouped.setdefault(number_row["pool_id"], []).append(number_row["number"])
        return {
            **dict(row),
            "numbers": {pool_id: tuple(numbers) for pool_id, numbers in grouped.items()},
        }
