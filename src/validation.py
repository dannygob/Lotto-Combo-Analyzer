from __future__ import annotations

from typing import Any, Literal, Sequence

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.domain import Ruleset
from src.preprocessing import detect_ruleset_change, validate_number_frame


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    severity: Literal["warning", "error"]
    code: str
    message: str
    row_index: int | None = None


class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    cleaned_rows: list[dict[str, Any]] = Field(default_factory=list)
    ruleset_change: dict[str, Any] | None = None


def validate_lottery_frame(
    frame: pd.DataFrame,
    ruleset: Ruleset | None = None,
    *,
    number_columns: Sequence[str] | None = None,
    minimum: int = 1,
    maximum: int = 99,
    expected_count: int | None = None,
) -> ValidationResult:
    """Validate a candidate lottery frame and return any ruleset drift or data-quality issues."""
    issues: list[ValidationIssue] = []
    if frame.empty:
        return ValidationResult(
            is_valid=False,
            issues=[ValidationIssue(severity="error", code="empty-frame", message="frame is empty")],
        )
    try:
        cleaned = validate_number_frame(
            frame,
            number_columns=number_columns,
            minimum=minimum,
            maximum=maximum,
            expected_count=expected_count,
        )
    except ValueError as exc:  # pragma: no cover - exercised by tests
        return ValidationResult(
            is_valid=False,
            issues=[ValidationIssue(severity="error", code="validation-error", message=str(exc))],
        )

    if ruleset is not None:
        pool_names = {pool.pool_id for pool in ruleset.pools}
        requested_names = set(number_columns or [])
        if requested_names and (requested_names.issubset(pool_names) or pool_names.issubset(requested_names)):
            ruleset_change = detect_ruleset_change(ruleset, cleaned, number_columns=list(number_columns))
        else:
            ruleset_change = None
    else:
        ruleset_change = None

    if ruleset_change is not None:
        issues.append(
            ValidationIssue(
                severity="warning",
                code="ruleset-change",
                message="observed draw structure differs from the active ruleset",
            )
        )
    return ValidationResult(
        is_valid=True,
        issues=issues,
        cleaned_rows=cleaned.to_dict(orient="records"),
        ruleset_change=ruleset_change,
    )
