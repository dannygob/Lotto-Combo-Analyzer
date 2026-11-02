from __future__ import annotations

from typing import Any, Literal, Sequence

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.domain import Ruleset
from src.preprocessing import detect_ruleset_change, validate_number_frame


class ValidationIssue(BaseModel):
    """
    Represents a single validation issue detected in the lottery frame.

    Attributes:
        severity: "warning" or "error".
        code: Machine-readable issue code.
        message: Human-readable description.
        row_index: Optional row index where the issue occurred.
    """
    model_config = ConfigDict(extra="forbid")

    severity: Literal["warning", "error"]
    code: str
    message: str
    row_index: int | None = None


class ValidationResult(BaseModel):
    """
    Final validation result for a lottery frame.

    Attributes:
        is_valid: Whether the frame passed validation.
        issues: List of warnings or errors.
        cleaned_rows: Cleaned and validated rows.
        ruleset_change: Optional ruleset drift information.
    """
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
    """
    Validate a candidate lottery frame and detect ruleset drift or data-quality issues.

    Steps:
        - ensure frame is not empty
        - validate numeric columns and bounds
        - detect ruleset changes if a ruleset is provided
        - return cleaned rows and issues
    """
    issues: list[ValidationIssue] = []

    # Empty frame check
    if frame.empty:
        return ValidationResult(
            is_valid=False,
            issues=[
                ValidationIssue(
                    severity="error",
                    code="empty-frame",
                    message="frame is empty",
                )
            ],
        )

    # Validate numeric structure
    try:
        cleaned = validate_number_frame(
            frame,
            number_columns=number_columns,
            minimum=minimum,
            maximum=maximum,
            expected_count=expected_count,
        )
    except ValueError as exc:
        return ValidationResult(
            is_valid=False,
            issues=[
                ValidationIssue(
                    severity="error",
                    code="validation-error",
                    message=str(exc),
                )
            ],
        )

    # Detect ruleset drift
    if ruleset is not None:
        pool_names = {pool.pool_id for pool in ruleset.pools}
        requested_names = set(number_columns or [])

        # Only detect drift if column names match pool IDs
        if requested_names and (
                requested_names.issubset(pool_names)
                or pool_names.issubset(requested_names)
        ):
            ruleset_change = detect_ruleset_change(
                ruleset,
                cleaned,
                number_columns=list(number_columns),
            )
        else:
            ruleset_change = None
    else:
        ruleset_change = None

    # Add warning if drift detected
    if ruleset_change is not None:
        issues.append(
            ValidationIssue(
                severity="warning",
                code="ruleset-change",
                message="observed draw structure differs from the active ruleset",
            )
        )

    # Final result
    return ValidationResult(
        is_valid=True,
        issues=issues,
        cleaned_rows=cleaned.to_dict(orient="records"),
        ruleset_change=ruleset_change,
    )
