from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ScheduleSession(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
    local_time: str = Field(pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")
    weekdays: tuple[int, ...] | None = None
    every_n_days: int | None = Field(default=None, ge=1, le=366)
    anchor_date: date | None = None
    fold: int = Field(default=0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_recurrence(self) -> ScheduleSession:
        if self.weekdays is not None:
            if not self.weekdays or any(day < 0 or day > 6 for day in self.weekdays):
                raise ValueError("weekdays must contain values from 0 (Monday) to 6 (Sunday)")
        if (self.weekdays is None) == (self.every_n_days is None):
            raise ValueError("configure exactly one recurrence: weekdays or every_n_days")
        if self.every_n_days is not None and self.anchor_date is None:
            raise ValueError("anchor_date is required for every_n_days recurrence")
        if self.weekdays is not None and self.anchor_date is not None:
            raise ValueError("anchor_date is only valid for every_n_days recurrence")
        return self

    def occurs_on(self, event_date: date) -> bool:
        if self.weekdays is not None:
            return event_date.weekday() in self.weekdays
        assert self.anchor_date is not None and self.every_n_days is not None
        delta = (event_date - self.anchor_date).days
        return delta >= 0 and delta % self.every_n_days == 0


class ScheduleException(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_date: date
    session_id: str = Field(min_length=1, max_length=64)
    action: Literal["cancel", "add"]
    local_time: str | None = Field(default=None, pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")
    fold: int = Field(default=0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_action(self) -> ScheduleException:
        if self.action == "add" and self.local_time is None:
            raise ValueError("added schedule exceptions require local_time")
        if self.action == "cancel" and self.local_time is not None:
            raise ValueError("cancelled schedule exceptions must not set local_time")
        return self


class LotterySchedule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lottery_id: str = Field(min_length=1, max_length=64)
    timezone: str = Field(min_length=1, max_length=100)
    effective_from: date
    effective_to: date | None = None
    sessions: tuple[ScheduleSession, ...] = Field(min_length=1)
    exceptions: tuple[ScheduleException, ...] = ()

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown IANA timezone {value!r}") from exc
        return value

    @model_validator(mode="after")
    def validate_schedule(self) -> LotterySchedule:
        if self.effective_to is not None and self.effective_to <= self.effective_from:
            raise ValueError("effective_to must be after effective_from")
        session_ids = [session.session_id for session in self.sessions]
        if len(session_ids) != len(set(session_ids)):
            raise ValueError("session_id values must be unique")
        exception_keys = [(item.event_date, item.session_id) for item in self.exceptions]
        if len(exception_keys) != len(set(exception_keys)):
            raise ValueError("only one exception per event date and session is allowed")
        known_sessions = set(session_ids)
        if any(item.session_id not in known_sessions for item in self.exceptions):
            raise ValueError("schedule exceptions must reference a configured session")
        return self


def _local_event_datetime(
    event_date: date,
    local_time: str,
    *,
    zone: ZoneInfo,
    fold: int,
) -> datetime:
    local_value = datetime.combine(event_date, time.fromisoformat(local_time))
    aware = local_value.replace(tzinfo=zone, fold=fold)
    round_trip = aware.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None)
    if round_trip != local_value:
        raise ValueError(
            f"scheduled local time {local_value.isoformat()} does not exist in timezone {zone.key}"
        )
    return aware


def local_event_datetime(
    event_date: date,
    local_time: str,
    *,
    timezone_name: str,
    fold: int = 0,
) -> datetime:
    """Convert a configured local wall time to a validated timezone-aware datetime."""
    return _local_event_datetime(
        event_date,
        local_time,
        zone=ZoneInfo(timezone_name),
        fold=fold,
    )


def scheduled_session_datetime(
    schedule: LotterySchedule,
    event_date: date,
    session_id: str,
    *,
    local_time_override: str | None = None,
) -> datetime:
    """Resolve and validate one historical event against a configured session schedule."""
    session = next(
        (item for item in schedule.sessions if item.session_id == session_id),
        None,
    )
    if session is None:
        raise ValueError(f"session {session_id!r} is not configured for this lottery")

    within_effective_range = event_date >= schedule.effective_from and (
        schedule.effective_to is None or event_date < schedule.effective_to
    )
    exception = next(
        (
            item
            for item in schedule.exceptions
            if item.event_date == event_date and item.session_id == session_id
        ),
        None,
    )
    if within_effective_range:
        if exception is not None and exception.action == "cancel":
            raise ValueError(
                f"CSV contains event {event_date.isoformat()}:{session_id}, "
                "which is cancelled in the configured schedule"
            )
        if exception is None and not session.occurs_on(event_date):
            raise ValueError(
                f"session {session_id!r} is not scheduled on {event_date.isoformat()}"
            )

    local_time = (
        local_time_override
        or (exception.local_time if exception is not None and exception.action == "add" else None)
        or session.local_time
    )
    fold = exception.fold if exception is not None else session.fold
    return local_event_datetime(
        event_date,
        local_time,
        timezone_name=schedule.timezone,
        fold=fold,
    )


def next_scheduled_event(schedule: LotterySchedule, after: datetime | None = None) -> dict[str, str]:
    """Resolve the next official scheduled event in the lottery's local timezone."""
    zone = ZoneInfo(schedule.timezone)
    if after is None:
        after = datetime.now(zone)
    elif after.tzinfo is None or after.utcoffset() is None:
        raise ValueError("after must be timezone-aware")
    after_local = after.astimezone(zone)
    exceptions = {(item.event_date, item.session_id): item for item in schedule.exceptions}

    for day_offset in range(0, 732):
        event_date = after_local.date() + timedelta(days=day_offset)
        if event_date < schedule.effective_from or (
            schedule.effective_to is not None and event_date >= schedule.effective_to
        ):
            continue
        candidates: list[tuple[datetime, str]] = []
        for session in schedule.sessions:
            exception = exceptions.get((event_date, session.session_id))
            occurs = session.occurs_on(event_date)
            if exception is not None:
                if exception.action == "cancel":
                    continue
                local_time = exception.local_time
                fold = exception.fold
            elif occurs:
                local_time = session.local_time
                fold = session.fold
            else:
                continue
            event_datetime = _local_event_datetime(
                event_date,
                local_time,
                zone=zone,
                fold=fold,
            )
            if event_datetime.astimezone(timezone.utc) > after.astimezone(timezone.utc):
                candidates.append((event_datetime, session.session_id))
        if candidates:
            event_datetime, session_id = min(
                candidates,
                key=lambda candidate: (
                    candidate[0].astimezone(timezone.utc),
                    candidate[1],
                ),
            )
            event_id = f"{schedule.lottery_id}:{event_date.isoformat()}:{session_id}"
            return {
                "event_id": event_id,
                "lottery_id": schedule.lottery_id,
                "event_date": event_date.isoformat(),
                "session_id": session_id,
                "scheduled_at": event_datetime.isoformat(),
                "timezone": schedule.timezone,
            }
    raise ValueError("no scheduled event found within the next two years")
