"""Immutable controller-visible records; no access to the live simulator state."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Observation:
    index: int
    capture_time: float
    available_time: float
    position: float
    velocity: float
    reference: float
    reference_velocity: float
    applied_action: float


@dataclass(frozen=True)
class AppliedAction:
    time: float
    value: float
    job_id: int | None


@dataclass(frozen=True)
class PolicyInput:
    time: float
    planned_apply_time: float
    observations: tuple[Observation, ...]
    actions: tuple[AppliedAction, ...]

    @property
    def latest(self) -> Observation:
        if not self.observations:
            raise ValueError("A policy needs at least one available observation")
        return self.observations[-1]
