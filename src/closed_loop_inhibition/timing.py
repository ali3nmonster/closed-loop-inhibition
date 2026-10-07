"""Event-driven physical time with causally restricted controller snapshots.

The policy is evaluated from an immutable start-time snapshot. Its result cannot
affect the plant until computation and actuator latency have elapsed. Virtual
time deliberately does not depend on the wall-clock cost of evaluating Python.
"""

from collections import deque
from dataclasses import dataclass, asdict
import heapq
import itertools
import math
from typing import Callable

import numpy as np

from .plants import Oscillator
from .records import AppliedAction, Observation, PolicyInput
from .signals import PiecewiseConstant

_NS = 1_000_000_000
_PRIORITY = {
    "reference": 0,
    "disturbance": 0,
    "apply": 1,
    "capture": 2,
    "available": 3,
    "complete": 4,
    "dispatch": 5,
    "sample": 6,
}


def _ticks(seconds: float) -> int:
    return int(round(seconds * _NS))


@dataclass(frozen=True)
class TimingConfig:
    duration: float = 12.0
    observation_interval: float = 0.01
    compute_duration: float = 0.05
    sensor_delay: float = 0.0
    actuator_delay: float = 0.0
    schedule: str = "serial"
    decision_interval: float = 0.01
    history_seconds: float = 1.0
    sample_interval: float = 0.01
    action_limit: float | None = None
    max_abs_state: float = 1e6

    def __post_init__(self) -> None:
        positive = ("duration", "observation_interval", "decision_interval",
                    "history_seconds", "sample_interval", "max_abs_state")
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            if name != "max_abs_state" and _ticks(value) < 1:
                raise ValueError(f"{name} is below the one-nanosecond clock resolution")
        for name in ("compute_duration", "sensor_delay", "actuator_delay"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.schedule not in {"serial", "fixed_cadence"}:
            raise ValueError("schedule must be 'serial' or 'fixed_cadence'")
        if self.action_limit is not None:
            if not math.isfinite(self.action_limit) or self.action_limit <= 0:
                raise ValueError("action_limit must be finite and positive or None")


@dataclass
class SimulationResult:
    samples: list[dict]
    events: list[dict]
    jobs: list[dict]
    applied_actions: list[AppliedAction]
    config: TimingConfig
    failure_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "config": asdict(self.config),
            "samples": self.samples,
            "events": self.events,
            "jobs": self.jobs,
            "applied_actions": [asdict(action) for action in self.applied_actions],
            "failure_reason": self.failure_reason,
        }


def simulate(
    plant: Oscillator,
    policy: Callable[[PolicyInput], float],
    config: TimingConfig,
    initial_state: tuple[float, float] = (0.0, 0.0),
    reference: PiecewiseConstant | None = None,
    disturbance: PiecewiseConstant | None = None,
) -> SimulationResult:
    """Run one rollout; policies cannot access the live state or future signals.

    At equal timestamps: signal changes, action application, sensing, sensor
    availability, inference completion, dispatch, reporting. Causal zero-delay
    children execute immediately according to their event priority. Sensor
    capture thus precedes a newly dispatched zero-delay result at that time.
    The fixed-cadence schedule has idealized parallel throughput and is valid
    for stateless snapshot policies, not an unspecified shared mutable cache.
    The divergence guard is checked at event times; it is not continuous root
    finding. Near the guard, verify censoring under a finer reporting grid.
    """
    reference = reference if reference is not None else PiecewiseConstant()
    disturbance = disturbance if disturbance is not None else PiecewiseConstant()
    state = np.asarray(initial_state, dtype=float).copy()
    if state.shape != (2,) or not np.all(np.isfinite(state)):
        raise ValueError("initial_state must contain two finite values")
    horizon = _ticks(config.duration)
    compute_ticks = _ticks(config.compute_duration)
    sensor_ticks = _ticks(config.sensor_delay)
    actuator_ticks = _ticks(config.actuator_delay)
    history_ticks = _ticks(config.history_seconds)
    queue: list[tuple] = []
    order = itertools.count()
    result = SimulationResult([], [], [], [AppliedAction(0.0, 0.0, None)], config)
    available: deque[Observation] = deque()
    busy: int | None = None
    last_used_index = -1
    time_ticks = 0
    action = 0.0
    reference_value = float(reference.initial)
    disturbance_value = float(disturbance.initial)

    def push(when: int, kind: str, payload=None) -> None:
        if when <= horizon:
            heapq.heappush(queue, (when, _PRIORITY[kind], next(order), kind, payload))

    for index, when in enumerate(range(0, horizon + 1, _ticks(config.observation_interval))):
        push(when, "capture", index)
    reporting_ticks = list(range(0, horizon + 1, _ticks(config.sample_interval)))
    if reporting_ticks[-1] != horizon:
        reporting_ticks.append(horizon)
    for when in reporting_ticks:
        push(when, "sample")
    if config.schedule == "fixed_cadence":
        for when in range(0, horizon + 1, _ticks(config.decision_interval)):
            push(when, "dispatch")
    for kind, signal in (("reference", reference), ("disturbance", disturbance)):
        previous_tick = -1
        for when, value in signal.changes:
            when_tick = _ticks(when)
            if when_tick <= previous_tick:
                raise ValueError("Distinct signal changes collide at clock resolution")
            previous_tick = when_tick
            push(when_tick, kind, value)

    while queue:
        when, _, _, kind, payload = heapq.heappop(queue)
        if when != time_ticks:
            state = plant.propagate(state, (when - time_ticks) / _NS,
                                    control=action, disturbance=disturbance_value)
            time_ticks = when
        time = when / _NS
        if not np.all(np.isfinite(state)) or np.max(np.abs(state)) > config.max_abs_state:
            result.failure_reason = f"state_guard_exceeded_at_{time:.9f}"
            result.events.append({"time": time, "kind": "failure", "reason": result.failure_reason})
            break

        if kind in {"reference", "disturbance"}:
            if kind == "reference":
                reference_value = float(payload)
            else:
                disturbance_value = float(payload)
            result.events.append({"time": time, "kind": kind, "value": float(payload)})

        elif kind == "capture":
            observation = Observation(
                index=payload, capture_time=time,
                available_time=(when + sensor_ticks) / _NS,
                position=float(state[0]), velocity=float(state[1]),
                reference=reference_value, reference_velocity=0.0,
                applied_action=action,
            )
            result.events.append({"time": time, "kind": "capture", **asdict(observation)})
            push(when + sensor_ticks, "available", observation)

        elif kind == "available":
            available.append(payload)
            cutoff = _ticks(payload.capture_time) - history_ticks
            dropped = []
            while available and _ticks(available[0].capture_time) < cutoff:
                dropped.append(available.popleft().index)
            result.events.append({"time": time, "kind": "available", "observation_index": payload.index,
                                  "expired_history_indices": dropped})
            if config.schedule == "serial":
                push(when, "dispatch")

        elif kind == "dispatch":
            if busy is not None and config.schedule == "serial":
                continue
            if not available or available[-1].index <= last_used_index:
                continue
            job_id = len(result.jobs)
            complete_time = (when + compute_ticks) / _NS
            apply_time = (when + compute_ticks + actuator_ticks) / _NS
            snapshot = PolicyInput(time, apply_time, tuple(available), tuple(result.applied_actions))
            raw_action = float(policy(snapshot))
            if not math.isfinite(raw_action):
                raise ValueError("Policy returned a nonfinite action")
            command = raw_action
            if config.action_limit is not None:
                command = max(-config.action_limit, min(config.action_limit, command))
            latest = snapshot.latest
            job = {
                "id": job_id, "start_time": time, "complete_time": complete_time,
                "apply_time": apply_time, "observation_index": latest.index,
                "capture_time": latest.capture_time, "available_time": latest.available_time,
                "observation_indices": [obs.index for obs in snapshot.observations],
                "skipped_latest_indices": list(range(last_used_index + 1, latest.index)),
                "raw_action": raw_action, "action": command, "status": "pending",
            }
            result.jobs.append(job)
            result.events.append({"time": time, "kind": "dispatch", "job_id": job_id,
                                  "observation_index": latest.index})
            last_used_index = latest.index
            if config.schedule == "serial":
                busy = job_id
            push(when + compute_ticks, "complete", job_id)

        elif kind == "complete":
            job = result.jobs[payload]
            job["status"] = "completed"
            result.events.append({"time": time, "kind": "complete", "job_id": payload})
            if config.schedule == "serial":
                busy = None
            push(when + actuator_ticks, "apply", payload)
            if config.schedule == "serial":
                push(when, "dispatch")

        elif kind == "apply":
            job = result.jobs[payload]
            action = job["action"]
            job["status"] = "applied"
            result.applied_actions.append(AppliedAction(time, action, payload))
            result.events.append({"time": time, "kind": "apply", "job_id": payload,
                                  "value": action, "observation_age": time - job["capture_time"]})

        elif kind == "sample":
            result.samples.append({"time": time, "position": float(state[0]), "velocity": float(state[1]),
                                   "reference": reference_value, "disturbance": disturbance_value,
                                   "action": action})
    return result
