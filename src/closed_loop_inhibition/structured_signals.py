"""Precomputed forcing on a clock independent of sensing and computation.

OU samples follow the exact stationary discrete transition. Between samples the
simulator holds the forcing constant; this is a sampled OU process, not an exact
continuous stochastic differential equation integrated through the plant.
"""

from bisect import bisect_right
from dataclasses import dataclass, replace
import math
from typing import Callable, Protocol

import numpy as np

from .records import PolicyInput
from .signals import PiecewiseConstant

_NS = 1_000_000_000


def _ticks(value: float, name: str, *, positive: bool = False) -> int:
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    ticks = int(round(value * _NS))
    if abs(value - ticks / _NS) > 1e-12 or (positive and ticks < 1):
        raise ValueError(f"{name} must align with the one-nanosecond simulation clock")
    return ticks


def _clock(duration: float, dt: float) -> tuple[float, ...]:
    end = _ticks(duration, "duration", positive=True)
    step = _ticks(dt, "dt", positive=True)
    return tuple(tick / _NS for tick in range(0, end + 1, step))


@dataclass(frozen=True)
class SampledSignal:
    """Immutable samples, right-continuous and held after the final sample.

    The first sample is at zero. A duration not divisible by the sample interval
    includes the final preceding sample and holds it through the requested end.
    Tuples prevent accidental mutation when a tape is reused across schedules.
    """

    times: tuple[float, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        times = tuple(float(time) for time in self.times)
        values = tuple(float(value) for value in self.values)
        if not times or len(times) != len(values):
            raise ValueError("Signal times and values must have the same nonzero length")
        ticks = tuple(_ticks(time, "sample time") for time in times)
        if ticks[0] != 0 or any(b <= a for a, b in zip(ticks, ticks[1:])):
            raise ValueError("Sample times must start at zero and increase strictly")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Signal values must be finite")
        object.__setattr__(self, "times", tuple(tick / _NS for tick in ticks))
        object.__setattr__(self, "values", values)

    def at(self, time: float) -> float:
        time = float(time)
        if not math.isfinite(time) or time < 0:
            raise ValueError("Query time must be finite and nonnegative")
        return self.values[bisect_right(self.times, time) - 1]

    def to_piecewise_constant(self) -> PiecewiseConstant:
        return PiecewiseConstant(
            initial=self.values[0],
            changes=tuple(zip(self.times[1:], self.values[1:])),
        )


def ou_tape(
    *, duration: float, dt: float, correlation_time: float, std: float, seed: int
) -> SampledSignal:
    """Draw a stationary mean-zero OU tape without per-trace normalization.

    ``std`` is the marginal standard deviation in expectation. Individual tapes
    retain their sampled mean and RMS, including fluctuations on long timescales.
    The random stream and forcing clock do not depend on controller dispatches.
    """
    times = _clock(duration, dt)
    if not math.isfinite(correlation_time) or correlation_time <= 0:
        raise ValueError("correlation_time must be finite and positive")
    if not math.isfinite(std) or std < 0:
        raise ValueError("std must be finite and nonnegative")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    decay = math.exp(-dt / correlation_time)
    innovation_std = std * math.sqrt(-math.expm1(-2 * dt / correlation_time))
    normal = np.random.default_rng(seed).standard_normal(len(times))
    values = np.empty(len(times), dtype=float)
    values[0] = std * normal[0]
    for index in range(1, len(values)):
        values[index] = decay * values[index - 1] + innovation_std * normal[index]
    return SampledSignal(times, tuple(values))


def sinusoid_tape(
    *, duration: float, dt: float, frequency: float, amplitude: float, phase: float = 0.0
) -> SampledSignal:
    """Sample a sinusoid in Hz, subsequently held at the independent tape clock."""
    times = _clock(duration, dt)
    if not math.isfinite(frequency) or frequency < 0:
        raise ValueError("frequency must be finite and nonnegative")
    if not math.isfinite(amplitude) or not math.isfinite(phase):
        raise ValueError("amplitude and phase must be finite")
    values = amplitude * np.sin(2 * np.pi * frequency * np.asarray(times) + phase)
    return SampledSignal(times, tuple(values))


def rectangular_pulse(*, onset: float, duration: float, amplitude: float) -> PiecewiseConstant:
    """A force pulse whose start and end align with the simulator clock."""
    start = _ticks(onset, "onset")
    length = _ticks(duration, "duration", positive=True)
    if not math.isfinite(amplitude):
        raise ValueError("amplitude must be finite")
    return PiecewiseConstant(changes=((start / _NS, amplitude), ((start + length) / _NS, 0.0)))


class _TimeSignal(Protocol):
    def at(self, time: float) -> float: ...


@dataclass(frozen=True)
class MeasurementNoisePolicy:
    """Add position measurement noise at capture time without changing physics.

    Each policy call gets a new immutable snapshot. Revisited historical
    observations receive the same noise, regardless of dispatch time or cadence.
    Velocity remains a separate, exact channel in this benchmark; it is not a
    numerical derivative of noisy position. The wrapper has no evolving state.
    """

    policy: Callable[[PolicyInput], float]
    signal: _TimeSignal

    def __call__(self, snapshot: PolicyInput) -> float:
        observations = []
        for observation in snapshot.observations:
            noise = float(self.signal.at(observation.capture_time))
            if not math.isfinite(noise):
                raise ValueError("Position measurement noise must be finite")
            position = observation.position + noise
            if not math.isfinite(position):
                raise ValueError("Noisy position must be finite")
            observations.append(replace(observation, position=position))
        return self.policy(replace(snapshot, observations=tuple(observations)))
