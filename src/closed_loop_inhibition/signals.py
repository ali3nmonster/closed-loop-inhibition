"""Time-indexed forcing independent of controller scheduling."""

from bisect import bisect_right
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PiecewiseConstant:
    initial: float = 0.0
    changes: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        changes = tuple((float(t), float(value)) for t, value in self.changes)
        object.__setattr__(self, "changes", changes)
        if not math.isfinite(self.initial):
            raise ValueError("Signal initial value must be finite")
        previous = -1.0
        for time, value in changes:
            if not math.isfinite(time) or not math.isfinite(value):
                raise ValueError("Signal times and values must be finite")
            if time < 0 or time <= previous:
                raise ValueError("Signal changes must have strictly increasing nonnegative times")
            previous = time

    def at(self, time: float) -> float:
        """Right-continuous value, including a change at exactly ``time``."""
        index = bisect_right([entry[0] for entry in self.changes], time)
        return float(self.initial if index == 0 else self.changes[index - 1][1])
