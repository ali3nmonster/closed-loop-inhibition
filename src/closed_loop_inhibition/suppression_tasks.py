"""Paired pulse/sham tasks for causal suppression experiments.

Every pair uses the same deterministic, stateless snapshot policy, including
the same active intervention. Subtracting an unmodified controller's sham
would mix disturbance response with the intervention's autonomous drift.
"""

from numbers import Integral
from typing import Callable

import numpy as np

from .imitation import encode_snapshot
from .metrics import summarize
from .plants import Oscillator, _finite_scalar
from .records import PolicyInput
from .signals import PiecewiseConstant
from .timing import SimulationResult, TimingConfig, simulate


def make_scenarios(count: int, seed: int, split: str) -> list[dict]:
    """Reproducible zero-reference pulses, independent of controller behavior.

    Onsets lie on the 0.05-second grid from 1.00 through 1.40 seconds. Pulse
    amplitude has random sign and uniform magnitude in [0.3, 0.6]. Separate
    splits must use independent declared seeds; changing an ID alone does not
    create independent exogenous scenarios.
    """
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
        raise ValueError("count must be a positive integer")
    if not isinstance(split, str) or not split:
        raise ValueError("split must be a nonempty string")
    rng = np.random.default_rng(seed)
    scenarios = []
    for index in range(count):
        onset = int(rng.integers(20, 29)) / 20
        sign = int(rng.choice((-1, 1)))
        scenarios.append({
            "id": f"{split}-{index:03d}",
            "initial_state": [0.0, 0.0],
            "reference": 0.0,
            "pulse_start": onset,
            "pulse_duration": 0.25,
            "amplitude": sign * float(rng.uniform(0.3, 0.6)),
            "duration": 5.0,
        })
    return scenarios


def _validate_scenario(scenario: dict) -> tuple[float, float, float, float]:
    onset = _finite_scalar(scenario["pulse_start"], "pulse_start")
    length = _finite_scalar(scenario["pulse_duration"], "pulse_duration")
    amplitude = _finite_scalar(scenario["amplitude"], "amplitude")
    duration = _finite_scalar(scenario["duration"], "duration")
    _finite_scalar(scenario["reference"], "reference")
    state = np.asarray(scenario["initial_state"], dtype=float)
    if state.shape != (2,) or not np.all(np.isfinite(state)):
        raise ValueError("initial_state must contain two finite values")
    if onset < 0 or length <= 0 or duration <= 0 or onset + length > duration:
        raise ValueError("Pulse must have positive duration and fit within the episode")
    if amplitude == 0:
        raise ValueError("Pulse amplitude must be nonzero for normalized response scores")
    return onset, length, amplitude, duration


def _run(
    plant: Oscillator,
    policy: Callable[[PolicyInput], float],
    scenario: dict,
    timing: TimingConfig,
    amplitude: float,
) -> SimulationResult:
    onset, length, _, duration = _validate_scenario(scenario)
    if abs(timing.duration - duration) > 1e-9:
        raise ValueError("Timing duration must match the scenario duration")
    # Keep both exogenous event timestamps in the sham, with zero amplitudes.
    # This also preserves event-time divergence checks in both arms.
    return simulate(
        plant, policy, timing,
        initial_state=tuple(scenario["initial_state"]),
        reference=PiecewiseConstant(initial=scenario["reference"]),
        disturbance=PiecewiseConstant(changes=((onset, amplitude), (onset + length, 0.0))),
    )


def paired_rollouts(
    plant: Oscillator,
    policy: Callable[[PolicyInput], float],
    scenario: dict,
    timing: TimingConfig,
) -> tuple[SimulationResult, SimulationResult]:
    """Return pulse and sham runs under the identical active policy.

    ``policy`` must be deterministic and stateless between calls, as in the
    existing history-window transformer adapter. No controller is substituted
    for the sham, and the initial state and constant reference are shared.
    Censored runs are retained so ``score_pair`` can report null scores.
    """
    _, _, amplitude, _ = _validate_scenario(scenario)
    return (
        _run(plant, policy, scenario, timing, amplitude),
        _run(plant, policy, scenario, timing, 0.0),
    )


def collect_paired_probes(
    plant: Oscillator,
    policy: Callable[[PolicyInput], float],
    scenario: dict,
    timing: TimingConfig,
    encoding: dict,
    window_seconds: float = 2.0,
) -> dict[str, np.ndarray]:
    """Encode matched, pre-call snapshots from two baseline-policy rollouts.

    Probes span dispatch times in [pulse_start, pulse_start + window_seconds],
    including endpoints. Only the controller-visible snapshot is encoded;
    the pulse specification and live simulator state never become features.
    Pairing uses the simulator's integer-nanosecond clock. Unmatched times or
    masks, an empty window, or censored rollouts raise instead of silently
    selecting a partial set of probes.
    """
    onset, _, amplitude, duration = _validate_scenario(scenario)
    window_seconds = _finite_scalar(window_seconds, "window_seconds")
    if window_seconds <= 0 or onset + window_seconds > duration + 1e-9:
        raise ValueError("Probe window must be positive and fit within the episode")
    start_tick = round(onset * 1e9)
    end_tick = round((onset + window_seconds) * 1e9)
    recorded: list[list[tuple]] = [[], []]

    for arm, perturbation in enumerate((amplitude, 0.0)):
        def record(snapshot: PolicyInput) -> float:
            tick = round(snapshot.time * 1e9)
            if start_tick <= tick <= end_tick:
                tokens, valid = encode_snapshot(snapshot, tau=plant.tau, **encoding)
                recorded[arm].append((tick, tokens, valid))
            return policy(snapshot)

        result = _run(plant, record, scenario, timing, perturbation)
        status = summarize(result)
        if status["censored"]:
            raise ValueError(f"Cannot collect paired probes from a censored run: {status['failure_reason']}")

    pulse, sham = recorded
    if not pulse or len(pulse) != len(sham):
        raise ValueError("Pulse and sham must have nonempty, equally sized probe windows")
    pulse_ticks = np.asarray([row[0] for row in pulse], dtype=np.int64)
    sham_ticks = np.asarray([row[0] for row in sham], dtype=np.int64)
    pulse_masks = np.stack([row[2] for row in pulse])
    sham_masks = np.stack([row[2] for row in sham])
    if not np.array_equal(pulse_ticks, sham_ticks):
        raise ValueError("Pulse and sham probe dispatch times differ")
    if not np.array_equal(pulse_masks, sham_masks):
        raise ValueError("Pulse and sham probe validity masks differ")
    return {
        "pulse_tokens": np.stack([row[1] for row in pulse]),
        "sham_tokens": np.stack([row[1] for row in sham]),
        "valid": pulse_masks,
        "times": pulse_ticks.astype(np.float64) / 1e9,
    }


def _window_grid(times: np.ndarray, start: float, end: float) -> np.ndarray:
    return np.concatenate(([start], times[(times > start) & (times < end)], [end]))


def _integral(values: np.ndarray, times: np.ndarray) -> float:
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapezoid(values, times))


def score_pair(
    pulse_run: SimulationResult,
    sham_run: SimulationResult,
    scenario: dict,
    recovery_seconds: float = 3.0,
) -> dict:
    """Measure normalized incremental recovery and absolute behavior together.

    ``J_response`` is the sampled trapezoidal integral of squared position
    difference divided by amplitude squared, over [pulse_end, pulse_end +
    recovery_seconds]. ``peak_response`` is the maximum absolute difference
    on that same recovery window divided by absolute amplitude. Window
    boundaries are linearly interpolated in position before squaring.

    Full-episode tracking, exact held-action effort, action variation, absolute
    error peaks and final-second tracking expose drift that pair subtraction can
    hide. If either run is censored or any requested window is incomplete,
    all comparison scores are null. Completing a run is not proof of control
    competence or asymptotic stability.
    """
    onset, length, amplitude, duration = _validate_scenario(scenario)
    recovery_seconds = _finite_scalar(recovery_seconds, "recovery_seconds")
    if recovery_seconds <= 0:
        raise ValueError("recovery_seconds must be positive")
    start, end = onset + length, onset + length + recovery_seconds
    pulse_summary, sham_summary = summarize(pulse_run), summarize(sham_run)
    score = {name: None for name in (
        "J_response", "peak_response", "pulse_tracking_rmse", "sham_tracking_rmse",
        "pulse_action_effort", "sham_action_effort", "pulse_action_variation", "sham_action_variation",
        "pulse_peak_abs_error", "sham_peak_abs_error",
        "pulse_last_second_rmse", "sham_last_second_rmse",
    )}
    score.update({
        "censored": False,
        "failure_reason": None,
        "pulse_observed_duration": pulse_summary["observed_duration"],
        "sham_observed_duration": sham_summary["observed_duration"],
        "recovery_start": start,
        "recovery_end": end,
    })
    reasons = []
    for label, result, summary in (("pulse", pulse_run, pulse_summary), ("sham", sham_run, sham_summary)):
        if summary["censored"]:
            reasons.append(f"{label}:{summary['failure_reason']}")
        elif abs(result.config.duration - duration) > 1e-9:
            reasons.append(f"{label}:scenario_duration_mismatch")
    if end > duration + 1e-9 or duration < 1.0:
        reasons.append("incomplete_scoring_window")
    if reasons:
        score.update(censored=True, failure_reason=";".join(reasons))
        return score

    pulse_times = np.array([sample["time"] for sample in pulse_run.samples])
    sham_times = np.array([sample["time"] for sample in sham_run.samples])
    pulse_positions = np.array([sample["position"] for sample in pulse_run.samples])
    sham_positions = np.array([sample["position"] for sample in sham_run.samples])
    # A shared union supports valid comparisons even if reporting grids differ.
    times = _window_grid(np.union1d(pulse_times, sham_times), start, end)
    difference = np.interp(times, pulse_times, pulse_positions) - np.interp(times, sham_times, sham_positions)
    normalized = difference / amplitude
    score["J_response"] = _integral(normalized * normalized, times)
    score["peak_response"] = float(np.max(np.abs(normalized)))

    for label, result, summary in (("pulse", pulse_run, pulse_summary), ("sham", sham_run, sham_summary)):
        for key in ("tracking_rmse", "action_effort", "action_variation", "peak_abs_error"):
            score[f"{label}_{key}"] = summary[key]
        source_times = np.array([sample["time"] for sample in result.samples])
        errors = np.array([sample["position"] - sample["reference"] for sample in result.samples])
        last_times = _window_grid(source_times, duration - 1.0, duration)
        last_errors = np.interp(last_times, source_times, errors)
        score[f"{label}_last_second_rmse"] = float(np.sqrt(_integral(last_errors * last_errors, last_times)))
    if any(not np.isfinite(value) for key, value in score.items() if key not in {"censored", "failure_reason"}):
        for key in ("J_response", "peak_response", "pulse_tracking_rmse", "sham_tracking_rmse",
                    "pulse_action_effort", "sham_action_effort", "pulse_action_variation", "sham_action_variation",
                    "pulse_peak_abs_error", "sham_peak_abs_error",
                    "pulse_last_second_rmse", "sham_last_second_rmse"):
            score[key] = None
        score.update(censored=True, failure_reason="nonfinite_pair_metric")
    return score
