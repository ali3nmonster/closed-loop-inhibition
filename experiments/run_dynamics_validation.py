"""Validate independently tunable plant, feedback, and perturbation timescales.

This runner uses analytical and recurrence references before any new transformer
experiment. Unstable linear conditions are expected outcomes, not discarded runs.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from closed_loop_inhibition.controllers import ClassicalPDController
from closed_loop_inhibition.dynamics_validation import (
    delayed_pd_reference, free_ringdown, ou_open_loop_covariance,
    plant_timescales, sampled_frequency_response, sampled_mode_summary,
)
from closed_loop_inhibition.plants import Oscillator
from closed_loop_inhibition.structured_signals import (
    MeasurementNoisePolicy, ou_tape, sinusoid_tape,
)
from closed_loop_inhibition.timing import TimingConfig, simulate

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/DYNAMICS_VALIDATION.md"


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite output cannot be silently serialized")
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: json.dumps(json_value(value)) if isinstance(value, (dict, list, tuple))
                         else value for key, value in row.items()} for row in rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(root=ROOT):
    return {str(path.relative_to(root)): sha(path)
            for directory in ("src", "experiments")
            for path in sorted((root / directory).rglob("*.py"))}


def freeze_manifest(config_path, protocol=PROTOCOL, root=ROOT):
    return {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": sha(config_path),
        "protocol_path": str(Path(protocol).resolve()),
        "protocol_sha256": sha(protocol),
        "source_sha256": source_hashes(root),
    }


def verify_manifest(manifest, *, root=ROOT):
    if sha(manifest["config_path"]) != manifest["config_sha256"]:
        raise ValueError("Validation config changed during the run")
    if sha(manifest["protocol_path"]) != manifest["protocol_sha256"]:
        raise ValueError("Validation protocol changed during the run")
    if source_hashes(root) != manifest["source_sha256"]:
        raise ValueError("Validation source files changed during the run")


def normalized_state_error(actual, expected, tau):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or actual.ndim != 2 or actual.shape[1] != 2:
        raise ValueError("State arrays must have matching (N, 2) shapes")
    if not len(actual) or not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        raise ValueError("State arrays must be finite and nonempty")
    scale = np.array([1.0, tau])
    denominator = max(float(np.max(np.linalg.norm(expected * scale, axis=1))), np.finfo(float).tiny)
    return float(np.max(np.linalg.norm((actual - expected) * scale, axis=1)) / denominator)


def empirical_growth(times, states, tau):
    times, states = np.asarray(times), np.asarray(states)
    norms = np.linalg.norm(states * np.array([1.0, tau]), axis=1)
    valid = np.isfinite(norms) & (norms > 1e-250)
    indices = np.flatnonzero(valid)
    if len(indices) < 6:
        return None
    indices = indices[len(indices) // 2:]
    return float(np.polyfit(times[indices], np.log(norms[indices]), 1)[0])


def fit_phasor(times, positions, omega, amplitude):
    """For sine input: q=a*cos(wt)+b*sin(wt)+c implies H=(b+i*a)/A."""
    times, positions = np.asarray(times), np.asarray(positions)
    design = np.column_stack((np.cos(omega * times), np.sin(omega * times), np.ones(len(times))))
    coefficients, _, rank, _ = np.linalg.lstsq(design, positions, rcond=None)
    if rank != 3 or amplitude == 0:
        raise ValueError("Sinusoidal response fit is not identifiable")
    phasor = complex(coefficients[1], coefficients[0]) / amplitude
    residual = float(np.sqrt(np.mean(np.square(positions - design @ coefficients))))
    return phasor, residual


def signal_statistics(values, lag, dt):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) <= lag or lag < 1 or not np.all(np.isfinite(values)):
        raise ValueError("Signal statistics need a finite vector longer than the positive lag")
    std = float(np.std(values))
    return {
        "samples": len(values), "sample_dt": dt,
        "mean": float(np.mean(values)), "rms": float(np.sqrt(np.mean(values**2))), "std": std,
        "lag_steps": lag, "lag_seconds": lag * dt,
        "lag_correlation": float(np.corrcoef(values[:-lag], values[lag:])[0, 1]) if std > 0 else None,
    }


def sample_arrays(run):
    return (np.array([row["time"] for row in run.samples]),
            np.array([[row["position"], row["velocity"]] for row in run.samples]))


def finite_window_metrics(run, burn_in, lag_seconds):
    """Sampled-position integrals and exact held-action integrals on a full window."""
    times, states = sample_arrays(run)
    duration = run.config.duration
    completed = run.failure_reason is None and len(times) > 1 and abs(times[-1] - duration) < 1e-9
    result = {
        "completed": completed, "failure_reason": run.failure_reason,
        "observed_duration": float(times[-1]) if len(times) else 0.0,
        "maximum_abs_state": float(np.max(np.abs(states))) if len(states) else None,
        "maximum_abs_position": float(np.max(np.abs(states[:, 0]))) if len(states) else None,
        "maximum_abs_command": max((abs(job["raw_action"]) for job in run.jobs), default=0.0),
        "maximum_abs_applied_action": max((abs(event.value) for event in run.applied_actions), default=0.0),
        "saturation_enabled": run.config.action_limit is not None,
        "window_start": burn_in, "window_end": duration,
        "mean_position": None, "rms_position": None, "std_position": None,
        "mean_action": None, "rms_action": None, "position_lag_correlation": None,
    }
    if not completed:
        return result
    if not 0 <= burn_in < duration:
        raise ValueError("Burn-in must precede the complete rollout endpoint")
    mask = times >= burn_in - 1e-10
    window_times, positions = times[mask], states[mask, 0]
    if abs(window_times[0] - burn_in) > 1e-9:
        raise ValueError("Burn-in must align with the reporting grid")
    length = duration - burn_in
    mean = float(np.trapezoid(positions, window_times) / length)
    square = float(np.trapezoid(positions**2, window_times) / length)
    result.update(mean_position=mean, rms_position=math.sqrt(max(0, square)),
                  std_position=math.sqrt(max(0, square - mean**2)))
    total, total_square = 0.0, 0.0
    for index, event in enumerate(run.applied_actions):
        start = max(burn_in, event.time)
        end = min(duration, run.applied_actions[index + 1].time
                  if index + 1 < len(run.applied_actions) else duration)
        if end > start:
            total += (end - start) * event.value
            total_square += (end - start) * event.value**2
    result["mean_action"] = total / length
    result["rms_action"] = math.sqrt(max(0, total_square / length))
    lag = max(1, int(round(lag_seconds / run.config.sample_interval)))
    result["position_lag_seconds"] = lag * run.config.sample_interval
    if len(positions) > lag and np.std(positions[:-lag]) > 0 and np.std(positions[lag:]) > 0:
        result["position_lag_correlation"] = float(np.corrcoef(positions[:-lag], positions[lag:])[0, 1])
    return result


def timing(config, *, duration, delay_steps=0, sample_interval=None):
    period = config["controller"]["period"]
    return TimingConfig(
        duration=duration, observation_interval=period, decision_interval=period,
        compute_duration=round(delay_steps * period, 9), sensor_delay=0.0, actuator_delay=0.0,
        schedule="fixed_cadence", history_seconds=period,
        sample_interval=sample_interval if sample_interval is not None else period,
        action_limit=None, max_abs_state=config["modes"]["state_guard"],
    )


def pd_policy(config, tau, *, open_loop=False):
    if open_loop:
        return lambda snapshot: 0.0
    return ClassicalPDController(kp=config["controller"]["kp"], kd=config["controller"]["kd"], tau=tau)


def add_check(checks, name, passed, **details):
    checks.append({"name": name, "passed": bool(passed), **details})


def save_checks(output, checks, *, complete=False):
    write_json(output / "checks.json", {"complete": complete, "all_passed": all(row["passed"] for row in checks),
                                        "passed": sum(row["passed"] for row in checks),
                                        "total": len(checks), "checks": checks})


def ringdown_stage(config, output, artifacts, checks):
    rows = []
    settings = config["ringdown"]
    for tau in config["plant"]["taus"]:
        plant = Oscillator(tau=tau, zeta=config["plant"]["zeta"])
        expected_scales = plant_timescales(plant)
        duration = round(settings["envelope_times"] * expected_scales["envelope_decay_time"], 9)
        step = round(tau / settings["samples_per_tau"], 9)
        initial = (settings["initial_position"], 0.0)
        run = simulate(plant, pd_policy(config, tau, open_loop=True),
                       timing(config, duration=duration, sample_interval=step), initial_state=initial)
        times, states = sample_arrays(run)
        reference = free_ringdown(plant, times, initial)
        error = normalized_state_error(states, reference, tau)
        peaks = find_peaks(states[:, 0])[0]
        peaks = peaks[states[peaks, 0] > 0]
        period = float(np.mean(np.diff(times[peaks]))) if len(peaks) >= 2 else None
        decay = float(-1 / np.polyfit(times[peaks], np.log(states[peaks, 0]), 1)[0]) if len(peaks) >= 2 else None
        period_error = abs(period / expected_scales["damped_period"] - 1) if period else None
        decay_error = abs(decay / expected_scales["envelope_decay_time"] - 1) if decay else None
        row = {**expected_scales, "duration": duration, "reporting_dt": step,
               "normalized_state_error": error, "positive_peak_count": len(peaks),
               "measured_period": period, "measured_envelope_time": decay,
               "period_relative_error": period_error, "envelope_relative_error": decay_error,
               "completed": run.failure_reason is None}
        rows.append(row)
        tolerances = config["tolerances"]
        add_check(checks, f"ringdown_state_tau_{tau}", error <= tolerances["relative_trajectory_error"] and row["completed"],
                  value=error, tolerance=tolerances["relative_trajectory_error"])
        add_check(checks, f"ringdown_period_tau_{tau}", period_error is not None and period_error <= tolerances["relative_ringdown_period_error"],
                  value=period_error, tolerance=tolerances["relative_ringdown_period_error"])
        add_check(checks, f"ringdown_decay_tau_{tau}", decay_error is not None and decay_error <= tolerances["relative_ringdown_decay_error"],
                  value=decay_error, tolerance=tolerances["relative_ringdown_decay_error"])
        np.savez_compressed(artifacts / f"ringdown_tau_{tau}.npz", times=times, states=states, reference=reference)
        write_json(output / "ringdown.json", rows)
        save_checks(output, checks)
    write_csv(output / "ringdown.csv", rows)
    return rows


def modes_stage(config, output, artifacts, checks):
    rows = []
    controller, settings = config["controller"], config["modes"]
    period = controller["period"]
    for tau in config["plant"]["taus"]:
        plant = Oscillator(tau=tau, zeta=config["plant"]["zeta"])
        for delay in settings["delay_steps"]:
            mode = sampled_mode_summary(plant, controller["kp"], controller["kd"], period, delay)
            initial = (settings["initial_position"], 0.0)
            run = simulate(plant, pd_policy(config, tau),
                           timing(config, duration=settings["duration"], delay_steps=delay), initial_state=initial)
            times, states = sample_arrays(run)
            reference = delayed_pd_reference(plant, initial,
                np.zeros(int(round(settings["duration"] / period))), fine_dt=period, period=period,
                delay_steps=delay, kp=controller["kp"], kd=controller["kd"])
            indices = np.rint(times / period).astype(int)
            expected = reference["states"][indices]
            error = normalized_state_error(states, expected, tau)
            measured = empirical_growth(times, states, tau)
            sign_applicable = mode["growth_rate"] is not None and abs(mode["growth_rate"]) > config["tolerances"]["growth_sign_minimum_rate"]
            sign_matches = measured is not None and np.sign(measured) == np.sign(mode["growth_rate"])
            row = {"tau": tau, "zeta": plant.zeta, "delay_steps": delay, "delay": delay * period,
                   "update_over_tau": period / tau, "delay_over_tau": delay * period / tau,
                   **mode, "measured_growth_rate": measured, "growth_sign_check_applicable": sign_applicable,
                   "growth_sign_matches": bool(sign_matches) if sign_applicable else None,
                   "normalized_state_error": error, "completed": run.failure_reason is None,
                   "failure_reason": run.failure_reason, "observed_duration": float(times[-1]),
                   "maximum_abs_state": float(np.max(np.abs(states))),
                   "maximum_abs_command": max((abs(job["raw_action"]) for job in run.jobs), default=0.0),
                   "maximum_abs_applied_action": max((abs(event.value) for event in run.applied_actions), default=0.0),
                   "saturation_enabled": False}
            rows.append(row)
            add_check(checks, f"delayed_map_tau_{tau}_m_{delay}", error <= config["tolerances"]["relative_trajectory_error"],
                      value=error, tolerance=config["tolerances"]["relative_trajectory_error"], censored=not row["completed"])
            if sign_applicable:
                add_check(checks, f"growth_sign_tau_{tau}_m_{delay}", sign_matches,
                          measured=measured, predicted=mode["growth_rate"], censored=not row["completed"])
            np.savez_compressed(artifacts / f"mode_tau_{tau}_m_{delay}.npz",
                                times=times, states=states, reference=expected)
            write_json(output / "modes.json", rows)
            save_checks(output, checks)
    write_csv(output / "modes.csv", rows)
    add_check(checks, "mode_grid_contains_stable_and_unstable", any(row["stable"] for row in rows)
              and any(row["classification"] == "unstable" for row in rows))
    crossings = [tau for tau in config["plant"]["taus"]
                 if any(row["stable"] for row in rows if row["tau"] == tau)
                 and any(row["classification"] == "unstable" for row in rows if row["tau"] == tau)]
    add_check(checks, "mode_grid_delay_driven_stability_crossing", bool(crossings), taus=crossings)
    save_checks(output, checks)
    return rows


def transfer_stage(config, output, artifacts, checks):
    rows = []
    settings, controller = config["transfer"], config["controller"]
    period, fine_dt = controller["period"], config["noise"]["dt"]
    conditions = [{"tau": tau, "delay_steps": 0, "controller": "open_loop"}
                  for tau in config["plant"]["taus"]]
    conditions += [{**anchor, "controller": "pd"} for anchor in config["anchors"]]
    for condition in conditions:
        tau, delay = condition["tau"], condition["delay_steps"]
        open_loop = condition["controller"] == "open_loop"
        plant = Oscillator(tau=tau, zeta=config["plant"]["zeta"])
        kp, kd = (0.0, 0.0) if open_loop else (controller["kp"], controller["kd"])
        mode = sampled_mode_summary(plant, kp, kd, period, delay)
        if not mode["stable"]:
            rows.append({**condition, "completed": False, "steady_response_available": False,
                         "reason": "Analytically unstable anchor; no attracting steady response", **mode})
            add_check(checks, f"transfer_anchor_stable_{condition['controller']}_tau_{tau}_m_{delay}", False,
                      spectral_radius=mode["spectral_radius"])
            write_json(output / "transfer.json", rows)
            save_checks(output, checks)
            continue
        envelope = plant.tau / plant.zeta if open_loop else mode["envelope_time"]
        warmup = round(math.ceil(settings["warmup_envelopes"] * envelope / period) * period, 9)
        for scaled_frequency in settings["scaled_frequencies"]:
            omega = scaled_frequency / tau
            duration = round(math.ceil((warmup + settings["measurement_periods"] * 2 * np.pi / omega) / period) * period, 9)
            tape = sinusoid_tape(duration=duration, dt=fine_dt, frequency=omega / (2 * np.pi),
                                 amplitude=settings["amplitude"])
            run = simulate(plant, pd_policy(config, tau, open_loop=open_loop),
                           timing(config, duration=duration, delay_steps=delay),
                           disturbance=tape.to_piecewise_constant())
            times, states = sample_arrays(run)
            selected = times >= warmup - 1e-10
            expected = sampled_frequency_response(plant, kp, kd, period, delay, [omega], force_dt=fine_dt)[0]
            completed = run.failure_reason is None and abs(times[-1] - duration) < 1e-9
            measured, residual, error = None, None, None
            reason = run.failure_reason
            if completed and np.count_nonzero(selected) >= 4:
                try:
                    measured, residual = fit_phasor(times[selected], states[selected, 0], omega, settings["amplitude"])
                    error = float(abs(measured - expected) / abs(expected))
                except ValueError as exc:
                    reason = str(exc)
            elif completed:
                reason = "Too few post-warmup samples for a response fit"
            row = {**condition, "omega_times_tau": scaled_frequency, "angular_frequency": omega,
                   "frequency_hz": omega / (2 * np.pi), "warmup": warmup, "duration": duration,
                   "force_dt": fine_dt, "measured_real": measured.real if measured is not None else None,
                   "measured_imag": measured.imag if measured is not None else None,
                   "predicted_real": float(expected.real), "predicted_imag": float(expected.imag),
                   "measured_gain": abs(measured) if measured is not None else None, "predicted_gain": float(abs(expected)),
                   "measured_phase_radians": float(np.angle(measured)) if measured is not None else None,
                   "predicted_phase_radians": float(np.angle(expected)),
                   "complex_relative_error": error, "fit_residual_rms": residual,
                   "completed": completed, "failure_reason": reason,
                   "steady_response_available": measured is not None}
            rows.append(row)
            add_check(checks, f"transfer_{condition['controller']}_tau_{tau}_m_{delay}_w_{scaled_frequency}",
                      error is not None and error <= config["tolerances"]["relative_complex_transfer_error"] and row["completed"],
                      value=error, tolerance=config["tolerances"]["relative_complex_transfer_error"])
            np.savez_compressed(artifacts / f"transfer_{condition['controller']}_tau_{tau}_m_{delay}_w_{scaled_frequency}.npz",
                                times=times, positions=states[:, 0])
            write_json(output / "transfer.json", rows)
            save_checks(output, checks)
    write_csv(output / "transfer.csv", rows)
    return rows


def noise_quality_stage(config, output, artifacts, checks):
    rows, pooled = [], []
    settings = config["noise"]
    for correlation_time in settings["correlation_times"]:
        lag = max(1, int(round(correlation_time / settings["dt"])))
        arrays = []
        for seed in settings["seeds"]:
            tape = ou_tape(duration=settings["quality_duration"], dt=settings["dt"],
                           correlation_time=correlation_time, std=settings["std"], seed=seed)
            values = np.asarray(tape.values)
            arrays.append(values)
            rows.append({"correlation_time": correlation_time, "seed": seed,
                         **signal_statistics(values, lag, settings["dt"])})
        combined = np.concatenate(arrays)
        left = np.concatenate([values[:-lag] for values in arrays])
        right = np.concatenate([values[lag:] for values in arrays])
        rms = float(np.sqrt(np.mean(combined**2)))
        correlation = float(np.corrcoef(left, right)[0, 1])
        expected_correlation = math.exp(-lag * settings["dt"] / correlation_time)
        row = {"correlation_time": correlation_time, "seeds": settings["seeds"],
               "mean": float(np.mean(combined)), "rms": rms, "std": float(np.std(combined)),
               "expected_std": settings["std"], "lag_steps": lag, "lag_seconds": lag * settings["dt"],
               "lag_correlation": correlation, "expected_lag_correlation": expected_correlation,
               "rms_relative_error": abs(rms / settings["std"] - 1),
               "correlation_absolute_error": abs(correlation - expected_correlation)}
        pooled.append(row)
        add_check(checks, f"ou_rms_tc_{correlation_time}", row["rms_relative_error"] <= config["tolerances"]["relative_noise_rms_error"],
                  value=row["rms_relative_error"], tolerance=config["tolerances"]["relative_noise_rms_error"])
        add_check(checks, f"ou_correlation_tc_{correlation_time}", row["correlation_absolute_error"] <= config["tolerances"]["absolute_noise_correlation_error"],
                  value=row["correlation_absolute_error"], tolerance=config["tolerances"]["absolute_noise_correlation_error"])
        write_json(output / "noise_quality.json", {"individual": rows, "pooled": pooled})
        save_checks(output, checks)
    write_csv(output / "noise_quality_individual.csv", rows)
    write_csv(output / "noise_quality_pooled.csv", pooled)
    return pooled


def noise_response_stage(config, output, artifacts, checks):
    settings, controller = config["noise"], config["controller"]
    duration, fine_dt, period = settings["duration"], settings["dt"], controller["period"]
    rows = []
    conditions = [{"tau": tau, "delay_steps": 0, "controller": "open_loop", "channel": "force"}
                  for tau in config["plant"]["taus"]]
    conditions += [{**anchor, "controller": "pd", "channel": channel}
                   for anchor in config["anchors"] for channel in settings["channels"]]
    for correlation_time in settings["correlation_times"]:
        for seed in settings["seeds"]:
            tape = ou_tape(duration=duration, dt=fine_dt, correlation_time=correlation_time,
                           std=settings["std"], seed=seed)
            values, times = np.asarray(tape.values), np.asarray(tape.times)
            np.savez_compressed(artifacts / f"noise_tc_{correlation_time}_seed_{seed}.npz", times=times, values=values)
            selected = times >= settings["burn_in"] - 1e-10
            stats = signal_statistics(values[selected], max(1, int(round(correlation_time / fine_dt))), fine_dt)
            capture_times = np.array([round(index * period, 9)
                                      for index in range(int(round(duration / period)) + 1)])
            capture_values = np.array([tape.at(t) for t in capture_times])
            capture_selected = capture_times >= settings["burn_in"] - 1e-10
            capture_lag = max(1, int(round(correlation_time / period)))
            capture_stats = signal_statistics(capture_values[capture_selected], capture_lag, period)
            for condition in conditions:
                tau, delay = condition["tau"], condition["delay_steps"]
                open_loop = condition["controller"] == "open_loop"
                plant = Oscillator(tau=tau, zeta=config["plant"]["zeta"])
                policy = pd_policy(config, tau, open_loop=open_loop)
                if condition["channel"] == "position_noise":
                    policy = MeasurementNoisePolicy(policy, tape)
                run = simulate(plant, policy,
                    timing(config, duration=duration, delay_steps=delay, sample_interval=settings["sample_interval"]),
                    disturbance=tape.to_piecewise_constant() if condition["channel"] == "force" else None)
                metrics = finite_window_metrics(run, settings["burn_in"], correlation_time)
                row = {**condition, "correlation_time": correlation_time, "seed": seed,
                       **metrics, **{f"input_{key}": value for key, value in stats.items()},
                       "input_clock": "independent_fine_tape", "input_channel": condition["channel"],
                       "capture_reference_error": None, "stationary_open_loop_rms_prediction": None}
                if condition["channel"] == "position_noise":
                    row.update({f"capture_input_{key}": value for key, value in capture_stats.items()})
                    row["capture_input_expected_lag_correlation"] = math.exp(-capture_lag * period / correlation_time)
                name = f"{condition['controller']}_tau_{tau}_m_{delay}_{condition['channel']}_tc_{correlation_time}_seed_{seed}"
                add_check(checks, f"noise_complete_{name}", metrics["completed"], failure_reason=run.failure_reason)
                if open_loop:
                    covariance = ou_open_loop_covariance(plant, fine_dt, correlation_time, settings["std"])
                    row["stationary_open_loop_rms_prediction"] = float(np.sqrt(covariance[0, 0]))
                    row["stationary_covariance"] = covariance.tolist()
                if seed == settings["seeds"][0]:
                    sample_times, states = sample_arrays(run)
                    np.savez_compressed(artifacts / f"noise_response_{name}.npz", times=sample_times, states=states,
                                        actions=np.array([sample["action"] for sample in run.samples]))
                    if not open_loop:
                        n_steps = int(round(duration / fine_dt))
                        reference = delayed_pd_reference(plant, (0, 0),
                            values[:n_steps] if condition["channel"] == "force" else np.zeros(n_steps),
                            fine_dt=fine_dt, period=period, delay_steps=delay,
                            kp=controller["kp"], kd=controller["kd"],
                            position_noise=capture_values if condition["channel"] == "position_noise" else None)
                        capture_mask = np.isclose(sample_times / period, np.rint(sample_times / period), atol=1e-8, rtol=0)
                        indices = np.rint(sample_times[capture_mask] / fine_dt).astype(int)
                        error = normalized_state_error(states[capture_mask], reference["states"][indices], tau)
                        row["capture_reference_error"] = error
                        add_check(checks, f"noise_recurrence_{name}", error <= config["tolerances"]["relative_trajectory_error"],
                                  value=error, tolerance=config["tolerances"]["relative_trajectory_error"])
                rows.append(row)
                write_json(output / "noise_response.json", rows)
                save_checks(output, checks)
            print(f"noise response: tc={correlation_time}, seed={seed}, {len(rows)}/{len(conditions) * len(settings['seeds']) * len(settings['correlation_times'])} rollouts", flush=True)
    write_csv(output / "noise_response.csv", rows)
    return rows


def make_plots(config, output, artifacts, ringdown, modes, transfer, noise):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    ax = axes[0, 0]
    for row in ringdown:
        with np.load(artifacts / f"ringdown_tau_{row['tau']}.npz") as data:
            ax.plot(data["times"] / row["tau"], data["states"][:, 0] / config["ringdown"]["initial_position"],
                    label=f"τ={row['tau']:g}s", alpha=0.8)
    ax.set(xlabel="Time / τ", ylabel="Position / initial position", title="Passive dynamics collapse under time scaling")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    taus, delays = config["plant"]["taus"], config["modes"]["delay_steps"]
    matrix = np.array([[next(row["spectral_radius"] for row in modes
                            if row["tau"] == tau and row["delay_steps"] == delay) for delay in delays] for tau in taus])
    extent = max(float(np.max(abs(matrix - 1))), 1e-6)
    rendered = ax.imshow(matrix, cmap="coolwarm", vmin=1 - extent, vmax=1 + extent, aspect="auto")
    for i in range(len(taus)):
        for j in range(len(delays)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=9)
    ax.set(xticks=range(len(delays)), xticklabels=[f"{delay * config['controller']['period'] * 1000:g}" for delay in delays],
           yticks=range(len(taus)), yticklabels=[f"τ={tau:g}, h/τ={config['controller']['period'] / tau:g}" for tau in taus],
           xlabel="Computation delay (ms)", title="PD spectral radius: stable below 1")
    fig.colorbar(rendered, ax=ax, label="Spectral radius")
    ax = axes[1, 0]
    for controller, tau, delay in dict.fromkeys((row["controller"], row["tau"], row["delay_steps"]) for row in transfer):
        selected = [row for row in transfer if row.get("measured_gain") is not None and (row["controller"], row["tau"], row["delay_steps"]) == (controller, tau, delay)]
        if not selected:
            continue
        x = [row["omega_times_tau"] for row in selected]
        line, = ax.plot(x, [row["predicted_gain"] for row in selected], "-", alpha=0.7,
                        label=f"{controller}, τ={tau:g}, m={delay}")
        ax.scatter(x, [row["measured_gain"] for row in selected], color=line.get_color(), s=25, marker="x")
    ax.set(xlabel="Angular frequency × τ", ylabel="Position / force gain",
           title="Held-input transfer: predicted lines, measured ×", yscale="log")
    ax.legend(fontsize=6, ncol=2)
    ax = axes[1, 1]
    for channel, marker in (("force", "o"), ("position_noise", "s")):
        for anchor in config["anchors"]:
            selected = [row for row in noise if row["controller"] == "pd" and row["channel"] == channel
                        and row["tau"] == anchor["tau"] and row["delay_steps"] == anchor["delay_steps"]]
            groups = [[row["rms_position"] for row in selected if row["correlation_time"] == tc
                       and row["rms_position"] is not None] for tc in config["noise"]["correlation_times"]]
            means = np.array([np.mean(group) if group else np.nan for group in groups])
            lower = np.array([np.min(group) if group else np.nan for group in groups])
            upper = np.array([np.max(group) if group else np.nan for group in groups])
            ax.errorbar(config["noise"]["correlation_times"], means, yerr=np.array([means - lower, upper - means]),
                        marker=marker, capsize=2, label=f"{channel}, τ={anchor['tau']:g}, m={anchor['delay_steps']}")
    ax.set(xscale="log", yscale="log", xlabel="Input correlation time (s)", ylabel="Position RMS",
           title="PD noise response: mean and range of 3 seeds")
    ax.legend(fontsize=6)
    fig.savefig(output / "dynamics_summary.png", dpi=160)
    fig.savefig(output / "dynamics_summary.svg")
    svg = output / "dynamics_summary.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/dynamics_validation.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/dynamics_validation")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "runs/dynamics_validation")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for directory in (args.output, args.artifacts):
        if directory.exists():
            raise ValueError(f"Fresh output and artifact directories are required: {directory}")
    manifest = freeze_manifest(args.config)
    args.output.mkdir(parents=True)
    args.artifacts.mkdir(parents=True)
    write_json(args.output / "config.json", config)
    write_json(args.output / "start_manifest.json", manifest)
    checks, results = [], {}
    start = time.monotonic()
    for name, function in (("ringdown", ringdown_stage), ("modes", modes_stage),
                           ("transfer", transfer_stage), ("noise_quality", noise_quality_stage),
                           ("noise_response", noise_response_stage)):
        print(f"Starting {name}", flush=True)
        results[name] = function(config, args.output, args.artifacts, checks)
        verify_manifest(manifest)
        print(f"Finished {name}: {len(checks)} checks, {sum(row['passed'] for row in checks)} passing; elapsed {time.monotonic() - start:.1f}s", flush=True)
    make_plots(config, args.output, args.artifacts, results["ringdown"], results["modes"],
               results["transfer"], results["noise_response"])
    verify_manifest(manifest)
    save_checks(args.output, checks, complete=True)
    rollout_counts = {name: len(results[name]) for name in ("ringdown", "modes", "noise_response")}
    rollout_counts["transfer"] = sum("duration" in row for row in results["transfer"])
    manifest.update(completed_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - start,
                    output_sha256={path.name: sha(path) for path in sorted(args.output.iterdir()) if path.is_file()},
                    artifact_sha256={path.name: sha(path) for path in sorted(args.artifacts.iterdir()) if path.is_file()},
                    all_checks_passed=all(row["passed"] for row in checks),
                    check_count=len(checks), checks_passed=sum(row["passed"] for row in checks),
                    rollout_counts=rollout_counts, rollout_count=sum(rollout_counts.values()))
    write_json(args.output / "manifest.json", manifest)
    print(f"Validation finished: {sum(row['passed'] for row in checks)}/{len(checks)} checks pass; results at {args.output}", flush=True)


if __name__ == "__main__":
    main()
