"""Conditional frequency responses and own-loop collective interventions.

The frequency assay replays teacher-generated histories. Its command harmonics
are not a neural controller's closed-loop transfer function or phase margin.
Preparation uses calibration seeds only; callers persist the returned record
before confirming with disjoint held-out noise streams.
"""

from copy import deepcopy
import hashlib
import json

import numpy as np

from .centered_calibration import weighted_mean
from .collective_suppression import branches, scale_branches
from .structured_signals import sinusoid_tape
from .timescale_diagnostics import (
    _averages, _bank, _clipped, _fit, _grid, _match, _raw,
    paired_recovery, response_summary,
)
from .timescale_maps import NeuralPolicy, encode, make_plant, make_timing, metrics, run_episode
from .timescale_maps import teacher_policy
from .timing import simulate


_FREQUENCY_BANKS = {}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _config(base, protocol):
    config = deepcopy(base)
    config["diagnostics"].update(protocol["dynamics"])
    for name in ("baseline_epsilon", "min_fractional_increase", "min_positive_fraction"):
        config["diagnostics"][name] = protocol[name]
    return config


def _validate_streams(protocol):
    settings = protocol["dynamics"]
    calibration = settings["calibration_seeds"]
    confirmation = settings["confirmation_seeds"]
    if (not calibration or not confirmation or len(set(calibration)) != len(calibration)
            or len(set(confirmation)) != len(confirmation) or set(calibration) & set(confirmation)):
        raise ValueError("Calibration and confirmation require distinct, disjoint nonempty seed lists")
    common_streams = [set(protocol[name]) for name in ("discovery_seeds", "confirmation_seeds")]
    all_streams = [*common_streams, set(calibration), set(confirmation)]
    if any(all_streams[a] & all_streams[b] for a in range(len(all_streams)) for b in range(a + 1, len(all_streams))):
        raise ValueError("Common-history and dynamics seed streams must all be disjoint")


def _group(model, group):
    if group is None:
        return None
    identifiers = branches(model)
    if len(group) != len(set(group)) or any(name not in identifiers for name in group):
        raise ValueError("Group must contain distinct known branch IDs")
    return [name for name in identifiers if name in group]


def alternative_group(identifiers, selected, seed):
    """Prefer a disjoint same-size group; retain maximal difference if too large."""
    selected = list(selected)
    if len(selected) != len(set(selected)) or not set(selected) <= set(identifiers):
        raise ValueError("Selected branch group is invalid")
    if not selected or len(selected) == len(identifiers):
        return None
    rng = np.random.default_rng(seed)
    complement = [name for name in identifiers if name not in selected]
    if len(complement) >= len(selected):
        chosen = set(rng.choice(complement, len(selected), replace=False).tolist())
    else:
        chosen = set(complement)
        chosen.update(rng.choice(selected, len(selected) - len(complement), replace=False).tolist())
    result = [name for name in identifiers if name in chosen]
    if len(result) != len(selected) or set(result) == set(selected):
        raise AssertionError("Alternative group must have the same size and differ from selection")
    return result


def _scales(group, scale):
    return {name: float(scale) for name in group}


def _response_delta(native, changed, groups):
    native_difference, changed_difference = native[0] - native[1], changed[0] - changed[1]
    return float(np.median([np.sqrt(np.mean(changed_difference[groups == group] ** 2))
                            - np.sqrt(np.mean(native_difference[groups == group] ** 2))
                            for group in np.unique(groups)]))


def _direction(native, weak, control, groups, epsilon=1e-12):
    def inner(first, second):
        return float(np.mean([np.mean(first[arm][groups == group] * second[arm][groups == group])
                              for arm in (0, 1) for group in np.unique(groups)]))

    weak_change = tuple(value - base for value, base in zip(weak, native))
    control_change = tuple(value - base for value, base in zip(control, native))
    norm = np.sqrt(inner(weak_change, weak_change) * inner(control_change, control_change))
    weak_delta = _response_delta(native, weak, groups)
    control_delta = _response_delta(native, control, groups)
    return {"weak_response_delta": weak_delta, "control_response_delta": control_delta,
            "response_direction_concordant": (bool(np.sign(weak_delta) == np.sign(control_delta))
                                                if abs(weak_delta) > epsilon else None),
            "command_change_cosine": float(np.clip(inner(weak_change, control_change) / norm, -1., 1.))
                                      if norm > 1e-20 else None}


def _identity():
    return {"branch_scales": {}, "center": 0., "gain": 1., "offset": 0.}


def match_directional_gain(rows, target, weak_response_delta, settings):
    """Resolve magnitude ties only among controls with the intended direction."""
    if abs(weak_response_delta) <= settings["response_direction_epsilon"]:
        return None, "joint_response_change_below_direction_floor"
    if target <= 1e-12:
        return None, "immediate_effect_below_matching_floor"
    eligible = [row for row in rows if row["direction_eligible"]]
    if not eligible:
        return None, "no_gain_value_with_matching_response_direction"
    return _match(eligible, target, settings), None


def prepare_dynamics(base, protocol, tau, noise_tau, seed, model, group):
    """Calibrate selected-checkpoint interventions without confirmation access."""
    _validate_streams(protocol)
    config, settings = _config(base, protocol), protocol["dynamics"]
    group = _group(model, group)
    result = {"tau": tau, "noise_tau": noise_tau, "seed": seed, "group": group,
              "base_config_sha256": _digest(base), "protocol_sha256": _digest(protocol),
              "status": "unclassifiable" if group is None else "empty_group" if not group else "prepared",
              "variants": {"native": _identity()}, "alternative_group": None,
              "alternative_overlap_count": None, "alternative_jaccard": None,
              "control_grids": {}, "unavailable_controls": {}, "failure": None}
    if group is None:
        result["unavailable_controls"] = {name: "unclassifiable_selection" for name in ("gain_matched", "alternative_matched")}
        return result
    if not group:
        for label in ("joint_weak", "joint_strong"):
            result["variants"][label] = {**_identity(), "achieved_rms": 0., "identity_group": True}
        result["unavailable_controls"] = {name: "empty_group_identity" for name in ("gain_matched", "alternative_matched")}
        return result
    plant = make_plant(config, tau)
    bank, error = _bank(config, tau, noise_tau, NeuralPolicy(model, plant, config),
                        settings["calibration_seeds"], settings["pulse_amplitudes"])
    if error is not None:
        result.update(status="calibration_failed", failure=error)
        return result
    native_raw = _raw(model, bank, config)
    native = _clipped(native_raw, config)
    raw_variants = {}
    for label in ("weak", "strong"):
        scales = _scales(group, settings[f"{label}_scale"])
        raw = _raw(scale_branches(model, scales), bank, config)
        fit = _fit(native_raw, raw, bank, config)
        result["variants"][f"joint_{label}"] = {"branch_scales": scales, **fit}
        raw_variants[label] = raw
    weak = _clipped(raw_variants["weak"], config, result["variants"]["joint_weak"])
    target = result["variants"]["joint_weak"]["achieved_rms"]
    center = weighted_mean(native[1], bank["groups"])
    weak_response_delta = _response_delta(native, weak, bank["groups"])
    direction_epsilon = settings["response_direction_epsilon"]
    result["joint_weak_calibration_response_delta"] = weak_response_delta
    gain_grid = []
    for value in _grid(settings["gain_grid"]):
        fit = _fit(native_raw, native_raw, bank, config, center=center, gain=float(value))
        direction = _direction(native, weak, _clipped(native_raw, config, fit), bank["groups"], direction_epsilon)
        gain_grid.append({"parameter": float(value), **fit, **direction,
                          "direction_eligible": bool(direction["response_direction_concordant"])
                                                and abs(direction["control_response_delta"]) > direction_epsilon})
    gain, unavailable = match_directional_gain(gain_grid, target, weak_response_delta, settings)
    if unavailable is not None:
        result["unavailable_controls"]["gain_matched"] = unavailable
    else:
        result["variants"]["gain_matched"] = {"branch_scales": {}, **gain,
            "magnitude_matched": gain["matched"], "direction_matched": True,
            "matched_and_direction": gain["matched"]}
    result["control_grids"]["gain_matched"] = gain_grid
    alternative = alternative_group(branches(model), group, settings["alternative_seed"] + seed)
    result["alternative_group"] = alternative
    if alternative is None:
        result["unavailable_controls"]["alternative_matched"] = "all_branches_selected"
    else:
        overlap = len(set(group) & set(alternative))
        result["alternative_overlap_count"] = overlap
        result["alternative_jaccard"] = overlap / len(set(group) | set(alternative))
        grid, cached_raw = [], {}
        for scale in _grid(settings["alternative_grid"]):
            raw = _raw(scale_branches(model, _scales(alternative, scale)), bank, config)
            cached_raw[float(scale)] = raw
            grid.append({"parameter": float(scale), **_fit(native_raw, raw, bank, config)})
        match = _match(grid, target, settings)
        result["variants"]["alternative_matched"] = {
            "branch_scales": _scales(alternative, match["parameter"]), **match,
            **_direction(native, weak, _clipped(cached_raw[match["parameter"]], config, match), bank["groups"], direction_epsilon)}
        alternative_setting = result["variants"]["alternative_matched"]
        alternative_setting.update(magnitude_matched=match["matched"],
                                   direction_matched=alternative_setting["response_direction_concordant"],
                                   matched_and_direction=bool(match["matched"] and alternative_setting["response_direction_concordant"]))
        result["control_grids"]["alternative_matched"] = grid
    return result


def confirm_dynamics(base, protocol, tau, noise_tau, seed, model, prepared):
    """Evaluate frozen group/offset/scale settings on fresh histories and loops."""
    _validate_streams(protocol)
    if ((prepared["tau"], prepared["noise_tau"], prepared["seed"]) != (tau, noise_tau, seed)
            or prepared["base_config_sha256"] != _digest(base)
            or prepared["protocol_sha256"] != _digest(protocol)):
        raise ValueError("Prepared dynamics record does not match this model cell and protocol")
    config, settings = _config(base, protocol), protocol["dynamics"]
    plant = make_plant(config, tau)
    result = {"tau": tau, "noise_tau": noise_tau, "seed": seed,
              "group": prepared["group"], "status": prepared["status"],
              "fixed_history": None, "rollouts": [], "summary": {},
              "unavailable_controls": prepared["unavailable_controls"]}
    variants = prepared["variants"]
    if "joint_weak" in variants:
        bank, error = _bank(config, tau, noise_tau, NeuralPolicy(model, plant, config),
                            settings["confirmation_seeds"], settings["pulse_amplitudes"])
        result["fixed_history"] = {"available": error is None, "failure": error}
        if error is None:
            native = _clipped(_raw(model, bank, config), config)
            for label in ("joint_weak", "joint_strong"):
                setting = variants[label]
                changed = scale_branches(model, setting["branch_scales"])
                outputs = _clipped(_raw(changed, bank, config), config, setting)
                result["fixed_history"][label] = response_summary(
                    native, outputs, bank["groups"], bank["amplitudes"], config["diagnostics"])
    for label, setting in variants.items():
        changed = scale_branches(model, setting["branch_scales"])
        policy = NeuralPolicy(changed, plant, config, offset=setting["offset"],
                              gain=setting["gain"], center=setting["center"])
        shams, pulses, recoveries = [], [], []
        for noise_seed in settings["confirmation_seeds"]:
            sham = run_episode(config, tau, noise_tau, policy, noise_seed, duration=settings["duration"])
            sham_stats = metrics(config, sham, burn_in=settings["pulse_onset"])
            shams.append(sham_stats)
            for amplitude in settings["pulse_amplitudes"]:
                pulse = run_episode(config, tau, noise_tau, policy, noise_seed, duration=settings["duration"],
                                    pulse={"onset": settings["pulse_onset"], "width": settings["pulse_width"], "amplitude": amplitude})
                pulse_stats = metrics(config, pulse, burn_in=settings["pulse_onset"])
                recovery = paired_recovery(pulse, sham, settings["pulse_onset"], amplitude)
                pulses.append(pulse_stats)
                recoveries.append(recovery)
                result["rollouts"].append({"variant": label, "noise_seed": noise_seed, "amplitude": amplitude,
                                            "sham": sham_stats, "pulse": pulse_stats, "paired_recovery": recovery})
        result["summary"][label] = {"sham": _averages(shams), "pulse": _averages(pulses),
                                     "paired_recovery": _averages(recoveries),
                                     "independent_noise_seeds": len(shams),
                                     "matched": setting.get("matched"),
                                     "magnitude_matched": setting.get("magnitude_matched"),
                                     "direction_matched": setting.get("direction_matched"),
                                     "matched_and_direction": setting.get("matched_and_direction"),
                                     "match_relative_error": setting.get("relative_error"),
                                     "response_direction_concordant": setting.get("response_direction_concordant"),
                                     "command_change_cosine": setting.get("command_change_cosine")}
    native = result["summary"]["native"]
    for summary in result["summary"].values():
        summary["delta_from_native"] = {
            arm: {key: value - native[arm][key] if value is not None and native[arm].get(key) is not None else None
                  for key, value in summary[arm].items()}
            for arm in ("sham", "pulse", "paired_recovery")}
    return result


def fit_harmonic(times, values, frequency):
    """Fit a*sin(ωt)+b*cos(ωt)+DC; positive phase denotes a sinusoid lead."""
    times, values = np.asarray(times, dtype=float), np.asarray(values, dtype=float)
    if (times.ndim != 1 or values.shape != times.shape or len(times) < 3
            or not np.isfinite(times).all() or not np.isfinite(values).all()
            or not np.isfinite(frequency) or frequency <= 0):
        raise ValueError("Harmonic fitting needs finite one-dimensional data and positive frequency")
    matrix = np.column_stack((np.sin(2 * np.pi * frequency * times),
                              np.cos(2 * np.pi * frequency * times), np.ones(len(times))))
    coefficients, _, rank, _ = np.linalg.lstsq(matrix, values, rcond=None)
    if rank < 3:
        raise ValueError("Harmonic design is rank deficient at the sampled frequency")
    sine, cosine, dc = coefficients
    return {"amplitude": float(np.hypot(sine, cosine)), "phase_rad": float(np.arctan2(cosine, sine)),
            "dc": float(dc), "fit_residual_rms": float(np.sqrt(np.mean((values - matrix @ coefficients) ** 2))),
            "response_rms": float(np.sqrt(np.mean(values ** 2)))}


def _array_digest(bank):
    digest = hashlib.sha256()
    for key in ("decision_times", "pulse", "sham"):
        arrays = bank[key] if isinstance(bank[key], tuple) else (bank[key],)
        for values in arrays:
            values = np.asarray(values)
            digest.update(key.encode())
            digest.update(str(values.shape).encode())
            digest.update(values.dtype.str.encode())
            digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def frequency_bank(base, protocol, tau, frequency, phase):
    """Deterministically cached physical teacher histories, common across cells."""
    settings = protocol["frequency"]
    if frequency <= 0 or frequency >= .5 / base["period"]:
        raise ValueError("Drive frequency must lie strictly below the decision-sampling Nyquist frequency")
    if not 0 <= settings["burn_in"] < settings["duration"]:
        raise ValueError("Frequency burn-in must lie within the episode")
    key = (_digest(base), _digest(settings), float(tau), float(frequency), float(phase))
    if key in _FREQUENCY_BANKS:
        return _FREQUENCY_BANKS[key]
    plant = make_plant(base, tau)
    teacher = teacher_policy(base, plant)
    arms, all_times = {}, []
    for label, amplitude in (("pulse", settings["amplitude"]), ("sham", 0.)):
        tokens, masks, times = [], [], []

        def record(snapshot):
            encoded, valid = encode(snapshot, plant, base)
            tokens.append(encoded)
            masks.append(valid)
            times.append(snapshot.time)
            return teacher(snapshot)

        signal = sinusoid_tape(duration=settings["duration"], dt=base["noise_dt"],
                                frequency=frequency, amplitude=amplitude, phase=phase)
        run = simulate(plant, record, make_timing(base, settings["duration"], base.get("sample_interval", .01)),
                       disturbance=signal.to_piecewise_constant())
        if run.failure_reason is not None:
            raise ValueError(f"Teacher frequency bank failed: {run.failure_reason}")
        times = np.asarray(times)
        mask = (times >= settings["burn_in"] - 1e-10) & (times < settings["duration"] - 1e-10)
        arms[label] = (np.asarray(tokens)[mask], np.asarray(masks)[mask])
        all_times.append(times[mask])
    if not np.array_equal(*all_times):
        raise ValueError("Teacher frequency and sham decision times differ")
    bank = {**arms, "decision_times": all_times[0]}
    bank["sha256"] = _array_digest(bank)
    _FREQUENCY_BANKS[key] = bank
    return bank


def frequency_assay(base, protocol, tau, noise_tau, seed, models, groups):
    """Conditional pre-application command harmonics on teacher histories."""
    settings = protocol["frequency"]
    rows = []
    for checkpoint, model in models.items():
        group = _group(model, groups[checkpoint])
        changed = scale_branches(model, _scales(group, settings["weak_scale"])) if group is not None else None
        for frequency in settings["frequencies"]:
            for phase in settings["phases"]:
                bank = frequency_bank(base, protocol, tau, frequency, phase)
                row = {"checkpoint": checkpoint, "frequency": frequency, "phase": phase,
                       "group": group, "group_size": len(group) if group is not None else None,
                       "bank_sha256": bank["sha256"], "available": group is not None,
                       "decision_samples": len(bank["decision_times"])}
                raw_native = _raw(model, bank, base)
                native = _clipped(raw_native, base)
                fit_native = fit_harmonic(bank["decision_times"], native[0] - native[1], frequency)
                row.update({f"native_{key}": value for key, value in fit_native.items()})
                row["native_clip_fraction"] = float(np.mean(np.abs(np.concatenate(raw_native)) >= base["action_limit"]))
                row["baseline_valid"] = fit_native["amplitude"] > settings["amplitude_epsilon"]
                if not row["baseline_valid"]:
                    row["native_phase_rad"] = None
                row.update(weak_amplitude=None, amplitude_change=None, gain_ratio_percent=None,
                           phase_difference_rad=None, phase_difference_deg=None, phase_valid=False,
                           weak_response_rms=None, weak_fit_residual_rms=None, weak_dc=None,
                           weak_phase_rad=None, weak_clip_fraction=None)
                if changed is not None:
                    raw_changed = _raw(changed, bank, base)
                    outputs = _clipped(raw_changed, base)
                    fit_changed = fit_harmonic(bank["decision_times"], outputs[0] - outputs[1], frequency)
                    row.update({f"weak_{key}": value for key, value in fit_changed.items()})
                    row["weak_clip_fraction"] = float(np.mean(np.abs(np.concatenate(raw_changed)) >= base["action_limit"]))
                    row["amplitude_change"] = fit_changed["amplitude"] - fit_native["amplitude"]
                    if row["baseline_valid"]:
                        row["gain_ratio_percent"] = (fit_changed["amplitude"] / fit_native["amplitude"] - 1) * 100
                    row["phase_valid"] = row["baseline_valid"] and fit_changed["amplitude"] > settings["amplitude_epsilon"]
                    if fit_changed["amplitude"] <= settings["amplitude_epsilon"]:
                        row["weak_phase_rad"] = None
                    if row["phase_valid"]:
                        difference = fit_changed["phase_rad"] - fit_native["phase_rad"]
                        row["phase_difference_rad"] = float(np.arctan2(np.sin(difference), np.cos(difference)))
                        row["phase_difference_deg"] = float(np.rad2deg(row["phase_difference_rad"]))
                rows.append(row)
    return {"tau": tau, "noise_tau": noise_tau, "seed": seed,
            "scope": "Pre-application command harmonics on common teacher-generated histories; "
                     "conditional pathway effects, not a native neural closed-loop transfer function or phase margin.",
            "phase_convention": "Fit a*sin(2*pi*f*t)+b*cos(2*pi*f*t)+DC. Positive atan2(b,a) is a lead. "
                                "Reported phase difference is joint-weak minus native, wrapped to [-pi,pi].",
            "sampling_interval": base["period"], "nyquist_hz": .5 / base["period"], "rows": rows}
