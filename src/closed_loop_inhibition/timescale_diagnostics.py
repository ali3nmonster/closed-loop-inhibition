"""Prospective fixed-history assays and own-history suppression interventions.

Preparation uses discovery and calibration streams only. Callers must persist
its returned JSON before calling confirmation. Common teacher histories measure
functional potential across checkpoints; they are not initialization rollouts.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from .centered_calibration import centered_commands, fit_command_offset, weighted_mean
from .interventions import enumerate_attention_heads, head_residuals, scale_attention_head
from .suppression_analysis import immediate_output_rms, norm_match_head, pick_rms_match


def response_summary(native, changed, groups, amplitudes, settings):
    """Equal-scenario response changes with a floor in physical command units."""
    native_pulse, native_sham = (np.asarray(x, dtype=float) for x in native)
    changed_pulse, changed_sham = (np.asarray(x, dtype=float) for x in changed)
    groups, amplitudes = np.asarray(groups), np.asarray(amplitudes, dtype=float)
    arrays = (native_pulse, native_sham, changed_pulse, changed_sham, groups, amplitudes)
    if any(array.ndim != 1 or len(array) != len(groups) for array in arrays) or not len(groups):
        raise ValueError("paired arrays must be nonempty one-dimensional arrays of equal length")
    if not all(np.isfinite(array).all() for array in (*arrays[:4], amplitudes)):
        raise ValueError("commands and amplitudes must be finite")
    rows, changes = [], []
    for group in np.unique(groups):
        mask = groups == group
        amplitude = float(amplitudes[mask][0])
        if amplitude == 0 or not np.all(amplitudes[mask] == amplitude):
            raise ValueError("each group requires one nonzero pulse amplitude")
        baseline = float(np.sqrt(np.mean((native_pulse[mask] - native_sham[mask]) ** 2)))
        modified = float(np.sqrt(np.mean((changed_pulse[mask] - changed_sham[mask]) ** 2)))
        valid = baseline > settings["baseline_epsilon"]
        fractional = modified / baseline - 1 if valid else None
        if valid:
            changes.append(fractional)
        rows.append({"group": str(group), "amplitude": amplitude,
                     "baseline_response_rms": baseline, "changed_response_rms": modified,
                     "absolute_response_change": modified - baseline,
                     "normalized_baseline_response_rms": baseline / abs(amplitude),
                     "normalized_changed_response_rms": modified / abs(amplitude),
                     "baseline_valid": valid, "fractional_increase": fractional})
    median = float(np.median(changes)) if changes else None
    positive_fraction = float(np.mean(np.asarray(changes) > 0)) if changes else 0.0
    eligible = all(row["baseline_valid"] for row in rows) and median is not None
    eligible = eligible and median > settings["min_fractional_increase"]
    eligible = eligible and positive_fraction >= settings["min_positive_fraction"]
    return {"per_group": rows, "median_fractional_increase": median,
            "median_absolute_response_change": float(np.median([
                row["absolute_response_change"] for row in rows])),
            "positive_fraction": positive_fraction, "eligible": bool(eligible)}


def choose_candidate(rows):
    """Maximum discovery median among eligible heads, with stable index ties."""
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        return None
    row = min(eligible, key=lambda x: (-x["median_fractional_increase"], x["layer"], x["head"]))
    return {"layer": row["layer"], "head": row["head"],
            "median_fractional_increase": row["median_fractional_increase"]}


def _settings(config):
    settings = config["diagnostics"]
    streams = [set(settings[f"{stage}_seeds"]) for stage in ("discovery", "calibration", "confirmation")]
    if any(streams[a] & streams[b] for a in range(3) for b in range(a + 1, 3)):
        raise ValueError("discovery, calibration, and confirmation streams must be disjoint")
    if not all(streams) or not settings["pulse_amplitudes"]:
        raise ValueError("diagnostic streams and pulse amplitudes must be nonempty")
    return settings


def _bank(config, tau, noise_tau, policy, seeds, amplitudes):
    from .timescale_maps import run_episode
    settings = _settings(config)
    pulse_tokens, sham_tokens, pulse_valid, sham_valid = [], [], [], []
    groups, amplitude_rows = [], []
    for noise_seed in seeds:
        sham, sham_probe = run_episode(config, tau, noise_tau, policy, noise_seed,
                                       duration=settings["duration"], record=True)
        if sham.failure_reason is not None:
            return None, {"reason": "sham_rollout_failed", "seed": noise_seed,
                          "failure_reason": sham.failure_reason}
        for amplitude in amplitudes:
            pulse, pulse_probe = run_episode(
                config, tau, noise_tau, policy, noise_seed, duration=settings["duration"],
                pulse={"onset": settings["pulse_onset"], "width": settings["pulse_width"],
                       "amplitude": amplitude}, record=True)
            if pulse.failure_reason is not None:
                return None, {"reason": "pulse_rollout_failed", "seed": noise_seed,
                              "amplitude": amplitude, "failure_reason": pulse.failure_reason}
            times = pulse_probe["decision_times"]
            if not np.array_equal(times, sham_probe["decision_times"]):
                raise ValueError("paired probe decision times differ")
            mask = (times >= settings["pulse_onset"] - 1e-10) & (times < settings["duration"] - 1e-10)
            if not np.any(mask):
                raise ValueError("no post-pulse decision histories")
            pulse_tokens.append(pulse_probe["tokens"][mask])
            sham_tokens.append(sham_probe["tokens"][mask])
            pulse_valid.append(pulse_probe["valid"][mask])
            sham_valid.append(sham_probe["valid"][mask])
            groups.extend([f"seed_{noise_seed}_amplitude_{amplitude:g}"] * int(mask.sum()))
            amplitude_rows.extend([amplitude] * int(mask.sum()))
    return {"pulse": (np.concatenate(pulse_tokens), np.concatenate(pulse_valid)),
            "sham": (np.concatenate(sham_tokens), np.concatenate(sham_valid)),
            "groups": np.asarray(groups), "amplitudes": np.asarray(amplitude_rows)}, None


def _raw(model, bank, config):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for arm in ("pulse", "sham"):
            tokens, valid = bank[arm]
            batches = [model(torch.as_tensor(tokens[start:start + 512], dtype=torch.float32),
                             torch.as_tensor(valid[start:start + 512], dtype=torch.bool))
                       .detach().cpu().numpy().reshape(-1)
                       for start in range(0, len(tokens), 512)]
            outputs.append(np.concatenate(batches).astype(float) * config["output_scale"])
    return tuple(outputs)


def _clipped(raw, config, setting=None):
    setting = setting or {}
    return tuple(centered_commands(values, action_limit=config["action_limit"],
                                    center=setting.get("center", 0), gain=setting.get("gain", 1),
                                    offset=setting.get("offset", 0)) for values in raw)


def _head_assay(model, bank, config):
    native = _clipped(_raw(model, bank, config), config)
    rows = []
    for layer, head in enumerate_attention_heads(model):
        changed = scale_attention_head(model, layer, head, config["diagnostics"]["weak_scale"])
        summary = response_summary(native, _clipped(_raw(changed, bank, config), config),
                                   bank["groups"], bank["amplitudes"], config["diagnostics"])
        rows.append({"layer": layer, "head": head, **summary})
    classifiable = all(group["baseline_valid"] for group in rows[0]["per_group"])
    count = sum(row["eligible"] for row in rows)
    return {"heads": rows, "classifiable": classifiable,
            "eligible_count": count if classifiable else None, "head_count": len(rows),
            "eligible_fraction": count / len(rows) if classifiable else None,
            "selected": choose_candidate(rows)}


def _residual_norms(model, bank):
    """RMS at the decision token, equal scenario/arm weighting."""
    groups = bank["groups"]
    arm_norms = []
    with torch.inference_mode():
        for arm in ("pulse", "sham"):
            tokens, valid = bank[arm]
            norm_batches = []
            for start in range(0, len(tokens), 128):
                batch_tokens = torch.as_tensor(tokens[start:start + 128], dtype=torch.float32)
                batch_valid = torch.as_tensor(valid[start:start + 128], dtype=torch.bool)
                residual = head_residuals(model, batch_tokens, batch_valid)
                last = batch_valid.long().sum(dim=1) - 1
                selected = residual[torch.arange(len(last)), last]
                norm_batches.append(selected.square().mean(dim=-1).cpu().numpy())
            squares = np.concatenate(norm_batches)
            arm_norms.append(np.mean([squares[groups == group].mean(axis=0)
                                     for group in np.unique(groups)], axis=0))
    norms = np.sqrt(np.mean(arm_norms, axis=0))
    return [{"layer": layer, "head": head, "residual_rms": float(norms[layer, head])}
            for layer, head in enumerate_attention_heads(model)]


def _fit(native_raw, changed_raw, bank, config, *, center=0.0, gain=1.0):
    transformed = tuple(center + gain * (raw - center) for raw in changed_raw)
    fit = fit_command_offset(native_raw[1], transformed[1], bank["groups"], config["action_limit"])
    setting = {**fit, "center": float(center), "gain": float(gain)}
    outputs = _clipped(changed_raw, config, setting)
    setting["achieved_rms"] = immediate_output_rms(*_clipped(native_raw, config), *outputs, bank["groups"])
    return setting


def _grid(spec):
    return np.round(np.arange(spec["min"], spec["max"] + spec["step"] / 2, spec["step"]), 10)


def _match(rows, target, settings):
    if target <= 1e-12:
        identity = next(row for row in rows if row["parameter"] == 1.0)
        return {**identity, "matched": False, "target_rms": target, "relative_error": None,
                "reason": "target_below_epsilon"}
    match = pick_rms_match([row["parameter"] for row in rows],
                          [row["achieved_rms"] for row in rows], target,
                          tolerance=settings["match_tolerance"])
    selected = next(row for row in rows if row["parameter"] == match["scale"])
    return {**selected, **{key: value for key, value in match.items() if key != "scale"},
            "reason": None if match["matched"] else "no_grid_value_within_tolerance"}


def _calibrate(model, bank, config, candidate):
    settings = config["diagnostics"]
    native_raw = _raw(model, bank, config)
    variants = {"native": {"head": "native", "head_scale": 1.0,
                            "center": 0.0, "gain": 1.0, "offset": 0.0}}
    for label in ("weak", "strong"):
        scale = settings[f"{label}_scale"]
        changed = scale_attention_head(model, candidate["layer"], candidate["head"], scale)
        variants[f"centered_{label}"] = {
            "head": "candidate", "head_scale": scale,
            **_fit(native_raw, _raw(changed, bank, config), bank, config)}
    target = variants["centered_weak"]["achieved_rms"]
    norms = _residual_norms(model, bank)
    alternative = norm_match_head(norms, candidate["layer"], candidate["head"])
    alternative_grid = []
    for scale in _grid(settings["alternative_grid"]):
        changed = scale_attention_head(model, alternative["layer"], alternative["head"], float(scale))
        alternative_grid.append({"parameter": float(scale),
                                 **_fit(native_raw, _raw(changed, bank, config), bank, config)})
    center = weighted_mean(_clipped(native_raw, config)[1], bank["groups"])
    gain_grid = [{"parameter": float(gain),
                  **_fit(native_raw, native_raw, bank, config, center=center, gain=float(gain))}
                 for gain in _grid(settings["gain_grid"])]
    matched_alt = _match(alternative_grid, target, settings)
    matched_gain = _match(gain_grid, target, settings)
    variants["alternative_matched"] = {"head": "alternative", "head_scale": matched_alt["parameter"],
                                        **matched_alt}
    variants["gain_matched"] = {"head": "native", "head_scale": 1.0, **matched_gain}
    return {"variants": variants, "alternative": alternative, "head_residual_rms": norms,
            "control_grids": {"alternative_matched": alternative_grid, "gain_matched": gain_grid}}


def prepare_model(config, tau, noise_tau, seed, model, checkpoint_dir):
    """Select and calibrate without touching confirmation random streams."""
    from .timescale_maps import NeuralPolicy, make_plant, teacher_policy
    settings, plant = _settings(config), make_plant(config, tau)
    result = {"tau": tau, "noise_tau": noise_tau, "seed": seed,
              "common_history": {}, "discovery": None, "calibration": None}
    common_config = deepcopy(config)
    common_config["noise_std"] = 0.0
    common_bank, error = _bank(common_config, tau, noise_tau, teacher_policy(config, plant),
                               [settings["discovery_seeds"][0]], settings["common_amplitudes"])
    if error is not None:
        raise ValueError(f"common teacher assay failed: {error}")
    checkpoint_dir = Path(checkpoint_dir)
    intermediate_epoch = config.get("training", {}).get("intermediate_epoch", 25)
    for label, filename in (("initial", "initial.pt"), ("epoch_025", f"epoch_{intermediate_epoch:03d}.pt"),
                            ("selected", "selected.pt")):
        checkpoint = torch.load(checkpoint_dir / filename, map_location="cpu", weights_only=True)
        checkpoint_model = deepcopy(model)
        checkpoint_model.load_state_dict(checkpoint["state_dict"])
        result["common_history"][label] = {"epoch": checkpoint["epoch"],
                                           **_head_assay(checkpoint_model, common_bank, config)}
    native_policy = NeuralPolicy(model, plant, config)
    discovery_bank, error = _bank(config, tau, noise_tau, native_policy,
                                  settings["discovery_seeds"], settings["pulse_amplitudes"])
    if error is not None:
        result["discovery"] = {"selected": None, "available": False, "failure": error,
                                 "eligible_count": None, "eligible_fraction": None}
        return result
    result["discovery"] = {"available": True, "failure": None,
                             **_head_assay(model, discovery_bank, config)}
    candidate = result["discovery"]["selected"]
    if candidate is None:
        return result
    result["common_history"]["selected_path_by_checkpoint"] = {
        label: next(row for row in result["common_history"][label]["heads"]
                    if (row["layer"], row["head"]) == (candidate["layer"], candidate["head"]))
        for label in ("initial", "epoch_025", "selected")}
    calibration_bank, error = _bank(config, tau, noise_tau, native_policy,
                                    settings["calibration_seeds"], settings["pulse_amplitudes"])
    if error is not None:
        result["calibration"] = {"available": False, "failure": error}
        return result
    result["calibration"] = {"available": True, "failure": None,
                              **_calibrate(model, calibration_bank, config, candidate)}
    return result


def paired_recovery(pulse, sham, onset, amplitude):
    """Own-history pulse-minus-sham state and action energy after pulse onset."""
    missing = {"complete": False, "incremental_position_rms": None,
               "normalized_position_energy": None, "incremental_action_rms": None}
    if pulse.failure_reason is not None or sham.failure_reason is not None:
        return missing
    if not pulse.samples or not sham.samples:
        return missing
    if pulse.config.duration != sham.config.duration:
        raise ValueError("paired recovery durations differ")
    end = pulse.config.duration
    if not 0 <= onset < end or not np.isfinite(amplitude) or amplitude == 0:
        raise ValueError("recovery requires a valid onset and nonzero finite amplitude")
    if any(run.samples[0]["time"] > 1e-10 or run.samples[-1]["time"] < end - 1e-10
           for run in (pulse, sham)):
        return missing
    times = np.asarray([row["time"] for row in pulse.samples])
    sham_times = np.asarray([row["time"] for row in sham.samples])
    if not np.array_equal(times, sham_times):
        raise ValueError("paired report times differ")
    mask = (times > onset) & (times < end)
    selected_times = np.concatenate(([onset], times[mask], [end]))
    horizon = end - onset
    difference = np.asarray([a["position"] - b["position"]
                             for a, b in zip(pulse.samples, sham.samples)])
    q = np.concatenate(([np.interp(onset, times, difference)], difference[mask],
                        [np.interp(end, times, difference)]))
    q_energy = float(np.trapezoid(q * q, selected_times))
    # Exact held-action integration, independent of the reporting grid.
    boundaries = np.asarray(sorted({onset, end} | {
        action.time for run in (pulse, sham) for action in run.applied_actions
        if onset < action.time < end}))
    action_rows = []
    for run in (pulse, sham):
        action_times = np.asarray([-1.] + [action.time for action in run.applied_actions])
        action_values = np.asarray([0.] + [action.value for action in run.applied_actions])
        indices = np.searchsorted(action_times, boundaries[:-1], side="right") - 1
        action_rows.append(action_values[indices])
    action_energy = float(np.sum((action_rows[0] - action_rows[1]) ** 2 * np.diff(boundaries)))
    return {"complete": True, "incremental_position_rms": float(np.sqrt(q_energy / horizon)),
            "normalized_position_energy": q_energy / amplitude ** 2,
            "incremental_action_rms": float(np.sqrt(action_energy / horizon))}


def _changed_model(model, prepared, setting):
    if setting["head"] == "native":
        return model
    head = prepared["discovery"]["selected"] if setting["head"] == "candidate" else prepared["calibration"]["alternative"]
    return scale_attention_head(model, head["layer"], head["head"], setting["head_scale"])


def _averages(rows):
    keys = sorted({key for row in rows for key in row if key != "failure_reason"})
    return {key: (float(np.mean([row[key] for row in rows]))
                  if all(isinstance(row.get(key), (int, float, bool)) and row[key] is not None for row in rows)
                  else None) for key in keys}


def confirm_model(config, tau, noise_tau, seed, model, prepared):
    """Apply frozen settings to held-out histories and independent own loops."""
    from .timescale_maps import NeuralPolicy, make_plant, metrics, run_episode
    if (prepared["tau"], prepared["noise_tau"], prepared["seed"]) != (tau, noise_tau, seed):
        raise ValueError("prepared diagnostic cell does not match confirmation cell")
    settings, plant = _settings(config), make_plant(config, tau)
    calibration = prepared["calibration"]
    variants = calibration["variants"] if calibration and calibration["available"] else {
        "native": {"head": "native", "head_scale": 1.0, "center": 0.0, "gain": 1.0, "offset": 0.0}}
    result = {"tau": tau, "noise_tau": noise_tau, "seed": seed,
              "fixed_history": None, "rollouts": [], "summary": {}}
    native = NeuralPolicy(model, plant, config)
    if calibration and calibration["available"]:
        bank, error = _bank(config, tau, noise_tau, native, settings["confirmation_seeds"],
                            settings["pulse_amplitudes"])
        if error is not None:
            result["fixed_history"] = {"available": False, "failure": error}
        else:
            baseline = _clipped(_raw(model, bank, config), config)
            fixed = {"available": True, "failure": None}
            for label in ("weak", "strong"):
                setting = variants[f"centered_{label}"]
                raw = _raw(_changed_model(model, prepared, setting), bank, config)
                for prefix, transform in (("raw", None), ("centered", setting)):
                    fixed[f"{prefix}_{label}"] = response_summary(
                        baseline, _clipped(raw, config, transform), bank["groups"],
                        bank["amplitudes"], settings)
            result["fixed_history"] = fixed
    for variant, setting in variants.items():
        changed = _changed_model(model, prepared, setting)
        policy = NeuralPolicy(changed, plant, config, offset=setting["offset"],
                              gain=setting["gain"], center=setting["center"])
        sham_rows, pulse_rows, recovery_rows = [], [], []
        for noise_seed in settings["confirmation_seeds"]:
            sham = run_episode(config, tau, noise_tau, policy, noise_seed, duration=settings["duration"])
            sham_metrics = metrics(config, sham, burn_in=settings["pulse_onset"])
            sham_rows.append(sham_metrics)
            for amplitude in settings["pulse_amplitudes"]:
                pulse = run_episode(config, tau, noise_tau, policy, noise_seed, duration=settings["duration"],
                                    pulse={"onset": settings["pulse_onset"], "width": settings["pulse_width"],
                                           "amplitude": amplitude})
                pulse_metrics = metrics(config, pulse, burn_in=settings["pulse_onset"])
                recovery = paired_recovery(pulse, sham, settings["pulse_onset"], amplitude)
                pulse_rows.append(pulse_metrics)
                recovery_rows.append(recovery)
                result["rollouts"].append({"variant": variant, "noise_seed": noise_seed,
                                            "amplitude": amplitude, "sham": sham_metrics,
                                            "pulse": pulse_metrics, "paired_recovery": recovery})
        result["summary"][variant] = {"sham": _averages(sham_rows), "pulse": _averages(pulse_rows),
                                       "paired_recovery": _averages(recovery_rows),
                                       "independent_noise_seeds": len(sham_rows),
                                       "matched": setting.get("matched"),
                                       "match_relative_error": setting.get("relative_error")}
    native_summary = result["summary"]["native"]
    for variant, summary in result["summary"].items():
        summary["delta_from_native"] = {
            arm: {key: (value - native_summary[arm][key]
                         if value is not None and native_summary[arm].get(key) is not None else None)
                  for key, value in summary[arm].items()}
            for arm in ("sham", "pulse", "paired_recovery")}
    return result
