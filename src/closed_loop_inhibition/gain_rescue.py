"""Per-delay scalar controls fitted on independent native closed-loop histories.

A positive least-squares gain matches the paired pulse-minus-sham command
waveform. The offset separately matches the native sham command mean, with
actuator clipping included. Thus ``gain_matched`` approximates the weak response
waveform while retaining the native calibration mean; ``weak_rescue`` attempts
to restore both. Neither fit claims to preserve an equilibrium or transfer
function. Confirmation always uses the frozen calibration settings.
"""

from copy import deepcopy

import numpy as np

from .centered_calibration import fit_command_offset, weighted_mean
from .collective_dynamics import _array_digest, _digest
from .collective_suppression import scale_branches
from .delay_sweep import (
    FixedDelayPolicy, _bank_from_records, _configuration, _evaluate_variants,
)
from .timescale_diagnostics import _clipped, _raw, response_summary
from .timescale_maps import make_plant


INHERITED_VARIANTS = ("native", "joint_weak", "joint_strong")
CONTROL_VARIANTS = ("gain_matched", "weak_rescue")


def _validate(protocol):
    streams = [protocol[name] for name in ("calibration_seeds", "confirmation_seeds")]
    if (any(not values or len(values) != len(set(values)) or
            any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values)
            for values in streams) or set(streams[0]) & set(streams[1])):
        raise ValueError("Calibration and confirmation streams must be distinct, disjoint nonempty integer lists")
    bounds = protocol["gain_bounds"]
    if (len(bounds) != 2 or not np.isfinite(bounds).all() or not 0 < bounds[0] < bounds[1]
            or not np.isfinite(protocol["response_energy_floor"]) or protocol["response_energy_floor"] <= 0):
        raise ValueError("Gain bounds must be positive ordered finite values and response energy floor positive")


def _vectors(pair, groups, name):
    if len(pair) != 2:
        raise ValueError(f"{name} requires pulse and sham arrays")
    values = tuple(np.asarray(value, dtype=float) for value in pair)
    if (not len(groups) or np.asarray(groups).ndim != 1
            or any(value.ndim != 1 or len(value) != len(groups) or not np.isfinite(value).all() for value in values)):
        raise ValueError(f"{name} requires nonempty finite paired vectors matching groups")
    return values


def fit_response_gain(source_raw, target, native, groups, action_limit, gain_bounds, response_energy_floor):
    """Fit equal-probe LS pulse responses, then center clipped sham commands.

    The gain minimizes ``mean_groups((g * (source_p-source_s) -
    (target_p-target_s))**2)`` before clipping, constrained to fixed positive
    bounds. Clipping can prevent waveform matching; diagnostics report that
    residual rather than optimizing confirmation performance. ``native`` is
    the clipped mean reference, distinct from the response target.
    """
    groups = np.asarray(groups)
    source_raw = _vectors(source_raw, groups, "source_raw")
    target = _vectors(target, groups, "target")
    native = _vectors(native, groups, "native")
    if (len(gain_bounds) != 2 or not np.isfinite(gain_bounds).all()
            or not 0 < gain_bounds[0] < gain_bounds[1]
            or not np.isfinite(response_energy_floor) or response_energy_floor <= 0):
        raise ValueError("Invalid positive gain bounds or response energy floor")
    source_response, target_response = source_raw[0] - source_raw[1], target[0] - target[1]
    denominator = weighted_mean(source_response ** 2, groups)
    if denominator <= response_energy_floor:
        return None, "source_response_below_energy_floor"
    target_energy = weighted_mean(target_response ** 2, groups)
    if target_energy <= response_energy_floor:
        return None, "target_response_below_energy_floor"
    unconstrained = weighted_mean(source_response * target_response, groups) / denominator
    gain = float(np.clip(unconstrained, *gain_bounds))
    center = weighted_mean(source_raw[1], groups)
    transformed_sham = center + gain * (source_raw[1] - center)
    offset_fit = fit_command_offset(native[1], transformed_sham, groups, action_limit)
    return {"center": center, "gain": gain, "offset": offset_fit["offset"],
            "unconstrained_gain": float(unconstrained),
            "gain_at_bound": bool(gain != unconstrained), "gain_bounds": list(gain_bounds),
            "source_response_energy": denominator, "target_response_energy": target_energy,
            "objective": "equal_probe_preclip_pulse_minus_sham_least_squares",
            "mean_reference": "native_sham", "mean_fit": offset_fit}, None


def waveform_diagnostics(output, target, native, weak, groups, floor):
    """Retain response-shape error without refitting on the evaluation bank."""
    groups = np.asarray(groups)
    output, target, native, weak = [_vectors(pair, groups, name) for pair, name in
                                    ((output, "output"), (target, "target"), (native, "native"), (weak, "weak"))]
    response = output[0] - output[1]
    desired = target[0] - target[1]
    intervention = (weak[0] - weak[1]) - (native[0] - native[1])
    def measure(mask):
        def mean(value):
            return weighted_mean(value[mask], groups[mask])
        error_energy = mean((response - desired) ** 2)
        source_energy, target_energy = mean(response ** 2), mean(desired ** 2)
        intervention_energy = mean(intervention ** 2)
        return {"response_error_rms": float(np.sqrt(error_energy)),
                "output_response_rms": float(np.sqrt(source_energy)),
                "target_response_rms": float(np.sqrt(target_energy)),
                "relative_response_error": float(np.sqrt(error_energy / target_energy)) if target_energy > floor else None,
                "intervention_response_rms": float(np.sqrt(intervention_energy)),
                "residual_over_intervention": float(np.sqrt(error_energy / intervention_energy)) if intervention_energy > floor else None,
                "response_cosine": float(np.clip(mean(response * desired) / np.sqrt(source_energy * target_energy), -1., 1.))
                                    if source_energy > floor and target_energy > floor else None,
                "sham_mean_change_from_native": mean(output[1] - native[1]),
                "sham_mean_change_from_target": mean(output[1] - target[1]),
                "command_error_rms": float(np.sqrt((mean((output[0] - target[0]) ** 2)
                                                       + mean((output[1] - target[1]) ** 2)) / 2))}
    return {**measure(np.ones(len(groups), dtype=bool)),
            "per_group": [{"group": str(group), **measure(groups == group)} for group in np.unique(groups)]}


def _policies(model, config, tau, protocol, variants):
    return {name: FixedDelayPolicy(scale_branches(model, setting["branch_scales"]), make_plant(config, tau), config,
                                  offset=setting["offset"], gain=setting["gain"], center=setting["center"],
                                  delay_cue=protocol["delay_cue"])
            for name, setting in variants.items()}


def _history(model, bank, config, protocol, variants):
    raw = {name: _raw(scale_branches(model, setting["branch_scales"]), bank, config)
           for name, setting in variants.items()}
    output = {name: _clipped(raw[name], config, setting) for name, setting in variants.items()}
    native, weak = output["native"], output["joint_weak"]
    replay_error = max(float(np.max(np.abs(value - bank[f"{arm}_commands"])))
                       for value, arm in zip(native, ("pulse", "sham")))
    if replay_error > 1e-6:
        raise AssertionError("Native fixed-history replay does not reproduce recorded commands")
    result = {"available": True, "failure": None, "bank_sha256": _array_digest(bank),
              "decision_rows": len(bank["groups"]), "probe_groups": len(np.unique(bank["groups"])),
              "native_replay_max_abs_error": replay_error, "native_replay_tolerance": 1e-6,
              "native_replay_passed": True, "variants": {}, "waveforms": []}
    for name, values in output.items():
        target_name = "joint_weak" if name == "gain_matched" else "native"
        setting = variants[name]
        preclip = tuple(setting["center"] + setting["gain"] * (value - setting["center"]) + setting["offset"]
                        for value in raw[name])
        result["variants"][name] = {
            "target_response_label": target_name,
            "response": response_summary(native, values, bank["groups"], bank["amplitudes"], protocol),
            "waveform": waveform_diagnostics(values, output[target_name], native, weak, bank["groups"], protocol["response_energy_floor"]),
            "clipping_fraction": {arm: weighted_mean((np.abs(value) >= config["action_limit"]).astype(float), bank["groups"])
                                  for arm, value in zip(("pulse", "sham"), preclip)},
        }
    for group in np.unique(bank["groups"]):
        mask = bank["groups"] == group
        result["waveforms"].append({"group": str(group), "amplitude": float(bank["amplitudes"][mask][0]),
                                   "decision_times": bank["decision_times"][mask].tolist(),
                                   "responses": {name: (value[0][mask] - value[1][mask]).tolist()
                                                 for name, value in output.items()},
                                   "raw_commands": {name: {arm: values[mask].tolist() for arm, values in
                                                          zip(("pulse", "sham"), pair)} for name, pair in raw.items()},
                                   "commands": {name: {arm: values[mask].tolist() for arm, values in
                                                      zip(("pulse", "sham"), pair)} for name, pair in output.items()}})
    return result


def _run(config, protocol, tau, noise_tau, model, variants, seeds):
    stream_protocol = {**protocol, "confirmation_seeds": seeds}
    evaluated, records = _evaluate_variants(config, stream_protocol, tau, noise_tau,
                                           _policies(model, config, tau, protocol, variants), variants, record_native=True)
    for summary in evaluated["summary"].values():
        summary["calibration_delay"] = config["delay"]
    bank, error = _bank_from_records(records, tau, protocol)
    return evaluated, bank, error


def prepare_rescue(base, protocol, tau, noise_tau, seed, model, inherited, delay):
    """Fit only on declared calibration streams; persist before confirmation."""
    _validate(protocol)
    config = _configuration(base, protocol, delay)
    if ((inherited["tau"], inherited["noise_tau"], inherited["seed"]) != (tau, noise_tau, seed)
            or inherited["base_config_sha256"] != _digest(base)):
        raise ValueError("Inherited preparation does not match model cell or base configuration")
    variants = {name: deepcopy(inherited["variants"][name]) for name in INHERITED_VARIANTS}
    result = {"tau": tau, "noise_tau": noise_tau, "seed": seed, "delay": float(delay),
              "delay_cue": protocol["delay_cue"], "period": base["period"], "group": deepcopy(inherited["group"]),
              "status": "prepared", "variants": variants, "unavailable_controls": {},
              "base_config_sha256": _digest(base), "protocol_sha256": _digest(protocol),
              "inherited_preparation_sha256": _digest(inherited)}
    evaluated, bank, error = _run(config, protocol, tau, noise_tau, model,
                                  {"native": variants["native"]}, protocol["calibration_seeds"])
    result.update(calibration=evaluated, physical_rollouts=evaluated["physical_rollouts"])
    if error is not None:
        result["status"] = "calibration_censored"
        result["unavailable_controls"] = {name: error["reason"] for name in CONTROL_VARIANTS}
        evaluated["fixed_history"] = {"available": False, "failure": error, "variants": {}}
        return result
    raw = {name: _raw(scale_branches(model, setting["branch_scales"]), bank, config)
           for name, setting in variants.items()}
    native = _clipped(raw["native"], config, variants["native"])
    weak = _clipped(raw["joint_weak"], config, variants["joint_weak"])
    for name, source, target in (("gain_matched", "native", weak), ("weak_rescue", "joint_weak", native)):
        fit, unavailable = fit_response_gain(raw[source], target, native, bank["groups"], config["action_limit"],
                                             protocol["gain_bounds"], protocol["response_energy_floor"])
        if unavailable is not None:
            result["unavailable_controls"][name] = unavailable
            result["status"] = "controls_partly_unavailable"
        else:
            variants[name] = {"branch_scales": deepcopy(variants[source]["branch_scales"]), **fit,
                              "source_variant": source, "target_response_label": "joint_weak" if name == "gain_matched" else "native",
                              "calibration_delay": float(delay)}
    evaluated["fixed_history"] = _history(model, bank, config, protocol, variants)
    for name in result["unavailable_controls"]:
        evaluated["fixed_history"]["variants"][name] = None
    return result


def confirm_rescue(base, protocol, tau, noise_tau, seed, model, prepared, delay):
    """Evaluate every available frozen setting, preserving failures and nulls."""
    _validate(protocol)
    config = _configuration(base, protocol, delay)
    if ((prepared["tau"], prepared["noise_tau"], prepared["seed"], prepared["delay"]) != (tau, noise_tau, seed, delay)
            or prepared["base_config_sha256"] != _digest(base) or prepared["protocol_sha256"] != _digest(protocol)):
        raise ValueError("Prepared rescue does not match model cell, delay, base configuration or protocol")
    variants = deepcopy(prepared["variants"])
    result, bank, error = _run(config, protocol, tau, noise_tau, model, variants, protocol["confirmation_seeds"])
    result.update(tau=tau, noise_tau=noise_tau, seed=seed, delay=float(delay), delay_cue=protocol["delay_cue"],
                  period=base["period"], group=deepcopy(prepared["group"]), status=prepared["status"],
                  frozen_settings=variants, unavailable_controls=deepcopy(prepared["unavailable_controls"]),
                  base_config_sha256=_digest(base), protocol_sha256=_digest(protocol),
                  prepared_sha256=_digest(prepared),
                  fixed_history=({"available": False, "failure": error, "variants": {}} if error is not None
                                 else _history(model, bank, config, protocol, variants)))
    for name in prepared["unavailable_controls"]:
        result["summary"][name] = None
        result["fixed_history"]["variants"][name] = None
    return result
