"""Causal, equilibrium-anchored scalar and history-kernel rescue.

Every correction is fixed at the declared calibration delay. The history
kernel acts only on observed normalized position, velocity, captured actions,
and the currently held action. Pending commands are part of the physical map,
never of the policy. A complete kernel restores the local command derivative
by construction; finite-trajectory confirmation is the scientific test.
"""

from copy import deepcopy
from dataclasses import replace
import math

import numpy as np
import torch

from .centered_calibration import weighted_mean
from .collective_dynamics import _array_digest, _digest
from .collective_suppression import scale_branches
from .delay_sweep import _bank_from_records, _configuration, _evaluate_variants, timing_audit
from .gain_rescue import _validate, waveform_diagnostics
from .loop_mechanism import EventCycleMap, impulse_analysis, linear_analysis
from .signals import PiecewiseConstant
from .timescale_diagnostics import response_summary
from .timescale_maps import encode, make_plant, make_timing
from .timing import simulate


VARIANTS = ("native", "joint_weak", "weak_equilibrium", "weak_scalar", "weak_kernel")
CORRECTIONS = VARIANTS[2:]


def feature_groups(history_count):
    """Disjoint observed-coordinate groups; indices run oldest to newest."""
    return {
        "older_position": list(range(0, 3 * (history_count - 1), 3)),
        "latest_position": [3 * (history_count - 1)],
        "older_velocity": list(range(1, 3 * (history_count - 1), 3)),
        "latest_velocity": [3 * (history_count - 1) + 1],
        "captured_action_history": list(range(2, 3 * history_count, 3)),
        "current_action": [3 * history_count],
    }


def observed_coordinates(tokens, valid, anchor):
    """Extract causal coordinates, with absent older deviations set to zero.

    During warmup, valid observations occupy a prefix in the padded model
    encoding. Their correction coefficients are the matching newest suffix of
    the mature history. This does not feed fabricated observations to the net.
    """
    tokens, valid = np.asarray(tokens), np.asarray(valid)
    anchor = np.asarray(anchor, dtype=float)
    count = (len(anchor) - 1) // 3
    if (anchor.ndim != 1 or len(anchor) != 3 * count + 1 or count < 1
            or not np.isfinite(anchor).all() or tokens.ndim != 3
            or tokens.shape[2] != 10 or valid.shape != tokens.shape[:2]
            or valid.dtype != np.bool_):
        raise ValueError("Expected finite observed anchor and padded token bank")
    lengths = valid.sum(axis=1)
    if (np.any(lengths < 1) or np.any(lengths > count)
            or not np.array_equal(valid, np.arange(valid.shape[1])[None] < lengths[:, None])
            or not np.isfinite(tokens[valid]).all()):
        raise ValueError("Each history needs a nonempty contiguous valid prefix")
    output = np.broadcast_to(anchor, (len(tokens), len(anchor))).copy()
    for n in np.unique(lengths):
        mask = lengths == n
        output[mask, 3 * (count - n):3 * count] = tokens[mask, :n][:, :, [0, 1, 4]].reshape(int(mask.sum()), 3 * n)
        output[mask, -1] = tokens[mask, n - 1, 8]
    return output


def _inherited_raw(raw, setting):
    return setting.get("center", 0.) + setting.get("gain", 1.) * (raw - setting.get("center", 0.)) + setting.get("offset", 0.)


def _correct(raw, coordinates, setting):
    correction = setting.get("correction")
    if not correction or correction.get("identity", False):
        return raw, np.zeros_like(raw)
    deviation = coordinates - np.asarray(correction["anchor_coordinates_float32"])
    kernel = deviation @ np.asarray(correction["kernel"])
    result = correction["anchor_position"] + correction["gain"] * (raw - correction["weak_anchor_raw_float32"]) + kernel
    return result, kernel


class KernelPolicy:
    """Float32 neural evaluation, float64 physical correction, clipping last."""

    def __init__(self, model, base, tau, setting, delay_cue=.05, record_diagnostics=True):
        self.config, self.setting = deepcopy(base), deepcopy(setting)
        self.plant = make_plant(base, tau)
        scales = setting.get("branch_scales", {})
        self.model = (scale_branches(model, scales) if scales else deepcopy(model)).eval()
        self.delay_cue = delay_cue
        self.record_diagnostics = record_diagnostics
        self.history_count = min(base["max_tokens"], round(base["history_seconds"] / base["period"]) + 1)
        self.records = []

    def __call__(self, snapshot):
        snapshot = replace(snapshot, planned_apply_time=snapshot.time + self.delay_cue)
        tokens, valid = encode(snapshot, self.plant, self.config)
        with torch.inference_mode():
            output = float(self.model(torch.from_numpy(tokens)[None], torch.from_numpy(valid)[None])[0])
        raw = float(_inherited_raw(output * self.config["output_scale"], self.setting))
        correction = self.setting.get("correction")
        coordinates = (observed_coordinates(tokens[None], valid[None], correction["anchor_coordinates_float32"])[0]
                       if correction else None)
        changed, kernel = _correct(np.asarray([raw]), coordinates[None] if coordinates is not None else None, self.setting)
        changed = float(changed[0])
        limit = self.config["action_limit"]
        command = float(np.clip(changed, -limit, limit))
        if self.record_diagnostics:
            if not self.records or abs(snapshot.time) < 1e-12:
                self.records.append({"count": 0, "clipped_count": 0, "warmup_count": 0, "squares": {}, "sums": {}, "max_abs": {}})
            record = self.records[-1]
            record["count"] += 1
            record["clipped_count"] += abs(changed) >= limit
            record["warmup_count"] += int(valid.sum()) < self.history_count
            pieces = {"raw_command_change": changed - raw,
                      "clipped_command_change": command - float(np.clip(raw, -limit, limit)),
                      "kernel": float(kernel[0]), "offset": 0., "gain": 0.}
            if correction and not correction.get("identity", False):
                pieces["offset"] = correction["anchor_position"] - correction["weak_anchor_raw_float32"]
                pieces["gain"] = (correction["gain"] - 1.) * (raw - correction["weak_anchor_raw_float32"])
                terms = np.asarray(correction["kernel"]) * (coordinates - np.asarray(correction["anchor_coordinates_float32"]))
                pieces.update({"kernel_" + name: float(terms[indices].sum())
                               for name, indices in feature_groups(self.history_count).items()})
            for name, value in pieces.items():
                record["squares"][name] = record["squares"].get(name, 0.) + value * value
                record["sums"][name] = record["sums"].get(name, 0.) + value
                record["max_abs"][name] = max(record["max_abs"].get(name, 0.), abs(value))
        return command

    def diagnostics(self, seeds, amplitudes):
        identities = [(seed, amplitude) for seed in seeds for amplitude in [0.] + list(amplitudes)]
        if len(identities) != len(self.records):
            raise AssertionError("Policy diagnostics do not match physical rollouts")
        rows = []
        for (seed, amplitude), record in zip(identities, self.records):
            count = record["count"]
            rows.append({"noise_seed": seed, "amplitude": amplitude, "decision_count": count,
                         "warmup_decision_count": record["warmup_count"],
                         "post_correction_clipping_fraction": record["clipped_count"] / count,
                         "components": {name: {"rms": math.sqrt(value / count),
                                                "mean": record["sums"][name] / count,
                                                "max_abs": record["max_abs"][name]}
                                        for name, value in record["squares"].items()}})
        return {"convention": "decision-weighted physical command moments on each policy own histories, including warmup; censored prefixes retained as diagnostics only",
                "rollouts": rows}


class KernelCycleMap(EventCycleMap):
    """Complete physical map; added feedback uses observed state coordinates."""

    def __init__(self, model, base, tau, delay, settings=None, delay_cue=.05):
        super().__init__(model, base, tau, delay, settings, delay_cue)
        self.correction = deepcopy(self.settings.get("correction"))
        if self.correction:
            dimension = 3 * self.history_count + 1
            for name in ("anchor_coordinates", "anchor_coordinates_float32", "kernel"):
                array = np.asarray(self.correction[name])
                if array.shape != (dimension,) or not np.isfinite(array).all():
                    raise ValueError("Correction must contain finite observed-coordinate vectors")
            if not all(np.isfinite(self.correction[name]) for name in
                       ("anchor_position", "gain", "weak_anchor_raw_float64", "weak_anchor_raw_float32")):
                raise ValueError("Correction scalars must be finite")
            if self.correction["gain"] < 0:
                raise ValueError("Correction gain must be nonnegative")

    def raw_command(self, state):
        raw = super().raw_command(state)
        correction = self.correction
        if not correction or correction.get("identity", False):
            return raw
        observed = self._state(state)[:3 * self.history_count + 1]
        anchor = torch.as_tensor(correction["anchor_coordinates"], dtype=torch.float64)
        kernel = torch.as_tensor(correction["kernel"], dtype=torch.float64)
        return (correction["anchor_position"]
                + correction["gain"] * (raw - correction["weak_anchor_raw_float64"])
                + kernel @ (observed - anchor))


def fit_scalar(source, target, groups, bounds, floor):
    """Positive equal-probe LS response gain; no new mean offset is fitted."""
    source, target = [tuple(np.asarray(v, dtype=float) for v in pair) for pair in (source, target)]
    if (any(v.ndim != 1 or len(v) != len(groups) or not np.isfinite(v).all() for v in source + target)
            or len(bounds) != 2 or not np.isfinite(bounds).all() or not 0 < bounds[0] < bounds[1]
            or not np.isfinite(floor) or floor <= 0):
        raise ValueError("Invalid gain inputs")
    response, desired = source[0] - source[1], target[0] - target[1]
    energy, target_energy = weighted_mean(response**2, groups), weighted_mean(desired**2, groups)
    if energy <= floor or target_energy <= floor:
        return None, "source_or_target_response_below_energy_floor"
    unconstrained = weighted_mean(response * desired, groups) / energy
    gain = float(np.clip(unconstrained, *bounds))
    return {"gain": gain, "unconstrained_gain": float(unconstrained),
            "gain_at_bound": bool(gain != unconstrained), "gain_bounds": list(bounds),
            "source_response_energy": energy, "target_response_energy": target_energy,
            "response_error_energy": weighted_mean((gain * response - desired)**2, groups),
            "objective": "equal_probe_preclip_weak_pulse_minus_sham_to_native_clipped_response_least_squares",
            "mean_offset_refit": False}, None


def _bank_outputs(model, bank, config, variants):
    raw, outputs = {}, {}
    for name, setting in variants.items():
        scales = setting.get("branch_scales", {})
        network = (scale_branches(model, scales) if scales else deepcopy(model)).eval()
        pairs, changed_pairs = [], []
        for arm in ("pulse", "sham"):
            tokens, valid = bank[arm]
            with torch.inference_mode():
                values = np.concatenate([network(torch.as_tensor(tokens[start:start + 512], dtype=torch.float32),
                                                torch.as_tensor(valid[start:start + 512], dtype=torch.bool))
                                         .numpy().reshape(-1) for start in range(0, len(tokens), 512)]).astype(float)
            original = _inherited_raw(values * config["output_scale"], setting)
            correction = setting.get("correction")
            coordinates = observed_coordinates(tokens, valid, correction["anchor_coordinates_float32"]) if correction else None
            changed, _ = _correct(original, coordinates, setting)
            pairs.append(changed)
            changed_pairs.append(np.clip(changed, -config["action_limit"], config["action_limit"]))
        raw[name], outputs[name] = tuple(pairs), tuple(changed_pairs)
    return raw, outputs


def _fixed_history(model, bank, config, protocol, variants, *, include_audit=False):
    raw, output = _bank_outputs(model, bank, config, variants)
    native, weak = output["native"], output["joint_weak"]
    replay_error = max(float(np.max(np.abs(value - bank[f"{arm}_commands"])))
                       for value, arm in zip(native, ("pulse", "sham")))
    if replay_error > 1e-6:
        raise AssertionError("Native fixed-history replay differs from recorded command")
    result = {"available": True, "failure": None, "bank_sha256": _array_digest(bank),
              "decision_rows": len(bank["groups"]), "probe_groups": len(np.unique(bank["groups"])),
              "native_replay_max_abs_error": replay_error, "native_replay_passed": True,
              "native_replay_tolerance": 1e-6, "variants": {}, "waveforms": []}
    for name, values in output.items():
        result["variants"][name] = {
            "target_response_label": "native",
            "response": response_summary(native, values, bank["groups"], bank["amplitudes"], protocol),
            "waveform": waveform_diagnostics(values, native, native, weak, bank["groups"], protocol["response_energy_floor"]),
            "clipping_fraction": {arm: weighted_mean((np.abs(value) >= config["action_limit"]).astype(float), bank["groups"])
                                  for arm, value in zip(("pulse", "sham"), raw[name])}}
    for group in np.unique(bank["groups"]):
        mask = bank["groups"] == group
        waveform = {"group": str(group), "amplitude": float(bank["amplitudes"][mask][0]),
                    "decision_times": bank["decision_times"][mask].tolist(),
                    "responses": {name: (pair[0][mask] - pair[1][mask]).tolist() for name, pair in output.items()}}
        if include_audit:
            waveform["raw_commands"] = {name: {arm: value[mask].tolist() for arm, value in zip(("pulse", "sham"), pair)}
                                         for name, pair in raw.items()}
            waveform["commands"] = {name: {arm: value[mask].tolist() for arm, value in zip(("pulse", "sham"), pair)}
                                    for name, pair in output.items()}
            correction = next((setting["correction"] for setting in variants.values() if "correction" in setting), None)
            if correction:
                waveform["observed_coordinates"] = {arm: observed_coordinates(tokens[mask], valid[mask], correction["anchor_coordinates_float32"]).tolist()
                                                      for arm, (tokens, valid) in ((arm, bank[arm]) for arm in ("pulse", "sham"))}
        result["waveforms"].append(waveform)
    return result


def _run(base, protocol, tau, noise_tau, model, variants, seeds, delay, duration=None):
    evaluation = deepcopy(protocol)
    evaluation["confirmation_seeds"] = list(seeds)
    if duration is not None:
        evaluation["probe"]["duration"] = duration
    config = _configuration(base, evaluation, delay)
    policies = {name: KernelPolicy(model, config, tau, setting, protocol["delay_cue"])
                for name, setting in variants.items()}
    result, records = _evaluate_variants(config, evaluation, tau, noise_tau, policies, variants, record_native=True)
    result["correction_on_own_histories"] = {
        name: policy.diagnostics(seeds, evaluation["probe"]["pulse_amplitudes"]) for name, policy in policies.items()}
    for summary in result["summary"].values():
        summary["calibration_delay"] = protocol["calibration_delay"]
    bank, error = _bank_from_records(records, tau, evaluation)
    return result, bank, error, config, evaluation


def _anchor(model, base, protocol, tau, variants):
    delay = protocol["calibration_delay"]
    native = KernelCycleMap(model, base, tau, delay, variants["native"], protocol["delay_cue"])
    weak = KernelCycleMap(model, base, tau, delay, variants["joint_weak"], protocol["delay_cue"])
    state, equilibrium = native.equilibrium()
    if not equilibrium["differentiable"] or equilibrium["clipped"] or equilibrium["state_residual_max"] > 1e-9:
        raise ValueError("Native anchor requires a resolved interior differentiable equilibrium")
    dimension = 3 * native.history_count + 1
    coordinates = state[:dimension]
    native_k = native.linearize(state)[3][:dimension]
    weak_state = weak.uniform_state(equilibrium["position"])
    weak_k = torch.autograd.functional.jacobian(weak.raw_command, torch.tensor(weak_state, dtype=torch.float64)).detach().numpy()[:dimension]
    with torch.no_grad():
        weak64 = float(weak.raw_command(weak_state))
        tokens, valid = weak.tokens(weak_state)
        # The original encoder performs separate float32 division/multiplication.
        # Build its canonical uniform-history tokens using those same stages.
        tokens32 = tokens.float().numpy()
        q, u = equilibrium["position"], equilibrium["action"]
        tokens32[:weak.history_count, 0] = np.float32(q) / np.float32(base["state_scale"])
        encoded_action = np.float32(u / base["action_limit"]) * np.float32(base["action_limit"] / base["output_scale"])
        tokens32[:weak.history_count, 4] = encoded_action
        tokens32[:weak.history_count, 8] = encoded_action
        weak32 = float(_inherited_raw(float(weak.float_model(torch.from_numpy(tokens32)[None], valid[None])[0]) * base["output_scale"], variants["joint_weak"]))
    coordinates32 = observed_coordinates(tokens32[None], valid.numpy()[None], coordinates)[0]
    return {"available": True, "position": equilibrium["position"], "coordinates": coordinates.tolist(),
            "coordinates_float32": coordinates32.tolist(),
            "equilibrium": equilibrium, "native_gradient": native_k.tolist(), "weak_gradient": weak_k.tolist(),
            "weak_anchor_raw_float64": weak64, "weak_anchor_raw_float32": weak32,
            "weak_inherited_anchor_would_clip": abs(weak64) >= base["action_limit"],
            "weak_gradient_convention": "derivative of inherited pre-clipping f_w; final clipping follows all corrections",
            "history_count": native.history_count, "observed_dimension": dimension,
            "current_action_age": native.current_action_age, "calibration_delay": delay,
            "coordinate_convention": "oldest-to-newest (q/state_scale, tau*v/state_scale, captured_action/output_scale), then held_action/output_scale; physical command derivative"}


def prepare_kernel(base, protocol, tau, noise_tau, seed, model, inherited):
    """Prepare once at the declared anchor delay using calibration tapes only."""
    _validate(protocol)
    if protocol["calibration_delay"] != base["delay"]:
        raise ValueError("Calibration delay must equal training delay")
    if ((inherited["tau"], inherited["noise_tau"], inherited["seed"]) != (tau, noise_tau, seed)
            or inherited["base_config_sha256"] != _digest(base)):
        raise ValueError("Inherited preparation does not match model cell or base configuration")
    variants = {name: deepcopy(inherited["variants"][name]) for name in VARIANTS[:2]}
    try:
        anchor = _anchor(model, base, protocol, tau, variants)
    except (ValueError, RuntimeError, np.linalg.LinAlgError, OverflowError) as error:
        anchor = {"available": False, "failure": str(error), "calibration_delay": protocol["calibration_delay"]}
    identity = not inherited.get("group")
    if identity and variants["native"] != variants["joint_weak"]:
        # Historical metadata may differ; compare actual policy parameters.
        keys = ("branch_scales", "offset", "gain", "center")
        if any(variants["native"].get(k, {} if k == "branch_scales" else 1. if k == "gain" else 0.) !=
               variants["joint_weak"].get(k, {} if k == "branch_scales" else 1. if k == "gain" else 0.) for k in keys):
            raise ValueError("An empty suppression group must implement an identity intervention")
    result = {"tau": tau, "noise_tau": noise_tau, "seed": seed, "delay": protocol["calibration_delay"],
              "calibration_delay": protocol["calibration_delay"], "delay_cue": protocol["delay_cue"], "period": base["period"],
              "group": deepcopy(inherited["group"]), "eligible": not identity,
              "status": "prepared", "variants": variants, "unavailable_controls": {}, "anchor": anchor,
              "base_config_sha256": _digest(base), "protocol_sha256": _digest(protocol),
              "inherited_preparation_sha256": _digest(inherited)}
    if not anchor["available"]:
        result["status"] = "anchor_unavailable"
        result["unavailable_controls"] = {name: anchor["failure"] for name in CORRECTIONS}
        evaluated, bank, error, config, evaluation = _run(base, protocol, tau, noise_tau, model,
                                                        {"native": variants["native"]}, protocol["calibration_seeds"], protocol["calibration_delay"])
        result.update(calibration=evaluated, physical_rollouts=evaluated["physical_rollouts"])
        evaluated["fixed_history"] = ({"available": False, "failure": error, "variants": {}} if error is not None else
                                      _fixed_history(model, bank, config, evaluation, variants, include_audit=True))
        for name in CORRECTIONS:
            evaluated["fixed_history"]["variants"][name] = None
        return result
    def corrected(gain, kernel):
        return {**deepcopy(variants["joint_weak"]), "calibration_delay": protocol["calibration_delay"],
                "correction": {"identity": identity, "anchor_position": anchor["position"],
                               "anchor_coordinates": deepcopy(anchor["coordinates"]),
                               "anchor_coordinates_float32": deepcopy(anchor["coordinates_float32"]),
                               "weak_anchor_raw_float32": anchor["weak_anchor_raw_float32"],
                               "weak_anchor_raw_float64": anchor["weak_anchor_raw_float64"],
                               "gain": float(gain), "kernel": np.asarray(kernel).tolist(),
                               "warmup": "available newest suffix only; missing history deviation zero"}}
    zero = np.zeros(anchor["observed_dimension"])
    variants["weak_equilibrium"] = corrected(1., zero)
    evaluated, bank, error, config, evaluation = _run(base, protocol, tau, noise_tau, model,
                                                    {"native": variants["native"]}, protocol["calibration_seeds"], protocol["calibration_delay"])
    result.update(calibration=evaluated, physical_rollouts=evaluated["physical_rollouts"])
    if error is not None:
        result["status"] = "calibration_censored"
        result["unavailable_controls"] = {name: error["reason"] for name in CORRECTIONS[1:]}
        evaluated["fixed_history"] = {"available": False, "failure": error, "variants": {}}
        return result
    raw, output = _bank_outputs(model, bank, config, {name: variants[name] for name in VARIANTS[:2]})
    fit, unavailable = fit_scalar(raw["joint_weak"], output["native"], bank["groups"],
                                   protocol["gain_bounds"], protocol["response_energy_floor"])
    if identity:
        fit, unavailable = {"gain": 1., "identity_intervention": True, "mean_offset_refit": False}, None
    anchor["scalar_fit"] = fit
    if unavailable:
        result["status"] = "controls_partly_unavailable"
        result["unavailable_controls"] = {name: unavailable for name in CORRECTIONS[1:]}
    else:
        gain = fit["gain"]
        delta = np.asarray(anchor["native_gradient"]) - gain * np.asarray(anchor["weak_gradient"])
        if identity:
            delta = zero.copy()
        variants["weak_scalar"] = corrected(gain, zero)
        variants["weak_kernel"] = corrected(gain, delta)
        anchor["kernel"] = delta.tolist()
        anchor["decomposition"] = {}
        for name, indices in feature_groups(anchor["history_count"]).items():
            native_k, weak_k = np.asarray(anchor["native_gradient"])[indices], np.asarray(anchor["weak_gradient"])[indices]
            anchor["decomposition"][name] = {"indices": indices, "native": native_k.tolist(),
                                            "weak": weak_k.tolist(), "scalar": (gain * weak_k).tolist(),
                                            "correction": delta[indices].tolist(),
                                            "correction_l2": float(np.linalg.norm(delta[indices]))}
    evaluated["fixed_history"] = _fixed_history(model, bank, config, evaluation, variants, include_audit=True)
    for name in result["unavailable_controls"]:
        evaluated["fixed_history"]["variants"][name] = None
    return result


def _validate_prepared(base, protocol, tau, noise_tau, seed, prepared, delay):
    _validate(protocol)
    if ((prepared["tau"], prepared["noise_tau"], prepared["seed"]) != (tau, noise_tau, seed)
            or prepared["base_config_sha256"] != _digest(base) or prepared["protocol_sha256"] != _digest(protocol)
            or prepared["calibration_delay"] != protocol["calibration_delay"]):
        raise ValueError("Prepared kernel does not match cell, base configuration or protocol")
    if delay not in protocol["delays"]:
        raise ValueError("Delay is outside the frozen confirmation grid")


def confirm_kernel(base, protocol, tau, noise_tau, seed, model, prepared, delay, duration=None):
    """Confirm frozen coefficients on fresh native and each policy own histories."""
    _validate_prepared(base, protocol, tau, noise_tau, seed, prepared, delay)
    duration = protocol["probe"]["duration"] if duration is None else duration
    if duration not in protocol.get("confirmation_durations", [protocol["probe"]["duration"]]):
        raise ValueError("Duration is outside the frozen confirmation horizons")
    variants = deepcopy(prepared["variants"])
    result, bank, error, config, evaluation = _run(base, protocol, tau, noise_tau, model, variants,
                                                 protocol["confirmation_seeds"], delay, duration)
    result.update(tau=tau, noise_tau=noise_tau, seed=seed, delay=float(delay), duration=float(duration),
                  delay_cue=protocol["delay_cue"], period=base["period"], calibration_delay=protocol["calibration_delay"],
                  group=deepcopy(prepared["group"]), eligible=prepared["eligible"], status=prepared["status"],
                  frozen_settings=variants, unavailable_controls=deepcopy(prepared["unavailable_controls"]),
                  base_config_sha256=_digest(base), protocol_sha256=_digest(protocol), prepared_sha256=_digest(prepared),
                  fixed_history=({"available": False, "failure": error, "variants": {}} if error is not None else
                                 _fixed_history(model, bank, config, evaluation, variants)))
    for name in prepared["unavailable_controls"]:
        result["summary"][name] = None
        result["fixed_history"]["variants"][name] = None
    return result


def simulator_parity(cycle, *, duration=2., tolerance=2e-6, command_tolerance=1e-6):
    """Check the corrected float32 physical policy against its smooth full map."""
    config = deepcopy(cycle.base)
    config["delay"] = cycle.delay
    duration = math.ceil(max(float(duration), cycle.base["history_seconds"] + cycle.delay + 3 * cycle.period) / cycle.period) * cycle.period
    disturbance = PiecewiseConstant(0., ((.75, .02), (.85, 0.)))
    policy = KernelPolicy(cycle.float_model, config, cycle.tau,
                          {**cycle.settings, "branch_scales": {}}, cycle.delay_cue, record_diagnostics=False)
    snapshots = []
    def capture(snapshot):
        snapshots.append(snapshot)
        return policy(snapshot)
    run = simulate(cycle.plant, capture, make_timing(config, duration, cycle.period),
                   initial_state=(.005, -.01), disturbance=disturbance)
    audit = timing_audit(run, cycle.period, cycle.delay)
    state_errors, command_errors = [], []
    for index, snapshot in enumerate(snapshots[:-1]):
        if snapshot.time < max(cycle.base["history_seconds"], cycle.delay + cycle.period) - 1e-10:
            continue
        state = cycle.from_snapshot(snapshot, run.jobs)
        following = cycle.from_snapshot(snapshots[index + 1], run.jobs)
        with torch.no_grad():
            predicted = cycle.step(state, disturbance.at(snapshot.time)).numpy()
            command = float(cycle.command(state))
        state_errors.append(float(np.max(np.abs(predicted - following))))
        command_errors.append(abs(command - run.jobs[index]["action"]))
    maximum_state, maximum_command = max(state_errors, default=None), max(command_errors, default=None)
    return {"passed": bool(state_errors and maximum_state <= tolerance and maximum_command <= command_tolerance and run.failure_reason is None),
            "physical_rollouts": 1, "duration": duration, "censored": run.failure_reason is not None,
            "failure_reason": run.failure_reason, "mature_steps_compared": len(state_errors),
            "state_error_max_normalized": maximum_state, "command_error_max_physical": maximum_command,
            "state_tolerance_normalized": tolerance, "command_tolerance_physical": command_tolerance, "timing": audit}


def analyze_kernel(base, protocol, tau, noise_tau, seed, model, prepared, delay):
    """Verify local matching and characterize each corrected complete loop."""
    _validate_prepared(base, protocol, tau, noise_tau, seed, prepared, delay)
    spec = protocol.get("mechanism", {})
    duration = spec.get("duration", 12.)
    frequencies = spec.get("frequencies", [.25, .5, 1., 2., 4., 8.])
    amplitudes = sorted(set(abs(float(a)) for a in spec.get("pulse_amplitudes", [-.02, -.0001, .0001, .02])))
    cycles = {name: KernelCycleMap(model, base, tau, delay, setting, protocol["delay_cue"])
              for name, setting in prepared["variants"].items()}
    native = cycles["native"]
    anchor_available = prepared["anchor"].get("available", True)
    if anchor_available and not math.isclose(native.current_action_age, prepared["anchor"]["current_action_age"], abs_tol=1e-12):
        raise ValueError("Frozen anchor requires the same command-age phase at both delays")
    if anchor_available:
        common = native.uniform_state(prepared["anchor"]["position"])
        native_a, _, _, native_k = native.linearize(common)
    output, identities, count = {}, {}, 0
    for name, cycle in cycles.items():
        parity = simulator_parity(cycle, duration=spec.get("parity_duration", 2.),
                                  tolerance=spec.get("parity_state_tolerance", 2e-6),
                                  command_tolerance=spec.get("parity_command_tolerance", 1e-6))
        count += 1
        if not parity["passed"]:
            raise AssertionError(f"Corrected full-loop map differs from physical simulator: {name}: {parity}")
        if anchor_available:
            state = cycle.uniform_state(prepared["anchor"]["position"])
            common_a, _, _, common_k = cycle.linearize(state)
            with torch.no_grad():
                command = float(cycle.command(state))
            identities[name] = {"command_error": command - prepared["anchor"]["position"],
                            "jacobian_error_max": float(np.max(np.abs(common_k - native_k))),
                            "full_map_error_max": float(np.max(np.abs(common_a - native_a))),
                            "dominant_radius_error": float(max(abs(np.linalg.eigvals(common_a))) - max(abs(np.linalg.eigvals(native_a)))),
                            "common_position": prepared["anchor"]["position"],
                            "interpretation": "local matching is an algebraic construction check, not independent mechanistic evidence",
                                "spectrum_caution": "near-zero poles of redundant nilpotent history coordinates can be ill-conditioned despite matrix equality"}
        if name in CORRECTIONS:
            equilibrium = state
            with torch.no_grad():
                raw = float(cycle.raw_command(state))
                residual = float(np.max(np.abs(cycle.step(state).numpy() - state)))
            operating = {"position": prepared["anchor"]["position"], "velocity": 0., "action": prepared["anchor"]["position"],
                         "converged": residual < spec.get("equilibrium_tolerance", 1e-9), "state_residual_max": residual,
                         "method": "known_native_anchor; other_roots_not_excluded", "raw_command": raw,
                         "clipped": abs(raw) > cycle.action_limit,
                         "distance_to_clipping_kink": abs(abs(raw) - cycle.action_limit),
                         "differentiable": abs(abs(raw) - cycle.action_limit) > 1e-8}
        else:
            equilibrium, operating = cycle.equilibrium()
        a, b, c, k = cycle.linearize(equilibrium)
        linear = linear_analysis(a, b, c, cycle.period, frequencies, round(duration / cycle.period))
        impulses = impulse_analysis(cycle, equilibrium, a, b, c, amplitudes=amplitudes,
                                    width=spec.get("pulse_width", .1), duration=duration,
                                    verification_amplitude=spec.get("verification_amplitude", 1e-6),
                                    settling_fraction=spec.get("settling_fraction", .02))
        issues = []
        if operating["state_residual_max"] > spec.get("equilibrium_tolerance", 1e-9):
            issues.append("equilibrium_not_resolved")
        if not operating["differentiable"]:
            issues.append("equilibrium_at_clipping_kink")
        if (impulses["verification_relative_error_max"] is None or
                impulses["verification_relative_error_max"] > spec.get("linearization_relative_tolerance", .01)):
            issues.append("tiny_pulse_linear_prediction_failed")
        if name == "weak_kernel" and (identities[name]["jacobian_error_max"] > 1e-9 or identities[name]["full_map_error_max"] > 1e-9):
            issues.append("constructed_local_kernel_identity_failed")
        output[name] = {"status": "complete" if not issues else "numerical_failure", "numerical_issues": issues,
                        "dimension": cycle.dimension, "history_count": cycle.history_count,
                        "pending_command_count": cycle.pending_count, "current_action_age": cycle.current_action_age,
                        "settings": deepcopy(cycle.settings), "equilibrium": operating, "equilibrium_state": equilibrium.tolist(),
                        "state_jacobian": a.tolist(), "force_input_jacobian": b.tolist(), "position_readout": c.tolist(),
                        "command_jacobian": k.tolist(), "linear": linear, "impulses": impulses, "simulator_parity": parity}
    return {"tau": tau, "noise_tau": noise_tau, "seed": seed, "delay": delay, "period": base["period"],
            "delay_cue": protocol["delay_cue"], "calibration_delay": protocol["calibration_delay"],
            "status": "complete" if all(r["status"] == "complete" for r in output.values()) else "numerical_failure",
            "physical_rollouts": count, "variants": output, "anchor_identity": identities,
            "anchor_available": anchor_available, "unavailable_controls": deepcopy(prepared["unavailable_controls"]),
            "prepared_sha256": _digest(prepared), "base_config_sha256": _digest(base), "protocol_sha256": _digest(protocol)}
