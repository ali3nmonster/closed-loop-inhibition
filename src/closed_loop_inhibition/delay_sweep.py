"""Frozen-controller latency interventions at a fixed observation cadence.

Only physical command delivery changes. The explicit planned-delay feature is
held at its training value; all realized state and action history remains true.
Inherited intervention settings are never fitted or selected on these streams.
"""

from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np

from .collective_dynamics import _array_digest, _digest, _direction
from .collective_suppression import scale_branches
from .centered_calibration import weighted_mean
from .suppression_analysis import immediate_output_rms
from .timescale_diagnostics import _averages, _clipped, _raw, paired_recovery, response_summary
from .timescale_maps import NeuralPolicy, make_plant, metrics, run_episode


@dataclass
class FixedDelayPolicy(NeuralPolicy):
    """Hide physical latency changes only from the explicit future-delay cue."""

    delay_cue: float = .05

    def __call__(self, snapshot):
        if not np.isfinite(self.delay_cue) or self.delay_cue < 0:
            raise ValueError("delay_cue must be finite and nonnegative")
        return super().__call__(replace(snapshot, planned_apply_time=snapshot.time + self.delay_cue))


def clamp_delay_tokens(tokens, valid, tau, delay_cue):
    """Copy recorded true-delay tokens, changing valid rows of feature 7 only."""
    tokens, valid = np.asarray(tokens), np.asarray(valid)
    if (tokens.ndim != 3 or tokens.shape[:2] != valid.shape or tokens.shape[2] != 10
            or valid.dtype != np.bool_ or not np.isfinite(tau) or tau <= 0
            or not np.isfinite(delay_cue) or delay_cue < 0):
        raise ValueError("Delay clamping requires valid encoded histories and physical scales")
    result = tokens.copy()
    result[:, :, 7][valid] = delay_cue / tau
    return result


def timing_audit(run, period, delay):
    """Check actual dispatch and application clocks, including censored prefixes."""
    starts = np.asarray([job["start_time"] for job in run.jobs])
    intervals = np.diff(starts)
    complete_lags = np.asarray([job["complete_time"] - job["start_time"] for job in run.jobs])
    planned_lags = np.asarray([job["apply_time"] - job["start_time"] for job in run.jobs])
    applied = [action for action in run.applied_actions if action.job_id is not None]
    actual_lags = np.asarray([action.time - run.jobs[action.job_id]["start_time"] for action in applied])
    if (len(starts) and not np.allclose(starts, np.arange(len(starts)) * period, rtol=0., atol=2e-9)
            or not np.allclose(complete_lags, delay, rtol=0., atol=2e-9)
            or not np.allclose(planned_lags, delay, rtol=0., atol=2e-9)
            or not np.allclose(actual_lags, delay, rtol=0., atol=2e-9)):
        raise AssertionError("Physical dispatch or action timing differs from the frozen sweep protocol")
    if run.failure_reason is None:
        # The horizon need not be a multiple of the dispatch period.
        expected_jobs = int(np.floor((run.config.duration + 1e-10) / period)) + 1
        if len(starts) != expected_jobs:
            raise AssertionError("An uncensored rollout is missing scheduled dispatches")
        expected_applied = sum(job["apply_time"] <= run.config.duration + 1e-10 for job in run.jobs)
        if len(applied) != expected_applied:
            raise AssertionError("An uncensored rollout is missing delivered actions")
    def limits(values):
        return (float(np.min(values)), float(np.max(values))) if len(values) else (None, None)
    interval_min, interval_max = limits(intervals)
    lag_min, lag_max = limits(actual_lags)
    return {"passed": True, "dispatch_count": len(starts), "applied_count": len(applied),
            "undelivered_count": len(starts) - len(applied),
            "dispatch_interval_min": interval_min, "dispatch_interval_max": interval_max,
            "action_lag_min": lag_min, "action_lag_max": lag_max,
            "censored": run.failure_reason is not None, "failure_reason": run.failure_reason}


def _configuration(base, protocol, delay):
    if not np.isfinite(delay) or delay < 0:
        raise ValueError("Physical delay must be finite and nonnegative")
    cue = protocol["delay_cue"]
    if cue != base["delay"]:
        raise ValueError("Primary delay cue must equal the training delay")
    seeds = protocol["confirmation_seeds"]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Confirmation streams must be nonempty and distinct")
    probe = protocol["probe"]
    if (not 0 <= probe["pulse_onset"] < probe["duration"]
            or not 0 < probe["pulse_width"] <= probe["duration"] - probe["pulse_onset"]
            or not probe["pulse_amplitudes"]
            or any(not np.isfinite(a) or a == 0 for a in probe["pulse_amplitudes"])):
        raise ValueError("Invalid pulse definition")
    config = deepcopy(base)
    config["delay"] = float(delay)
    return config


def _bank_from_records(records, tau, protocol):
    blocks = {key: [] for key in ("pulse_tokens", "pulse_valid", "sham_tokens", "sham_valid",
                                  "groups", "amplitudes", "decision_times", "pulse_commands", "sham_commands")}
    probe = protocol["probe"]
    for row in records:
        if row["sham_censored"] or row["pulse_censored"]:
            return None, {"reason": "native_rollout_censored", "noise_seed": row["noise_seed"],
                          "amplitude": row["amplitude"]}
        pulse, sham = row["pulse"], row["sham"]
        times = pulse["decision_times"]
        if not np.array_equal(times, sham["decision_times"]):
            raise AssertionError("Native paired probe decision times differ")
        mask = (times >= probe["pulse_onset"] - 1e-10) & (times < probe["duration"] - 1e-10)
        if not mask.any():
            raise ValueError("The fixed-history assay has no post-pulse decisions")
        for arm, values in (("pulse", pulse), ("sham", sham)):
            valid = values["valid"][mask]
            blocks[f"{arm}_tokens"].append(clamp_delay_tokens(values["tokens"][mask], valid, tau, protocol["delay_cue"]))
            blocks[f"{arm}_valid"].append(valid)
            blocks[f"{arm}_commands"].append(values["commands"][mask])
        count = int(mask.sum())
        blocks["groups"].append(np.full(count, f"seed_{row['noise_seed']}_amplitude_{row['amplitude']:g}"))
        blocks["amplitudes"].append(np.full(count, row["amplitude"]))
        blocks["decision_times"].append(times[mask])
    arrays = {key: np.concatenate(values) for key, values in blocks.items()}
    return {"pulse": (arrays["pulse_tokens"], arrays["pulse_valid"]),
            "sham": (arrays["sham_tokens"], arrays["sham_valid"]),
            **{key: arrays[key] for key in ("groups", "amplitudes", "decision_times", "pulse_commands", "sham_commands")}}, None


def _fixed_history(model, bank, config, protocol, variants):
    outputs = {label: _clipped(_raw(scale_branches(model, setting["branch_scales"]), bank, config), config, setting)
               for label, setting in variants.items()}
    native = outputs["native"]
    weak = outputs.get("joint_weak")
    target = immediate_output_rms(*native, *weak, bank["groups"]) if weak is not None else None
    result = {"available": True, "failure": None, "bank_sha256": _array_digest(bank),
              "decision_rows": len(bank["groups"]), "probe_groups": len(np.unique(bank["groups"])),
              "native_replay_max_abs_error": max(float(np.max(np.abs(a - bank[f"{arm}_commands"])))
                                                  for a, arm in zip(native, ("pulse", "sham"))),
              "variants": {}}
    # Batched float32 replay can differ slightly from one-snapshot evaluation.
    result["native_replay_tolerance"] = 1e-6
    result["native_replay_passed"] = result["native_replay_max_abs_error"] <= result["native_replay_tolerance"]
    if not result["native_replay_passed"]:
        raise AssertionError("Recorded clamped histories do not reproduce native policy commands")
    for label, output in outputs.items():
        setting = variants[label]
        magnitude = immediate_output_rms(*native, *output, bank["groups"])
        relative_error = abs(magnitude - target) / target if target is not None and target > 1e-12 else None
        direction = (_direction(native, weak, output, bank["groups"], protocol["response_direction_epsilon"])
                     if weak is not None else {})
        reference = setting.get("achieved_rms")
        response = response_summary(native, output, bank["groups"], bank["amplitudes"], protocol)
        result["variants"][label] = {
            "response": response,
            "all_groups_baseline_valid": all(row["baseline_valid"] for row in response["per_group"]),
            "immediate_output_rms": magnitude, "calibration_immediate_output_rms": reference,
            "magnitude_change_from_calibration": magnitude - reference if reference is not None else None,
            "sham_command_mean_change": weighted_mean(output[1] - native[1], bank["groups"]),
            "pulse_command_mean_change": weighted_mean(output[0] - native[0], bank["groups"]),
            "weak_relative_magnitude_error": relative_error, **direction,
            "matched_at_calibration": setting.get("matched_and_direction"),
            "magnitude_matched_on_shifted_histories": relative_error <= protocol["match_tolerance"] if relative_error is not None else None,
            "matched_on_shifted_histories": bool(relative_error <= protocol["match_tolerance"] and direction.get("response_direction_concordant"))
                                              if relative_error is not None else None,
        }
    return result


def _evaluate_variants(config, protocol, tau, noise_tau, policies, settings, record_native):
    probe, streams = protocol["probe"], protocol["confirmation_seeds"]
    result = {"rollouts": [], "summary": {}, "timing": [], "physical_rollouts": 0}
    native_records = []
    for label, policy in policies.items():
        record = record_native and label == "native"
        shams, pulses, recoveries = [], [], []
        for noise_seed in streams:
            outcome = run_episode(config, tau, noise_tau, policy, noise_seed, duration=probe["duration"], record=record)
            sham, sham_probe = outcome if record else (outcome, None)
            sham_stats = metrics(config, sham, burn_in=probe["pulse_onset"])
            shams.append(sham_stats)
            result["physical_rollouts"] += 1
            result["timing"].append({"variant": label, "noise_seed": noise_seed, "amplitude": 0.,
                                     **timing_audit(sham, config["period"], config["delay"])})
            for amplitude in probe["pulse_amplitudes"]:
                outcome = run_episode(config, tau, noise_tau, policy, noise_seed, duration=probe["duration"], record=record,
                                      pulse={"onset": probe["pulse_onset"], "width": probe["pulse_width"], "amplitude": amplitude})
                pulse, pulse_probe = outcome if record else (outcome, None)
                pulse_stats = metrics(config, pulse, burn_in=probe["pulse_onset"])
                recovery = paired_recovery(pulse, sham, probe["pulse_onset"], amplitude)
                pulses.append(pulse_stats)
                recoveries.append(recovery)
                result["physical_rollouts"] += 1
                result["timing"].append({"variant": label, "noise_seed": noise_seed, "amplitude": amplitude,
                                         **timing_audit(pulse, config["period"], config["delay"])})
                result["rollouts"].append({"variant": label, "noise_seed": noise_seed, "amplitude": amplitude,
                                            "sham": sham_stats, "pulse": pulse_stats, "paired_recovery": recovery})
                if record:
                    native_records.append({"noise_seed": noise_seed, "amplitude": amplitude, "sham": sham_probe,
                                           "pulse": pulse_probe, "sham_censored": sham_stats["censored"],
                                           "pulse_censored": pulse_stats["censored"]})
        setting = settings[label]
        result["summary"][label] = {"sham": _averages(shams), "pulse": _averages(pulses),
                                     "paired_recovery": _averages(recoveries), "independent_noise_seeds": len(shams),
                                     "calibration_delay": protocol["delay_cue"],
                                     **{key: setting.get(key) for key in ("matched", "magnitude_matched", "direction_matched",
                                        "matched_and_direction", "response_direction_concordant", "command_change_cosine")},
                                     "match_relative_error": setting.get("relative_error")}
    native = result["summary"].get("native", result["summary"].get("zero_policy"))
    for summary in result["summary"].values():
        summary["delta_from_native"] = {
            arm: {key: value - native[arm][key] if value is not None and native[arm].get(key) is not None else None
                  for key, value in summary[arm].items()}
            for arm in ("sham", "pulse", "paired_recovery")}
    return result, native_records


def confirm_delay(base, protocol, tau, noise_tau, seed, model, prepared, delay):
    """Run one delay with exact inherited settings and fresh paired force tapes."""
    config = _configuration(base, protocol, delay)
    if ((prepared["tau"], prepared["noise_tau"], prepared["seed"]) != (tau, noise_tau, seed)
            or prepared["base_config_sha256"] != _digest(base)):
        raise ValueError("Inherited preparation does not match this model cell or base configuration")
    variants = deepcopy(prepared["variants"])
    if "native" not in variants:
        raise ValueError("Inherited preparation must include the native controller")
    plant = make_plant(config, tau)
    policies = {label: FixedDelayPolicy(scale_branches(model, setting["branch_scales"]), plant, config,
                                        offset=setting["offset"], gain=setting["gain"], center=setting["center"],
                                        delay_cue=protocol["delay_cue"])
                for label, setting in variants.items()}
    result, records = _evaluate_variants(config, protocol, tau, noise_tau, policies, variants, record_native=True)
    bank, error = _bank_from_records(records, tau, protocol)
    result.update(tau=tau, noise_tau=noise_tau, seed=seed, delay=float(delay), delay_cue=protocol["delay_cue"],
                  period=base["period"], group=deepcopy(prepared["group"]), status=prepared["status"],
                  frozen_settings=variants, unavailable_controls=deepcopy(prepared["unavailable_controls"]),
                  base_config_sha256=_digest(base), protocol_sha256=_digest(protocol),
                  fixed_history=({"available": False, "failure": error, "variants": {}} if error is not None
                                 else _fixed_history(model, bank, config, protocol, variants)))
    return result


def passive_cell(base, protocol, tau, noise_tau):
    """A shared zero-action reference; its plant trajectory is delay independent."""
    config = _configuration(base, protocol, base["delay"])
    result, _ = _evaluate_variants(config, protocol, tau, noise_tau, {"zero_policy": lambda snapshot: 0.},
                                   {"zero_policy": {}}, record_native=False)
    result.update(tau=tau, noise_tau=noise_tau, delay_independent=True, simulation_delay=base["delay"],
                  base_config_sha256=_digest(base), protocol_sha256=_digest(protocol))
    return result
