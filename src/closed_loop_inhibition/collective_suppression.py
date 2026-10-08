"""Common-history survey of suppressive effects in residual-output branches.

The declared partition contains projected attention-head contributions and
complete MLP residual outputs. Attention output bias is shared and left fixed;
the MLP gate includes ``linear2.bias``. These are nonoverlapping parameter gates,
not independent computations or a decomposition of all inhibition in a model.
Input embeddings, normalization, skip paths and the final readout remain fixed.
"""

from collections.abc import Mapping
from copy import deepcopy
from numbers import Real

import numpy as np
import torch

from .neural import CausalTransformer
from .timescale_maps import make_plant, run_episode, teacher_policy


def branches(model):
    """Stable branch IDs in layer order; the frozen model has ten branches."""
    if not isinstance(model, CausalTransformer):
        raise ValueError("branch gates require a CausalTransformer")
    return [branch for layer_index, layer in enumerate(model.layers)
            for branch in ([f"L{layer_index}H{head}" for head in range(layer.self_attn.num_heads)]
                           + [f"L{layer_index}MLP"])]


def scale_branches(model, scales):
    """Clone and gate disjoint output parameters; unspecified gates equal one.

    An attention gate scales only the corresponding ``out_proj`` columns. An
    MLP gate scales both ``linear2.weight`` and ``linear2.bias``, multiplying its
    entire residual contribution after the native nonlinearity. Native forward
    computation, causal masks and all other parameters remain unchanged.
    """
    identifiers = branches(model)
    if not isinstance(scales, Mapping) or set(scales) - set(identifiers):
        raise ValueError("scales must map known branch IDs to finite nonnegative numbers")
    for value in scales.values():
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
            raise ValueError("branch scales must be finite nonnegative real numbers")
    changed = deepcopy(model)
    changed.eval()
    changed.requires_grad_(False)
    with torch.no_grad():
        for index, layer in enumerate(changed.layers):
            for head in range(layer.self_attn.num_heads):
                scale = float(scales.get(f"L{index}H{head}", 1.0))
                if scale != 1:
                    width = layer.self_attn.head_dim
                    layer.self_attn.out_proj.weight[:, head * width:(head + 1) * width].mul_(scale)
            scale = float(scales.get(f"L{index}MLP", 1.0))
            if scale != 1:
                layer.linear2.weight.mul_(scale)
                if layer.linear2.bias is not None:
                    layer.linear2.bias.mul_(scale)
    if any(not bool(torch.isfinite(parameter).all()) for parameter in changed.parameters()):
        raise ValueError("branch scale produces nonfinite model parameters")
    return changed


def _validate_protocol(protocol):
    names = [name for name in ("discovery_seeds", "calibration_seeds", "confirmation_seeds")
             if name in protocol]
    streams = [list(protocol[name]) for name in names]
    if any(not values or len(set(values)) != len(values) for values in streams):
        raise ValueError("diagnostic seed streams must be nonempty without duplicate seeds")
    if any(set(streams[a]) & set(streams[b]) for a in range(len(streams)) for b in range(a + 1, len(streams))):
        raise ValueError("discovery, calibration and confirmation seed streams must be disjoint")
    if not 0 < protocol["weak_scale"] < 1 < protocol["strong_scale"]:
        raise ValueError("weak and strong scales must straddle one")
    if protocol["baseline_epsilon"] <= 0 or protocol["min_fractional_increase"] < 0:
        raise ValueError("response floor must be positive and effect threshold nonnegative")
    if not 0 <= protocol["min_positive_fraction"] <= 1:
        raise ValueError("positive fraction threshold must lie in [0,1]")


def build_bank(base_config, protocol, tau, seeds, common=True, *, noise_tau=None, policy=None):
    """Build fixed paired teacher histories, shared across checkpoints.

    ``common=True`` uses the standardized background specified by ``probe``.
    The optional noncommon path uses the original physical noise amplitude and
    requires an explicit noise timescale. A supplied policy can record native
    histories for separately labeled calibration; the default is the teacher.
    Banks contain ordinary NumPy arrays so callers can persist and hash them.
    """
    _validate_protocol(protocol)
    probe = protocol["probe"]
    config = deepcopy(base_config)
    if common:
        config["noise_std"] = probe["background_std"]
        noise_tau = probe["background_tau"]
    elif noise_tau is None:
        raise ValueError("noncommon banks require an explicit noise_tau")
    if policy is None:
        policy = teacher_policy(config, make_plant(config, tau))
    blocks = {"pulse_tokens": [], "pulse_valid": [], "sham_tokens": [], "sham_valid": [],
              "groups": [], "amplitudes": [], "decision_times": [], "noise_seeds": []}
    for noise_seed in seeds:
        sham, sham_probe = run_episode(config, tau, noise_tau, policy, noise_seed,
                                       duration=probe["duration"], record=True)
        if sham.failure_reason is not None or sham.samples[-1]["time"] < probe["duration"] - 1e-10:
            raise ValueError("common-history sham rollout failed or was censored")
        for amplitude in probe["amplitudes"]:
            if not np.isfinite(amplitude) or amplitude == 0:
                raise ValueError("probe amplitudes must be finite and nonzero")
            pulse, pulse_probe = run_episode(
                config, tau, noise_tau, policy, noise_seed, duration=probe["duration"],
                pulse={"onset": probe["onset"], "width": probe["width"], "amplitude": amplitude},
                record=True)
            if pulse.failure_reason is not None or pulse.samples[-1]["time"] < probe["duration"] - 1e-10:
                raise ValueError("common-history pulse rollout failed or was censored")
            if not np.array_equal(pulse_probe["decision_times"], sham_probe["decision_times"]):
                raise ValueError("pulse and sham decision times differ")
            times = pulse_probe["decision_times"]
            mask = (times >= probe["onset"] - 1e-10) & (times < probe["duration"] - 1e-10)
            count = int(mask.sum())
            if not count:
                raise ValueError("probe window contains no decisions")
            for arm, records in (("pulse", pulse_probe), ("sham", sham_probe)):
                for field in ("tokens", "valid"):
                    blocks[f"{arm}_{field}"].append(records[field][mask])
            blocks["groups"].extend([f"seed_{noise_seed}_amplitude_{amplitude:g}"] * count)
            blocks["amplitudes"].extend([amplitude] * count)
            blocks["decision_times"].extend(times[mask].tolist())
            blocks["noise_seeds"].extend([noise_seed] * count)
    if not blocks["groups"]:
        raise ValueError("at least one seed and amplitude are required")
    return {"pulse": (np.concatenate(blocks["pulse_tokens"]), np.concatenate(blocks["pulse_valid"])),
            "sham": (np.concatenate(blocks["sham_tokens"]), np.concatenate(blocks["sham_valid"])),
            **{name: np.asarray(blocks[name])
               for name in ("groups", "amplitudes", "decision_times", "noise_seeds")}}


def bank_metadata(bank):
    """Small array inventory; callers persist the bank and supply its SHA."""
    return {"decision_rows": len(bank["groups"]), "scenario_count": len(np.unique(bank["groups"])),
            "noise_seeds": np.unique(bank["noise_seeds"]).astype(int).tolist(),
            "amplitudes": np.unique(bank["amplitudes"]).astype(float).tolist(),
            "first_decision_time": float(np.min(bank["decision_times"])),
            "last_decision_time": float(np.max(bank["decision_times"]))}


def predict_bank(model, bank, base_config, *, scales=None):
    """Raw physical pulse and sham commands, without clipping or centering."""
    evaluated = scale_branches(model, scales) if scales is not None else model
    evaluated.eval()
    device = next(evaluated.parameters()).device
    outputs = []
    with torch.inference_mode():
        for arm in ("pulse", "sham"):
            tokens, valid = bank[arm]
            blocks = [evaluated(torch.as_tensor(tokens[start:start + 512], dtype=torch.float32, device=device),
                                torch.as_tensor(valid[start:start + 512], dtype=torch.bool, device=device))
                      .detach().cpu().numpy().reshape(-1)
                      for start in range(0, len(tokens), 512)]
            outputs.append(np.concatenate(blocks).astype(float) * base_config["output_scale"])
    return tuple(outputs)


def _commands(model, bank, base_config, scales=None):
    return tuple(np.clip(values, -base_config["action_limit"], base_config["action_limit"])
                 for values in predict_bank(model, bank, base_config, scales=scales))


def response_effect(native, changed, bank, protocol):
    """Per-scenario RMS changes; every physical seed/amplitude has equal weight.

    Ratios and primary aggregate effects are unavailable if any native scenario
    is below the physical response floor. Raw responses and absolute effects
    remain visible, avoiding an artificial zero for an unresponsive checkpoint.
    """
    groups, amplitudes = np.asarray(bank["groups"]), np.asarray(bank["amplitudes"])
    arrays = [np.asarray(values, dtype=float) for values in (*native, *changed)]
    if any(values.shape != groups.shape or not np.isfinite(values).all() for values in arrays):
        raise ValueError("command arrays must be finite vectors matching bank groups")
    if groups.ndim != 1 or not len(groups) or amplitudes.shape != groups.shape:
        raise ValueError("bank groups and amplitudes must be nonempty equal-length vectors")
    native_difference, changed_difference = arrays[0] - arrays[1], arrays[2] - arrays[3]
    rows = []
    for group in np.unique(groups):
        mask = groups == group
        amplitude = float(amplitudes[mask][0])
        if not np.isfinite(amplitude) or amplitude == 0 or not np.all(amplitudes[mask] == amplitude):
            raise ValueError("each group requires a fixed finite nonzero amplitude")
        baseline = float(np.sqrt(np.mean(native_difference[mask] ** 2)))
        modified = float(np.sqrt(np.mean(changed_difference[mask] ** 2)))
        valid = baseline > protocol["baseline_epsilon"]
        rows.append({"group": str(group), "amplitude": amplitude,
                     "native_response_rms": baseline, "changed_response_rms": modified,
                     "normalized_native_response_rms": baseline / abs(amplitude),
                     "normalized_changed_response_rms": modified / abs(amplitude),
                     "absolute_change": modified - baseline,
                     "normalized_absolute_change": (modified - baseline) / abs(amplitude),
                     "fractional_change": modified / baseline - 1.0 if valid else None,
                     "baseline_valid": bool(valid)})
    classifiable = all(row["baseline_valid"] for row in rows)
    fractional = [row["fractional_change"] for row in rows] if classifiable else None
    median = float(np.median(fractional)) if classifiable else None
    positive = float(np.mean(np.asarray(fractional) > 0)) if classifiable else None
    eligible = classifiable and median > protocol["min_fractional_increase"]
    eligible = eligible and positive >= protocol["min_positive_fraction"]
    return {"per_group": rows, "classifiable": classifiable,
            "native_response_rms": float(np.sqrt(np.mean([row["native_response_rms"] ** 2 for row in rows]))),
            "normalized_native_response_rms": float(np.sqrt(np.mean([
                row["normalized_native_response_rms"] ** 2 for row in rows]))),
            "changed_response_rms": float(np.sqrt(np.mean([row["changed_response_rms"] ** 2 for row in rows]))),
            "median_fractional_change": median,
            "median_absolute_change": float(np.median([row["absolute_change"] for row in rows])),
            "median_normalized_absolute_change": float(np.median([
                row["normalized_absolute_change"] for row in rows])),
            "positive_fraction": positive, "eligible": bool(eligible)}


def _check_models(models):
    if set(models) != {"initial", "epoch_025", "selected"}:
        raise ValueError("models must contain initial, epoch_025 and selected checkpoints")
    identifiers = branches(models["selected"])
    if any(branches(model) != identifiers for model in models.values()):
        raise ValueError("checkpoint branch partitions differ")
    return identifiers


def discover_model(base_config, protocol, tau, noise_tau, seed, models, bank=None):
    """Select a suppressive group independently at each checkpoint, discovery only."""
    _validate_protocol(protocol)
    identifiers = _check_models(models)
    if bank is None:
        bank = build_bank(base_config, protocol, tau, protocol["discovery_seeds"])
    if set(np.unique(bank["noise_seeds"])) != set(protocol["discovery_seeds"]):
        raise ValueError("discovery bank does not use the declared discovery seeds")
    checkpoints = {}
    for name, model in models.items():
        native = _commands(model, bank, base_config)
        rows = [{"branch": branch, "weak": response_effect(
            native, _commands(model, bank, base_config, {branch: protocol["weak_scale"]}), bank, protocol)}
            for branch in identifiers]
        classifiable = rows[0]["weak"]["classifiable"]
        selected = [row["branch"] for row in rows if row["weak"]["eligible"]] if classifiable else None
        checkpoints[name] = {"classifiable": classifiable, "selected": selected,
                              "eligible_count": len(selected) if selected is not None else None,
                              "eligible_fraction": len(selected) / len(identifiers) if selected is not None else None,
                              "branches": rows}
    return {"tau": tau, "noise_tau": noise_tau, "seed": seed, "branch_ids": identifiers,
            "bank_metadata": bank_metadata(bank), "checkpoints": checkpoints}


def central_sensitivity(weak, strong, protocol):
    """Positive means weakening increases response relative to strengthening."""
    if [row["group"] for row in weak["per_group"]] != [row["group"] for row in strong["per_group"]]:
        raise ValueError("weak and strong effects must share the same scenarios")
    width = protocol["strong_scale"] - protocol["weak_scale"]
    rows = []
    for lower, upper in zip(weak["per_group"], strong["per_group"]):
        if lower["native_response_rms"] != upper["native_response_rms"]:
            raise ValueError("central sensitivities require identical native responses")
        difference = (lower["changed_response_rms"] - upper["changed_response_rms"]) / width
        rows.append({"group": lower["group"], "physical": difference,
                     "amplitude_normalized": difference / abs(lower["amplitude"]),
                     "fractional": difference / lower["native_response_rms"] if lower["baseline_valid"] else None})
    valid = weak["classifiable"] and strong["classifiable"]
    return {"per_group": rows, "classifiable": valid,
            "median_fractional": float(np.median([row["fractional"] for row in rows])) if valid else None,
            "median_physical": float(np.median([row["physical"] for row in rows])),
            "median_amplitude_normalized": float(np.median([row["amplitude_normalized"] for row in rows]))}


def aggregate_sensitivity(rows):
    """Sums across the declared gates; no claim of biological E/I balance."""
    classifiable = all(row["central_sensitivity"]["classifiable"] for row in rows)
    summary = {"classifiable": classifiable, "branch_count": len(rows)}
    for label, field in (("fractional", "median_fractional"), ("physical", "median_physical"),
                         ("amplitude_normalized", "median_amplitude_normalized")):
        values = [row["central_sensitivity"][field] for row in rows]
        if label == "fractional" and not classifiable:
            summary[label] = {"positive_sum": None, "negative_magnitude_sum": None, "signed_sum": None}
        else:
            summary[label] = {"positive_sum": float(sum(max(value, 0.0) for value in values)),
                              "negative_magnitude_sum": float(sum(max(-value, 0.0) for value in values)),
                              "signed_sum": float(sum(values))}
    return summary


def nonadditivity(joint, individual, selected):
    """Per-scenario joint delta minus summed single-gate deltas, then median."""
    rows = []
    by_branch = {row["branch"]: {item["group"]: item for item in row["weak"]["per_group"]}
                 for row in individual}
    if len(set(selected)) != len(selected) or set(selected) - set(by_branch):
        raise ValueError("selected branches must be unique and have individual effects")
    for group in joint["per_group"]:
        if any(group["group"] not in by_branch[branch] or
               by_branch[branch][group["group"]]["native_response_rms"] != group["native_response_rms"]
               for branch in selected):
            raise ValueError("nonadditivity requires identical groups and native responses")
        summed = sum(by_branch[branch][group["group"]]["absolute_change"] for branch in selected)
        difference = group["absolute_change"] - summed
        rows.append({"group": group["group"], "joint_absolute_change": group["absolute_change"],
                     "summed_single_absolute_change": summed, "absolute_nonadditivity": difference,
                     "normalized_nonadditivity": difference / abs(group["amplitude"]),
                     "fractional_nonadditivity": difference / group["native_response_rms"]
                     if group["baseline_valid"] else None})
    return {"per_group": rows, "classifiable": joint["classifiable"],
            "median_absolute_nonadditivity": float(np.median([row["absolute_nonadditivity"] for row in rows])),
            "median_normalized_nonadditivity": float(np.median([row["normalized_nonadditivity"] for row in rows])),
            "median_fractional_nonadditivity": float(np.median([row["fractional_nonadditivity"] for row in rows]))
            if joint["classifiable"] else None}


def _joint_curve(model, selected, native, bank, base_config, protocol):
    if selected is None:
        return None
    scales = list(dict.fromkeys([*protocol["joint_weak_scales"], protocol["weak_scale"], protocol["strong_scale"]]))
    return [{"scale": float(scale), "selected": list(selected), "group_size": len(selected),
             "empty_group": not selected,
             "response": response_effect(native, _commands(
                 model, bank, base_config, {branch: scale for branch in selected}), bank, protocol)}
            for scale in scales]


def confirm_model(base_config, protocol, tau, noise_tau, seed, models, discovery, bank=None):
    """Evaluate all branch signs and frozen groups on fresh common histories."""
    _validate_protocol(protocol)
    identifiers = _check_models(models)
    if (discovery["tau"], discovery["noise_tau"], discovery["seed"]) != (tau, noise_tau, seed):
        raise ValueError("discovery record belongs to a different trained model")
    if identifiers != discovery["branch_ids"]:
        raise ValueError("discovery branch partition differs")
    if bank is None:
        bank = build_bank(base_config, protocol, tau, protocol["confirmation_seeds"])
    if set(np.unique(bank["noise_seeds"])) != set(protocol["confirmation_seeds"]):
        raise ValueError("confirmation bank does not use the declared confirmation seeds")
    checkpoints, secondary = {}, {}
    trained_group = discovery["checkpoints"]["selected"]["selected"]
    for name, model in models.items():
        native = _commands(model, bank, base_config)
        rows = []
        for branch in identifiers:
            effects = {label: response_effect(native, _commands(model, bank, base_config,
                                                               {branch: protocol[f"{label}_scale"]}),
                                                bank, protocol) for label in ("weak", "strong")}
            rows.append({"branch": branch, **effects,
                         "central_sensitivity": central_sensitivity(effects["weak"], effects["strong"], protocol)})
        selected = discovery["checkpoints"][name]["selected"]
        joint = _joint_curve(model, selected, native, bank, base_config, protocol)
        joint_weak = next(row["response"] for row in joint if row["scale"] == protocol["weak_scale"]) if joint is not None else None
        checkpoints[name] = {"selected": selected, "native_response": response_effect(native, native, bank, protocol),
                              "branches": rows, "aggregate_sensitivity": aggregate_sensitivity(rows),
                              "joint": joint,
                              "nonadditivity_weak": nonadditivity(joint_weak, rows, selected)
                              if joint is not None else None}
        secondary[name] = {"joint": _joint_curve(model, trained_group, native, bank, base_config, protocol)}
    return {"tau": tau, "noise_tau": noise_tau, "seed": seed, "branch_ids": identifiers,
            "bank_metadata": bank_metadata(bank), "checkpoints": checkpoints,
            "trained_group_across_checkpoints": {
                "selected": trained_group,
                "selection_note": "Secondary: branches selected using trained-checkpoint discovery; learning contrast is conditioned on that selection.",
                "checkpoints": secondary}}


def analyze_model(base_config, protocol, tau, noise_tau, seed, models):
    """Convenience wrapper; staged callers should persist discovery themselves."""
    discovery = discover_model(base_config, protocol, tau, noise_tau, seed, models)
    confirmation = confirm_model(base_config, protocol, tau, noise_tau, seed, models, discovery)
    return {"discovery": discovery, "confirmation": confirmation}
