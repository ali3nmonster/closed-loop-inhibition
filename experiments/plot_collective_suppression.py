#!/usr/bin/env python3
"""Strict aggregation and plots for the ten-branch collective suppression assay.

Plotting is separate from frozen scientific execution. Every mean requires all
members of its declared population; missing results are never replaced by zero.
Checkpoint-specific discovery groups and fixed-trained-group contrasts remain
separate. Matched-control comparisons pair the same model in both arms.
"""

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ("initial", "epoch_025", "selected")
LABELS = {"initial": "Initialization", "epoch_025": "Epoch 25", "selected": "Selected checkpoint"}
matplotlib.rcParams["svg.hashsalt"] = "collective_suppression_v1"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    return isinstance(value, (int, float, bool, np.number)) and bool(np.isfinite(value))


def stats(values):
    values = list(values)
    valid = [float(value) for value in values if number(value)]
    complete = bool(values) and len(values) == len(valid)
    return {"mean": float(np.mean(valid)) if complete else None,
            "min": float(np.min(valid)) if complete else None,
            "max": float(np.max(valid)) if complete else None,
            "range": float(np.ptp(valid)) if complete else None,
            "n": len(values), "valid_n": len(valid)}


def circular_stats(values):
    """Circular degrees with min/max expressed around the circular mean."""
    ordinary = stats(values)
    if ordinary["mean"] is None:
        return ordinary
    values = np.asarray(list(values), dtype=float)
    vector = np.mean(np.exp(1j * np.deg2rad(values)))
    if abs(vector) < 1e-8:
        return {**ordinary, "mean": None, "min": None, "max": None, "range": None}
    center = float(np.rad2deg(np.angle(vector)))
    local = center + (values - center + 180.) % 360. - 180.
    return {**ordinary, "mean": center, "min": float(local.min()),
            "max": float(local.max()), "range": float(np.ptp(local))}


def get(record, *keys):
    for key in keys:
        if not isinstance(record, dict):
            return None
        record = record.get(key)
    return record


def scale(value, factor=100.):
    return float(value * factor) if number(value) else None


def delta(first, second):
    return float(first - second) if number(first) and number(second) else None


def percent(first, second):
    return float((first / second - 1.) * 100.) if number(first) and number(second) and second > 0 else None


def key(record):
    return record["tau"], record["noise_tau"], int(record["seed"])


def indexed(directory, expected):
    records = {}
    for path in sorted(directory.glob("*.json")):
        record = read_json(path)
        if key(record) in records:
            raise ValueError(f"Duplicate record: {path}")
        records[key(record)] = record
    if set(records) != set(expected):
        raise ValueError(f"Incomplete collection {directory}: {len(records)} of {len(expected)} models")
    return records


def joint_at(checkpoint, strength):
    rows = checkpoint.get("joint")
    if rows is None:
        return None
    selected = [row for row in rows if abs(row["scale"] - strength) < 1e-10]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one joint scale {strength}")
    return selected[0]["response"]


def own_loop_check(record, protocol):
    """Validate independent sham counts and summaries before plotting effects."""
    for variant, summary in record["summary"].items():
        rows = [row for row in record["rollouts"] if row["variant"] == variant]
        seeds = sorted({row["noise_seed"] for row in rows})
        expected_seeds = set(protocol["dynamics"]["confirmation_seeds"])
        amplitudes = protocol["dynamics"]["pulse_amplitudes"]
        if set(seeds) != expected_seeds or len(rows) != len(expected_seeds) * len(amplitudes):
            raise ValueError("Own-loop record has incomplete or duplicate seed/amplitude trials")
        if summary["independent_noise_seeds"] != len(seeds):
            raise ValueError("Own-loop summary overcounts independent noise tapes")
        unique = []
        for seed in seeds:
            if sorted(row["amplitude"] for row in rows if row["noise_seed"] == seed) != sorted(amplitudes):
                raise ValueError("Own-loop record has incorrect pulse-amplitude coverage")
            shams = [row["sham"] for row in rows if row["noise_seed"] == seed]
            if any(sham != shams[0] for sham in shams):
                raise ValueError("Repeated pulse arms disagree about their shared sham")
            unique.append(shams[0])
        for field in ("position_rms", "action_rms", "position_mean", "action_mean", "task_failure", "censored"):
            observed = stats(row[field] for row in unique)["mean"]
            reported = summary["sham"][field]
            if ((observed is None) != (reported is None)
                    or observed is not None and not np.isclose(observed, reported, rtol=1e-12, atol=1e-15)):
                raise ValueError(f"Wrong independent-tape mean: {variant}/{field}")


def model_rows(base, discovery, common, prepared, confirmed, weak_scale, protocol):
    rows, doses = [], []
    for identity, dis in discovery.items():
        conf = common[identity]
        cal = prepared[identity] if prepared is not None else None
        own = confirmed[identity] if confirmed is not None else None
        row = dict(zip(("tau", "noise_tau", "seed"), identity))
        for checkpoint in CHECKPOINTS:
            selected = dis["checkpoints"][checkpoint]["selected"]
            result = conf["checkpoints"][checkpoint]
            if result["selected"] != selected:
                raise ValueError("Confirmation changed a discovery-selected group")
            group_size = len(selected) if selected is not None else None
            row[f"{checkpoint}_group_size"] = group_size
            row[f"{checkpoint}_group_percent"] = scale(group_size, 100. / len(dis["branch_ids"]))
            row[f"{checkpoint}_mlp_count"] = sum(branch.endswith("MLP") for branch in selected) if selected is not None else None
            row[f"{checkpoint}_confirmed_single_percent"] = (
                100. * sum(branch["weak"]["eligible"] for branch in result["branches"]) / len(dis["branch_ids"])
                if all(branch["weak"]["classifiable"] for branch in result["branches"]) else None)
            effect = joint_at(result, weak_scale)
            row[f"{checkpoint}_joint_percent"] = scale(get(effect, "median_fractional_change"))
            row[f"{checkpoint}_joint_absolute"] = get(effect, "median_absolute_change")
            row[f"{checkpoint}_joint_amplitude_normalized"] = get(effect, "median_normalized_absolute_change")
            row[f"{checkpoint}_native_response"] = get(result, "native_response", "native_response_rms")
            row[f"{checkpoint}_native_classifiable"] = get(result, "native_response", "classifiable")
            row[f"{checkpoint}_nonadditivity_pp"] = scale(get(result, "nonadditivity_weak", "median_fractional_nonadditivity"))
            for units in ("fractional", "physical", "amplitude_normalized"):
                for component in ("positive_sum", "negative_magnitude_sum", "signed_sum"):
                    row[f"{checkpoint}_sensitivity_{units}_{component}"] = get(
                        result, "aggregate_sensitivity", units, component)
            secondary = conf["trained_group_across_checkpoints"]["checkpoints"][checkpoint]
            row[f"{checkpoint}_fixed_trained_group_percent"] = scale(
                get(joint_at(secondary, weak_scale), "median_fractional_change"))
            for dose in result["joint"] or []:
                doses.append({**dict(zip(("tau", "noise_tau", "seed"), identity)),
                              "checkpoint": checkpoint, "scale": dose["scale"],
                              "group_size": dose["group_size"],
                              "response_percent": scale(dose["response"]["median_fractional_change"]),
                              "response_absolute": dose["response"]["median_absolute_change"]})
        for suffix in ("group_percent", "joint_percent", "joint_absolute", "joint_amplitude_normalized",
                       "sensitivity_fractional_positive_sum", "fixed_trained_group_percent"):
            row[f"learning_change_{suffix}"] = delta(row[f"selected_{suffix}"], row[f"initial_{suffix}"])
        row["native_response_trained_initial_ratio"] = (
            row["selected_native_response"] / row["initial_native_response"]
            if row["selected_native_classifiable"] and row["initial_native_classifiable"]
            and row["initial_native_response"] > 0 else None)
        if cal is None or own is None:
            rows.append(row)
            continue
        own_loop_check(own, protocol)
        row["confirmation_unique_rollouts"] = 0
        row["confirmation_task_failures"] = 0
        row["confirmation_censored"] = 0
        row["deployment_classifiable"] = cal["group"] is not None
        row["deployment_group_nonempty"] = bool(cal["group"])
        row["alternative_jaccard"] = cal.get("alternative_jaccard")
        row["alternative_overlap_count"] = cal.get("alternative_overlap_count")
        native = own["summary"]["native"]
        for variant, result in own["summary"].items():
            trials = [item for item in own["rollouts"] if item["variant"] == variant]
            shams = {item["noise_seed"]:item["sham"] for item in trials}
            outcomes = [*shams.values(), *[item["pulse"] for item in trials]]
            row["confirmation_unique_rollouts"] += len(outcomes)
            row["confirmation_task_failures"] += sum(item["task_failure"] for item in outcomes)
            row["confirmation_censored"] += sum(item["censored"] for item in outcomes)
            for arm in ("sham", "pulse"):
                for field in ("task_failure", "censored"):
                    row[f"{variant}_{arm}_{field}_percent"] = scale(result[arm][field])
            if variant == "native":
                continue
            for field in ("position_rms", "action_rms"):
                row[f"{variant}_{field}_percent"] = percent(result["sham"][field], native["sham"][field])
            row[f"{variant}_recovery_percent"] = percent(
                result["paired_recovery"]["normalized_position_energy"],
                native["paired_recovery"]["normalized_position_energy"])
            for field in ("position_mean", "action_mean"):
                row[f"{variant}_{field}_shift"] = delta(result["sham"][field], native["sham"][field])
            if variant in ("gain_matched", "alternative_matched"):
                row[f"{variant}_available_percent"] = 100.
                row[f"{variant}_match_percent"] = scale(result["matched"])
                row[f"{variant}_match_error_percent"] = scale(result["match_relative_error"])
                row[f"{variant}_direction_percent"] = scale(result.get("response_direction_concordant"))
                row[f"{variant}_usable"] = bool(result["matched"] and result.get("response_direction_concordant"))
                for field in ("position_rms_percent", "action_rms_percent", "recovery_percent"):
                    # Population filtering happens below, after both arms from
                    # the same model have been retained.
                    row[f"{variant}_minus_weak_{field}"] = delta(row.get(f"{variant}_{field}"),
                                                                 row.get(f"joint_weak_{field}"))
        # JSON is sorted by key, so controls can precede joint_weak. Recompute
        # paired contrasts after all variants have been read.
        for control in ("gain_matched", "alternative_matched"):
            row.setdefault(f"{control}_available_percent", 0.)
            row.setdefault(f"{control}_usable", False)
            for field in ("position_rms_percent", "action_rms_percent", "recovery_percent"):
                row[f"{control}_minus_weak_{field}"] = delta(row.get(f"{control}_{field}"),
                                                             row.get(f"joint_weak_{field}"))
        rows.append(row)
    return rows, doses


def aggregate(base, rows):
    result = []
    fields = sorted({field for row in rows for field in row if field not in {"tau", "noise_tau", "seed"}})
    for tau, noise_tau in itertools.product(base["plant_taus"], base["noise_taus"]):
        pool = [row for row in rows if (row["tau"], row["noise_tau"]) == (tau, noise_tau)]
        cell = {"tau": tau, "noise_tau": noise_tau}
        for field in fields:
            subset = pool
            for control in ("gain_matched", "alternative_matched"):
                if field.startswith(f"{control}_minus_weak_"):
                    subset = [row for row in pool if row[f"{control}_usable"]]
                elif field in (f"{control}_match_percent", f"{control}_match_error_percent", f"{control}_direction_percent"):
                    subset = [row for row in pool if row[f"{control}_available_percent"] > 0]
            values = stats(row.get(field) for row in subset)
            cell.update({f"{field}_{name}": value for name, value in values.items()})
        result.append(cell)
    return result


def frequency_rows(base, records, protocol):
    models = []
    for identity, record in records.items():
        expected_phases = sorted(protocol["frequency"]["phases"])
        combinations = sorted({(row["checkpoint"], row["frequency"]) for row in record["rows"]})
        if set(combinations) != set(itertools.product(CHECKPOINTS, protocol["frequency"]["frequencies"])):
            raise ValueError("Frequency record omits a required checkpoint or frequency")
        for checkpoint, frequency in combinations:
            selected = [row for row in record["rows"] if (row["checkpoint"], row["frequency"]) == (checkpoint, frequency)]
            if sorted(row["phase"] for row in selected) != expected_phases:
                raise ValueError("Incomplete or duplicate frequency forcing phases")
            row = {**dict(zip(("tau", "noise_tau", "seed"), identity)),
                   "checkpoint": checkpoint, "frequency": frequency, "forcing_phases": len(selected)}
            for field in ("gain_ratio_percent", "native_amplitude", "weak_amplitude", "amplitude_change",
                          "native_fit_residual_rms", "weak_fit_residual_rms", "native_clip_fraction", "weak_clip_fraction"):
                row[field] = stats(item.get(field) for item in selected)["mean"]
            phase = circular_stats([item.get("phase_difference_deg") for item in selected])
            row["phase_difference_deg"] = phase["mean"]
            row["phase_valid_fraction"] = float(np.mean([bool(item.get("phase_valid")) for item in selected]))
            row["baseline_valid_fraction"] = float(np.mean([bool(item.get("baseline_valid")) for item in selected]))
            models.append(row)
    cells = []
    combinations = sorted({(row["checkpoint"], row["frequency"]) for row in models})
    for tau, noise_tau in itertools.product(base["plant_taus"], base["noise_taus"]):
        for checkpoint, frequency in combinations:
            selected = [row for row in models if (row["tau"], row["noise_tau"], row["checkpoint"], row["frequency"])
                        == (tau, noise_tau, checkpoint, frequency)]
            if {row["seed"] for row in selected} != set(base["seeds"]) or len(selected) != len(base["seeds"]):
                raise ValueError("Incomplete frequency model-seed population")
            row = {"tau": tau, "noise_tau": noise_tau, "checkpoint": checkpoint, "frequency": frequency}
            for field in ("gain_ratio_percent", "phase_difference_deg", "native_amplitude", "weak_amplitude",
                          "native_fit_residual_rms", "weak_fit_residual_rms", "phase_valid_fraction", "baseline_valid_fraction"):
                values = [item.get(field) for item in selected]
                measured = circular_stats(values) if field == "phase_difference_deg" else stats(values)
                row.update({f"{field}_{name}": value for name, value in measured.items()})
            cells.append(row)
    return models, cells


def edges(values):
    logs = np.log(np.asarray(values, dtype=float))
    if len(logs) == 1:
        return np.exp(logs[0] + np.asarray([-.35, .35]))
    mid = (logs[:-1] + logs[1:]) / 2
    return np.exp(np.r_[2 * logs[0] - mid[0], mid, 2 * logs[-1] - mid[-1]])


def panel(axis, base, cells, field, title, *, diverging=False, unit="", limits=None, center=0.):
    index = {(row["tau"], row["noise_tau"]): row for row in cells}
    data = np.array([[index[(tau, nt)].get(f"{field}_mean") if number(index[(tau, nt)].get(f"{field}_mean")) else np.nan
                      for tau in base["plant_taus"]] for nt in base["noise_taus"]])
    finite = data[np.isfinite(data)]
    if diverging:
        bound = max(float(np.max(np.abs(finite-center))) if len(finite) else 1., 1e-12)
        lo, hi = limits if limits is not None else (center-bound, center+bound)
        if center == 1. and limits is None:
            lo, hi = 0., max(2., float(np.max(finite)) if len(finite) else 2.)
        norm = TwoSlopeNorm(vmin=lo, vcenter=center, vmax=hi)
        palette = plt.get_cmap("RdBu_r").copy()
    else:
        lo, hi = limits if limits else (0., float(np.max(finite)) if len(finite) else 1.)
        norm = Normalize(lo, hi if hi > lo else lo + 1.)
        palette = plt.get_cmap("viridis").copy()
    palette.set_bad("#dddddd")
    artist = axis.pcolormesh(edges(base["plant_taus"]), edges(base["noise_taus"]),
                             np.ma.masked_invalid(data), cmap=palette, norm=norm,
                             edgecolors="white", linewidth=.8, shading="flat")
    axis.set(xscale="log", yscale="log", xlabel="Plant τ (s)", ylabel="Training-noise correlation time (s)")
    axis.set_xticks(base["plant_taus"], [f"{x:g}" for x in base["plant_taus"]])
    axis.set_yticks(base["noise_taus"], [f"{x:g}" for x in base["noise_taus"]])
    axis.minorticks_off()
    top = axis.secondary_xaxis("top")
    top.set_xticks(base["plant_taus"], [f"{base['delay']/x:g}" for x in base["plant_taus"]])
    top.set_xlabel("Delay / plant τ", fontsize=8, labelpad=2)
    top.tick_params(labelsize=8, pad=1)
    top.minorticks_off()
    axis.set_title(title, fontsize=10.5, pad=31)
    for iy, nt in enumerate(base["noise_taus"]):
        for ix, tau in enumerate(base["plant_taus"]):
            cell = index[(tau, nt)]
            value = data[iy, ix]
            n, valid = cell.get(f"{field}_n", 0), cell.get(f"{field}_valid_n", 0)
            label = f"{value:.3g}" if np.isfinite(value) else "N/A"
            label += f"\nn={valid}/{n}"
            color = "black"
            if np.isfinite(value):
                rgba = palette(norm(value))
                color = "white" if np.dot(rgba[:3], [.2126, .7152, .0722]) < .46 else "black"
            axis.text(tau, nt, label, ha="center", va="center", fontsize=8.3, color=color)
    bar = axis.figure.colorbar(artist, ax=axis, fraction=.045, pad=.035)
    bar.ax.tick_params(labelsize=8)
    bar.set_label(unit, fontsize=8.5)


def save_figure(fig, directory, name, title, note):
    fig.suptitle(title, fontsize=15, y=.995)
    fig.text(.5, .006, note, ha="center", va="bottom", fontsize=8.5)
    fig.tight_layout(rect=(0., .05, 1., .97), h_pad=1.8, w_pad=1.4)
    fig.savefig(directory / f"{name}.png", dpi=175)
    path = directory / f"{name}.svg"
    fig.savefig(path, metadata={"Date": None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def map_plots(base, cells, directory, common_only=False):
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    joint_bound = max([abs(row[f"{checkpoint}_joint_percent_mean"]) for row in cells for checkpoint in CHECKPOINTS
                       if number(row.get(f"{checkpoint}_joint_percent_mean"))] or [1.])
    sensitivity_max = max([row[f"{checkpoint}_sensitivity_fractional_positive_sum_mean"] for row in cells for checkpoint in CHECKPOINTS
                          if number(row.get(f"{checkpoint}_sensitivity_fractional_positive_sum_mean"))] or [1.])
    for col, checkpoint in enumerate(CHECKPOINTS):
        panel(axes[0, col], base, cells, f"{checkpoint}_group_percent", f"{LABELS[checkpoint]}: selected branches",
              limits=(0., 100.), unit="Discovery-selected branches (%)")
        panel(axes[1, col], base, cells, f"{checkpoint}_joint_percent", f"{LABELS[checkpoint]}: joint weakening",
              diverging=True, limits=(-max(joint_bound, 1e-12), max(joint_bound, 1e-12)), unit="Held-out response change (%)")
        panel(axes[2, col], base, cells, f"{checkpoint}_sensitivity_fractional_positive_sum", f"{LABELS[checkpoint]}: positive sensitivity sum",
              limits=(0., sensitivity_max), unit="Sum of median fractional sensitivities")
    save_figure(fig, directory, "collective_suppression", "Ten residual-output branches · collective functional suppression",
                "Common teacher histories are shared across checkpoints and training-noise conditions at fixed plant τ. Groups are selected separately at each checkpoint.\n"
                "Eight projected attention heads + two complete MLP outputs; other computations remain fixed. n = valid / required model seeds. Grey = undefined.")
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    specs = [("learning_change_group_percent", "Trained − initial: selected branch fraction", "Percentage points"),
             ("learning_change_joint_percent", "Trained − initial: joint suppression", "Percentage points"),
             ("learning_change_sensitivity_fractional_positive_sum", "Trained − initial: positive sensitivity sum", "Sensitivity units"),
             ("learning_change_joint_absolute", "Trained − initial: physical effect", "Command units"),
             ("learning_change_joint_amplitude_normalized", "Trained − initial: input-normalized effect", "Command change / force amplitude"),
             ("native_response_trained_initial_ratio", "Native response: trained / initial", "Response RMS ratio"),
             ("selected_joint_absolute", "Trained group: physical response change", "Command units"),
             ("selected_joint_amplitude_normalized", "Trained group: input-normalized change", "Command change / force amplitude"),
             ("selected_nonadditivity_pp", "Joint effect − summed individual effects", "Native-response percentage points")]
    for axis, (field, title, unit) in zip(axes.flat, specs):
        panel(axis, base, cells, field, title, diverging=True, unit=unit,
              center=1. if field=="native_response_trained_initial_ratio" else 0.)
    save_figure(fig, directory, "training_changes", "Training-associated changes · common held-out probes",
                "Primary contrasts compare each checkpoint's discovery-selected group; group identity and size may change. Physical and input-normalized effects retain task-scale dependence.\n"
                "Nonadditivity is computed per probe before taking the model median. This is a finite intervention, not a biological E/I balance measurement.")
    if common_only:
        return
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    specs = [("joint_weak_position_rms_percent", "Joint weakening: noise RMS"),
             ("joint_strong_position_rms_percent", "Joint strengthening: noise RMS"),
             ("gain_matched_minus_weak_position_rms_percent", "Matched gain − weakening: noise RMS"),
             ("joint_weak_action_rms_percent", "Joint weakening: action RMS"),
             ("joint_weak_recovery_percent", "Joint weakening: paired recovery energy"),
             ("alternative_matched_minus_weak_position_rms_percent", "Matched alternative − weakening: noise RMS")]
    for axis, (field, title) in zip(axes.flat, specs):
        panel(axis, base, cells, field, title, diverging=True,
              unit="Percentage points" if "minus_weak" in field else "Change from native (%)")
    save_figure(fig, directory, "collective_usefulness", "Own-history deployment · usefulness and matched controls",
                "Positive position RMS/recovery changes mean worse control; positive action RMS means greater effort. Groups use trained-checkpoint discovery.\n"
                "Control−weakening contrasts pair identical models and require successful magnitude matching plus response-direction concordance. n = valid / required models.")
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    specs = [("gain_matched_match_percent", "Gain-control matching success", False, (0., 100.), "% of available controls"),
             ("alternative_matched_match_percent", "Alternative-group matching success", False, (0., 100.), "% of available controls"),
             ("alternative_matched_direction_percent", "Alternative response-direction concordance", False, (0., 100.), "% of available controls"),
             ("gain_matched_match_error_percent", "Gain-control matching error", False, None, "% of target output change"),
             ("alternative_jaccard", "Selected / alternative group overlap", False, (0., 1.), "Intersection / union"),
             ("selected_mlp_count", "Selected MLP branches", False, (0., 2.), "Branches out of two"),
             ("joint_weak_position_mean_shift", "Weakening: actual mean-position shift", True, None, "Position units"),
             ("joint_weak_action_mean_shift", "Weakening: actual mean-command shift", True, None, "Command units"),
             ("joint_weak_sham_task_failure_percent", "Weakening: noise-regulation task failures", False, (0., 100.), "% of noise trials")]
    for axis, (field, title, signed, limits, unit) in zip(axes.flat, specs):
        panel(axis, base, cells, field, title, diverging=signed, limits=limits, unit=unit)
    save_figure(fig, directory, "collective_quality", "Calibration, group composition, and deployment checks",
                "A distinct same-size alternative group does not exist for an empty or all-branch selected group. Unavailable controls remain undefined.\n"
                "Large groups necessarily overlap; matching magnitude alone does not establish matching response direction. Mean correction is calibrated, then drift is measured afresh.")


def dose_plot(base, doses, directory):
    fig, axes = plt.subplots(len(base["noise_taus"]), len(base["plant_taus"]), figsize=(16, 12), squeeze=False)
    colors = {"initial": "#6b6b6b", "epoch_025": "#b66a16", "selected": "#196fa3"}
    for iy, nt in enumerate(reversed(base["noise_taus"])):
        for ix, tau in enumerate(base["plant_taus"]):
            axis = axes[iy, ix]
            for checkpoint in CHECKPOINTS:
                selected = [row for row in doses if (row["tau"], row["noise_tau"], row["checkpoint"]) == (tau, nt, checkpoint)]
                strengths = sorted({row["scale"] for row in selected} | {1.})
                native_classifiable = ({row["seed"] for row in selected if number(row["response_percent"])} == set(base["seeds"]))
                xs, means, lows, highs = [], [], [], []
                for strength in strengths:
                    values = [row["response_percent"] for row in selected if row["scale"] == strength]
                    measured = stats(values) if strength != 1. else {
                        "mean": 0. if native_classifiable else None, "min": 0., "max": 0., "n": len(base["seeds"])}
                    if strength != 1. and len(values) != len(base["seeds"]):
                        measured["mean"] = None
                    if measured["mean"] is not None:
                        xs.append(strength); means.append(measured["mean"]); lows.append(measured["min"]); highs.append(measured["max"])
                axis.errorbar(xs, means, yerr=[np.asarray(means)-lows, np.asarray(highs)-means],
                              marker="o", markersize=3, capsize=2, linewidth=1,
                              color=colors[checkpoint], label=LABELS[checkpoint])
            axis.axhline(0., color="black", linewidth=.5)
            axis.set_title(f"Plant τ={tau:g} s · training-noise τ={nt:g} s", fontsize=9)
            axis.set_xlabel("Selected-branch multiplier")
            axis.set_ylabel("Response change (%)")
            axis.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8)
    save_figure(fig, directory, "collective_dose_response", "Joint intervention strength · independently selected groups at each checkpoint",
                "Points are tested strengths; lines guide the eye. Bars show the minimum and maximum across model seeds, not confidence intervals. Identity at multiplier1 is exactly zero.\n"
                "Groups are fixed by discovery before these common-history confirmation assays. Cells with unavailable required responses are omitted rather than imputed.")


def frequency_plots(base, cells, directory):
    frequencies = sorted({row["frequency"] for row in cells})
    colors = plt.get_cmap("tab10").colors
    for field, filename, title, units in (("gain_ratio_percent", "collective_frequency_gain", "Joint weakening: frequency-dependent command amplitude", "Amplitude change (%)"),
                                          ("phase_difference_deg", "collective_frequency_phase", "Joint weakening: conditional command phase shift", "Weak − native phase (degrees)")):
        fig, axes = plt.subplots(3, len(base["plant_taus"]), figsize=(17, 12), squeeze=False)
        for iy, checkpoint in enumerate(CHECKPOINTS):
            for ix, tau in enumerate(base["plant_taus"]):
                axis = axes[iy, ix]
                panel_records = [row for row in cells if (row["tau"], row["checkpoint"]) == (tau, checkpoint)]
                complete_points = sum(number(row[f"{field}_mean"]) for row in panel_records)
                valid_models = sum(row[f"{field}_valid_n"] for row in panel_records)
                required_models = sum(row[f"{field}_n"] for row in panel_records)
                for ni, nt in enumerate(base["noise_taus"]):
                    selected = sorted((row for row in cells if (row["tau"], row["noise_tau"], row["checkpoint"]) == (tau, nt, checkpoint)), key=lambda x:x["frequency"])
                    xs, means, lows, highs = [], [], [], []
                    for row in selected:
                        value = row[f"{field}_mean"]
                        if number(value):
                            xs.append(row["frequency"] * np.exp((ni-1.5)*.025))
                            means.append(value); lows.append(row[f"{field}_min"]); highs.append(row[f"{field}_max"])
                    if xs:
                        axis.errorbar(xs, means, yerr=[np.maximum(0., np.asarray(means)-lows), np.maximum(0., np.asarray(highs)-means)],
                                      fmt="o", markersize=4, capsize=2, color=colors[ni], label=f"Training noise τ={nt:g} s")
                axis.axhline(0., color="black", linewidth=.5)
                axis.set(xscale="log", xlabel="Force frequency (Hz)", ylabel=units)
                axis.set_xticks(frequencies, [f"{value:g}" for value in frequencies])
                axis.minorticks_off()
                axis.grid(alpha=.15)
                axis.set_title(f"{LABELS[checkpoint]} · plant τ={tau:g} s", fontsize=10)
                axis.text(.02, .98, f"Complete points {complete_points}/{len(panel_records)}; valid {valid_models}/{required_models}",
                          transform=axis.transAxes, ha="left", va="top", fontsize=6.6,
                          bbox={"facecolor":"white", "alpha":.8, "edgecolor":"none", "pad":1.})
        axes[0, 0].legend(fontsize=7, loc="lower left")
        save_figure(fig, directory, filename, title,
                    "Replayed common teacher histories; these are pre-application command responses, not native closed-loop transfer functions or phase margins.\n"
                    "Markers: equal-seed means after pooling two forcing phases; bars: seed ranges. Each plotted point requires all 3 model seeds. Slight horizontal offsets separate noise conditions.\n"
                    "Valid counts cover model × frequency × training-noise entries per panel; missing required phase/amplitude responses are omitted. See frequency_cells.csv.")


def summary(rows, base):
    result = {"models": len(rows), "cells": len(base["plant_taus"])*len(base["noise_taus"]),
              "scope": "Ten residual-output branches (eight projected attention heads and two complete MLP outputs); other computations fixed.",
              "aggregation": "Equal model-seed weights within each cell. Required missing values invalidate a mean. Model seeds share environment/probe tapes. Control contrasts use the identical matched, direction-concordant model cohort.",
              "checkpoints": {}, "learning": {}, "deployment": {}, "controls": {}}
    for field in ("confirmation_unique_rollouts", "confirmation_task_failures", "confirmation_censored"):
        result[field] = sum(row[field] for row in rows)
    for checkpoint in CHECKPOINTS:
        fields = ("group_size", "group_percent", "confirmed_single_percent", "joint_percent", "joint_absolute",
                  "joint_amplitude_normalized", "native_response", "sensitivity_fractional_positive_sum", "sensitivity_fractional_negative_magnitude_sum", "mlp_count", "nonadditivity_pp")
        result["checkpoints"][checkpoint] = {field: stats(row.get(f"{checkpoint}_{field}") for row in rows) for field in fields}
    for field in ("group_percent", "joint_percent", "joint_absolute", "joint_amplitude_normalized", "sensitivity_fractional_positive_sum", "fixed_trained_group_percent"):
        values = [row.get(f"learning_change_{field}") for row in rows]
        result["learning"][field] = {**stats(values), "positive_models": sum(number(x) and x>0 for x in values), "negative_models": sum(number(x) and x<0 for x in values)}
    result["learning"]["native_response_trained_initial_ratio"] = stats(row["native_response_trained_initial_ratio"] for row in rows)
    result["learning"]["interpretation"] = "A lower native-relative score can coexist with greater physical or input-normalized suppressive effects as responsiveness changes; inspect all scales before describing a change in inhibition."
    for variant in ("joint_weak", "joint_strong"):
        fields = ("position_rms_percent", "action_rms_percent", "recovery_percent", "position_mean_shift", "action_mean_shift")
        result["deployment"][variant] = {}
        for field in fields:
            values = [row.get(f"{variant}_{field}") for row in rows]
            result["deployment"][variant][field] = {**stats(values), "negative_models":sum(number(v) and v<0 for v in values), "positive_models":sum(number(v) and v>0 for v in values)}
    for control in ("gain_matched", "alternative_matched"):
        cohort = [row for row in rows if row[f"{control}_usable"]]
        fields = ("position_rms_percent", "action_rms_percent", "recovery_percent")
        result["controls"][control] = {"available_models":sum(row[f"{control}_available_percent"]>0 for row in rows),
                                       "matched_models":sum(row.get(f"{control}_match_percent")==100 for row in rows),
                                       "matched_direction_concordant_models":len(cohort),
                                       "paired_control_minus_weak":{}}
        for field in fields:
            values = [row.get(f"{control}_minus_weak_{field}") for row in cohort]
            result["controls"][control]["paired_control_minus_weak"][field] = {**stats(values), "control_lower_models":sum(number(v) and v<0 for v in values), "control_higher_models":sum(number(v) and v>0 for v in values)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/collective_suppression")
    parser.add_argument("--source", type=Path, default=ROOT / "results/timescale_maps")
    parser.add_argument("--weak-scale", type=float)
    args = parser.parse_args()
    directory = args.input.resolve()
    base = read_json(args.source / "config.json")
    protocol = read_json(directory / "config.json")
    if base != read_json(directory / "base_config.json"):
        raise ValueError("Plot source configuration differs from the inherited configuration")
    weak_scale = protocol["weak_scale"] if args.weak_scale is None else args.weak_scale
    manifest = read_json(directory / "run_manifest.json")
    expected = list(itertools.product(base["plant_taus"], base["noise_taus"], base["seeds"]))
    collections = {name: indexed(directory / name, expected) for name in
                   ("discovery", "common_confirmation", "dynamics_prepared", "dynamics_confirmation", "frequency")}
    for name in collections:
        for path in (directory / name).glob("*.json"):
            if manifest["completed_sha256"].get(str(path.relative_to(ROOT))) != sha(path):
                raise ValueError(f"Unsealed or changed scientific plot input: {path}")
    models, doses = model_rows(base, collections["discovery"], collections["common_confirmation"],
                               collections["dynamics_prepared"], collections["dynamics_confirmation"], weak_scale, protocol)
    cells = aggregate(base, models)
    fmodels, fcells = frequency_rows(base, collections["frequency"], protocol)
    for filename, rows in (("model_metrics.csv", models), ("collective_cells.csv", cells),
                           ("dose_response.csv", doses), ("frequency_models.csv", fmodels), ("frequency_cells.csv", fcells)):
        write_csv(directory / filename, rows)
    report = summary(models, base)
    report["frequency"] = {"model_checkpoint_frequency_entries":len(fmodels),
                           "amplitude_classifiable_entries":sum(number(row["gain_ratio_percent"]) for row in fmodels),
                           "phase_classifiable_entries":sum(number(row["phase_difference_deg"]) for row in fmodels),
                           "scope":"Conditional pre-application command response to common teacher histories; not native closed-loop gain, damping or phase margin."}
    write_json(directory / "summary.json", report)
    plots = directory / "plots"
    plots.mkdir(exist_ok=True)
    map_plots(base, cells, plots)
    dose_plot(base, doses, plots)
    frequency_plots(base, fcells, plots)
    inputs = [Path(__file__).resolve(), args.source.resolve() / "config.json", directory / "config.json", directory / "run_manifest.json"]
    inputs += [path for name in collections for path in sorted((directory / name).glob("*.json"))]
    outputs = [directory / name for name in ("model_metrics.csv", "collective_cells.csv", "dose_response.csv", "frequency_models.csv", "frequency_cells.csv", "summary.json")]
    outputs += sorted(plots.glob("*.png")) + sorted(plots.glob("*.svg"))
    write_json(directory / "plot_manifest.json", {"inputs_sha256": {str(path.relative_to(ROOT)):sha(path) for path in inputs},
                                                 "outputs_sha256": {str(path.relative_to(ROOT)):sha(path) for path in outputs},
                                                 "weak_scale":weak_scale, "numpy":np.__version__, "matplotlib":matplotlib.__version__})
    print(json.dumps({"models":len(models), "plots":len(list(plots.glob('*.png'))), "output":str(directory)}))


if __name__ == "__main__":
    main()
