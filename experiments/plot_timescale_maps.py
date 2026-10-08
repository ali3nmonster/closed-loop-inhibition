#!/usr/bin/env python3
"""Aggregate the frozen force-noise experiment and draw discrete timing maps.

Within a model, noise tapes receive equal weight. Across a grid cell, model
initialization seeds receive equal weight. Missing required numeric results
invalidate an aggregate rather than silently selecting successful trajectories.
Candidate effects are conditional on discovery; matched-control effects are
additionally conditional on passing the prospectively declared matching rule.
"""

import argparse
import csv
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np

matplotlib.rcParams["svg.hashsalt"] = "timescale_maps_v1"


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("centered_weak", "centered_strong", "gain_matched", "alternative_matched")


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _parse(value):
    if value == "":
        return None
    if value in {"True", "False"}:
        return value == "True"
    try:
        return float(value)
    except ValueError:
        return value


def read_csv(path):
    with path.open(newline="") as handle:
        return [{key: _parse(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    return isinstance(value, (int, float, bool, np.number)) and bool(np.isfinite(value))


def strict_stats(values):
    values = list(values)
    valid = [float(value) for value in values if number(value)]
    complete = bool(values) and len(valid) == len(values)
    return {"mean": float(np.mean(valid)) if complete else None,
            "min": float(np.min(valid)) if complete else None,
            "max": float(np.max(valid)) if complete else None,
            "range": float(np.ptp(valid)) if complete else None,
            "n": len(values), "valid_n": len(valid)}


def mean(values):
    return strict_stats(values)["mean"]


def get(record, *keys):
    for key in keys:
        if not isinstance(record, dict):
            return None
        record = record.get(key)
    return record


def scaled(value, scale):
    return float(value * scale) if number(value) else None


def ratio(changed, native, *, percent=False):
    if not number(changed) or not number(native) or native <= 0:
        return None
    return float((changed / native - 1) * 100 if percent else changed / native)


def difference(changed, native):
    return float(changed - native) if number(changed) and number(native) else None


def _key(record):
    return record["tau"], record["noise_tau"], int(record["seed"])


def _indexed_json(directory, expected):
    records = {}
    for path in sorted(directory.glob("*.json")):
        record = read_json(path)
        key = _key(record)
        if key in records:
            raise ValueError(f"Duplicate diagnostic record {key}")
        records[key] = record
    if set(records) != set(expected):
        raise ValueError(f"Incomplete diagnostic collection {directory}: expected {len(expected)}, got {len(records)}")
    return records


def _performance_pool(performance, config, tau, noise_tau, controller, seed=None):
    rows = [row for row in performance
            if (row["tau"], row["noise_tau"], row["controller"], row["model_seed"])
            == (tau, noise_tau, controller, seed)]
    expected = set(config["evaluation"]["seeds"])
    if len(rows) != len(expected) or {row["background_seed"] for row in rows} != expected:
        raise ValueError(f"Incomplete/duplicate performance tapes for {(tau, noise_tau, controller, seed)}")
    return rows


def _fractional_response(summary):
    groups = get(summary, "per_group")
    if not groups or not all(group["baseline_valid"] for group in groups):
        return None
    return scaled(summary["median_fractional_increase"], 100)


def _unique_shams(confirmation, variant, expected):
    rows = [row for row in confirmation["rollouts"] if row["variant"] == variant]
    seeds = {row["noise_seed"] for row in rows}
    if seeds != set(expected):
        raise ValueError(f"Confirmation {variant} has incorrect independent noise seeds")
    selected = []
    for seed in sorted(seeds):
        duplicates = [row["sham"] for row in rows if row["noise_seed"] == seed]
        if any(row != duplicates[0] for row in duplicates[1:]):
            raise ValueError("Repeated pulse signs contain inconsistent sham metrics")
        selected.append(duplicates[0])
    return selected


def model_rows(config, performance, training, prepared=None, confirmed=None):
    expected = list(itertools.product(config["plant_taus"], config["noise_taus"], config["seeds"]))
    training_index = {_key(row): row for row in training}
    if len(training) != len(expected) or set(training_index) != set(expected):
        raise ValueError("Training table is incomplete or contains duplicate model seeds")
    rows = []
    for tau, noise_tau, seed in expected:
        pool = _performance_pool(performance, config, tau, noise_tau, "transformer", seed)
        teacher = _performance_pool(performance, config, tau, noise_tau, "teacher")
        passive = _performance_pool(performance, config, tau, noise_tau, "passive")
        row = {"tau": tau, "noise_tau": noise_tau, "seed": seed,
               "performance_independent_tapes": len(pool),
               "best_epoch": training_index[(tau, noise_tau, seed)]["best_epoch"],
               "validation_mse": training_index[(tau, noise_tau, seed)]["validation_mse"]}
        for field in ("position_rms", "position_mean", "action_rms", "action_mean",
                      "saturation_fraction", "task_failure", "censored"):
            row[f"performance_{field}"] = mean(item[field] for item in pool)
        row["performance_failure_percent"] = scaled(row["performance_task_failure"], 100)
        row["performance_censored_percent"] = scaled(row["performance_censored"], 100)
        row["performance_saturation_percent"] = scaled(row["performance_saturation_fraction"], 100)
        row["teacher_position_rms"] = mean(item["position_rms"] for item in teacher)
        row["passive_position_rms"] = mean(item["position_rms"] for item in passive)
        row["ratio_to_teacher"] = ratio(row["performance_position_rms"], row["teacher_position_rms"])
        row["ratio_to_passive"] = ratio(row["performance_position_rms"], row["passive_position_rms"])
        if prepared is None:
            rows.append(row)
            continue
        prep, confirm = prepared[(tau, noise_tau, seed)], confirmed[(tau, noise_tau, seed)]
        discovery = prep["discovery"]
        row["discovery_available"] = bool(discovery and discovery["available"])
        row["discovery_classifiable"] = bool(discovery and discovery.get("classifiable", False))
        row["candidate_selected"] = get(discovery, "selected") is not None
        row["candidate_layer"] = get(discovery, "selected", "layer")
        row["candidate_head"] = get(discovery, "selected", "head")
        row["candidate_prevalence_percent"] = (100. * row["candidate_selected"]
                                                if row["discovery_classifiable"] else None)
        row["calibration_available"] = bool(get(prep, "calibration", "available"))
        row["calibration_available_percent"] = 100. * row["calibration_available"]
        for checkpoint in ("initial", "epoch_025", "selected"):
            assay = prep["common_history"][checkpoint]
            row[f"common_{checkpoint}_epoch"] = assay["epoch"]
            row[f"common_{checkpoint}_classifiable"] = assay["classifiable"]
            row[f"common_{checkpoint}_eligible_percent"] = scaled(assay["eligible_fraction"], 100)
        row["common_eligible_change_pp"] = difference(row["common_selected_eligible_percent"],
                                                      row["common_initial_eligible_percent"])
        fixed = confirm["fixed_history"]
        row["confirmation_assay_available"] = bool(fixed and fixed["available"])
        for variant in ("raw_weak", "raw_strong", "centered_weak", "centered_strong"):
            assay = get(fixed, variant)
            row[f"response_{variant}_percent"] = _fractional_response(assay)
            row[f"response_{variant}_absolute"] = get(assay, "median_absolute_response_change")
            row[f"response_{variant}_eligible"] = get(assay, "eligible")
        row["confirmation_eligible_percent"] = (scaled(row["response_centered_weak_eligible"], 100)
                                                 if row["response_centered_weak_percent"] is not None else None)
        row["confirmation_raw_eligible_percent"] = (scaled(row["response_raw_weak_eligible"], 100)
                                                     if row["response_raw_weak_percent"] is not None else None)
        native = confirm["summary"]["native"]
        for variant, summary in confirm["summary"].items():
            shams = _unique_shams(confirm, variant, config["diagnostics"]["confirmation_seeds"])
            if summary["independent_noise_seeds"] != len(shams):
                raise ValueError("Confirmation summary overcounts independent shams")
            for field in ("position_rms", "action_rms", "position_mean", "action_mean",
                          "task_failure", "censored"):
                recomputed = mean(item[field] for item in shams)
                reported = summary["sham"][field]
                if (recomputed is None) != (reported is None) or (
                    recomputed is not None and not np.isclose(recomputed, reported, rtol=1e-12, atol=1e-15)
                ):
                    raise ValueError(f"Incorrect independent-seed aggregation for {variant}/{field}")
                row[f"confirmation_{variant}_{field}"] = reported
            for field in ("task_failure", "censored"):
                row[f"confirmation_{variant}_pulse_{field}"] = summary["pulse"][field]
            if variant == "native":
                continue
            for field in ("position_rms", "action_rms"):
                row[f"effect_{variant}_{field}_percent"] = ratio(summary["sham"][field],
                                                                  native["sham"][field], percent=True)
            row[f"effect_{variant}_recovery_percent"] = ratio(
                summary["paired_recovery"]["normalized_position_energy"],
                native["paired_recovery"]["normalized_position_energy"], percent=True)
            for field in ("position_mean", "action_mean", "task_failure", "censored"):
                row[f"effect_{variant}_{field}"] = difference(summary["sham"][field], native["sham"][field])
            row[f"{variant}_matched"] = summary["matched"]
            row[f"{variant}_match_error_percent"] = scaled(summary["match_relative_error"], 100)
            row[f"{variant}_matched_percent"] = scaled(summary["matched"], 100)
        row["both_controls_matched"] = bool(row.get("gain_matched_matched") and row.get("alternative_matched_matched"))
        row["both_controls_matched_percent"] = 100. * row["both_controls_matched"]
        rows.append(row)
    return rows


def aggregate(config, models, performance):
    """Wide per-cell table retaining required and valid denominators per metric."""
    rows = []
    model_fields = sorted({key for row in models for key in row
                           if key not in {"tau", "noise_tau", "seed", "candidate_layer", "candidate_head",
                                          "teacher_position_rms", "passive_position_rms"}})
    for tau, noise_tau in itertools.product(config["plant_taus"], config["noise_taus"]):
        pool = [row for row in models if (row["tau"], row["noise_tau"]) == (tau, noise_tau)]
        row = {"tau": tau, "noise_tau": noise_tau, "model_count": len(pool),
               "latency_over_tau": config["delay"] / tau,
               "period_over_tau": config["period"] / tau,
               "noise_over_tau": noise_tau / tau}
        for controller in ("passive", "teacher"):
            baseline = _performance_pool(performance, config, tau, noise_tau, controller)
            for field in ("position_rms", "action_rms", "task_failure", "censored"):
                stats = strict_stats(item[field] for item in baseline)
                row.update({f"{controller}_{field}_{name}": value for name, value in stats.items()})
        for field in model_fields:
            selected = pool
            if field.startswith(("effect_", "response_", "confirmation_eligible", "confirmation_raw_eligible")):
                selected = [item for item in pool if item.get("candidate_selected")]
            if field.startswith(("gain_matched_", "alternative_matched_", "calibration_available_percent")):
                selected = [item for item in pool if item.get("candidate_selected")]
            stats = strict_stats(item.get(field) for item in selected)
            row.update({f"{field}_{name}": value for name, value in stats.items()})
        for variant in ("gain_matched", "alternative_matched"):
            selected = [item for item in pool if item.get(f"{variant}_matched")]
            field = f"effect_{variant}_position_rms_percent"
            stats = strict_stats(item.get(field) for item in selected)
            row.update({f"matched_{field}_{name}": value for name, value in stats.items()})
        row["candidate_count"] = sum(bool(item.get("candidate_selected")) for item in pool)
        row["missing_candidate_count"] = len(pool) - row["candidate_count"]
        rows.append(row)
    return rows


def edges(centers):
    centers = np.asarray(centers, dtype=float)
    if len(centers) == 1:
        return centers[0] * np.asarray([1. / np.sqrt(2.), np.sqrt(2.)])
    logs = np.log(centers)
    mids = (logs[:-1] + logs[1:]) / 2
    return np.exp(np.r_[logs[0] - (mids[0] - logs[0]), mids, logs[-1] + (logs[-1] - mids[-1])])


def _matrix(config, cells, key):
    index = {(row["tau"], row["noise_tau"]): row for row in cells}
    return np.asarray([[index[(tau, noise_tau)].get(key) if number(index[(tau, noise_tau)].get(key))
                        else np.nan for tau in config["plant_taus"]]
                       for noise_tau in config["noise_taus"]], dtype=float)


def panel(axis, config, cells, key, title, *, nkey=None, cmap="viridis", limit=None,
          diverging=False, unit="", proportion=False, flagkey=None, center=0.):
    data = _matrix(config, cells, key)
    finite = data[np.isfinite(data)]
    if diverging:
        bound = max(float(np.max(np.abs(finite - center))) if len(finite) else 1., 1e-12)
        norm = TwoSlopeNorm(vmin=center - bound, vcenter=center, vmax=center + bound)
        cmap = "RdBu_r"
    else:
        low, high = limit if limit is not None else (0., float(np.max(finite)) if len(finite) else 1.)
        if high <= low:
            high = low + 1.
        norm = Normalize(vmin=low, vmax=high)
    palette = plt.get_cmap(cmap).copy()
    palette.set_bad("#dddddd")
    image = axis.pcolormesh(edges(config["plant_taus"]), edges(config["noise_taus"]),
                            np.ma.masked_invalid(data), cmap=palette, norm=norm,
                            edgecolors="white", linewidth=.8, shading="flat")
    axis.set(xscale="log", yscale="log", xlabel="Plant τ (s)", ylabel="Noise correlation time (s)")
    axis.set_xticks(config["plant_taus"], [f"{value:g}" for value in config["plant_taus"]])
    axis.set_yticks(config["noise_taus"], [f"{value:g}" for value in config["noise_taus"]])
    axis.minorticks_off()
    secondary = axis.secondary_xaxis("top")
    secondary.set_xticks(config["plant_taus"], [f"{config['delay'] / value:g}" for value in config["plant_taus"]])
    secondary.set_xlabel("Delay / plant τ", fontsize=8, labelpad=2)
    secondary.tick_params(labelsize=8, pad=1)
    secondary.minorticks_off()
    axis.set_title(title, fontsize=11, pad=31)
    index = {(row["tau"], row["noise_tau"]): row for row in cells}
    for iy, noise_tau in enumerate(config["noise_taus"]):
        for ix, tau in enumerate(config["plant_taus"]):
            value = data[iy, ix]
            cell = index[(tau, noise_tau)]
            text = (f"{value:.3f}" if key.startswith("ratio_to_teacher") else
                    f"{value:.2f}" if proportion and np.isfinite(value) else f"{value:.3g}")
            if not np.isfinite(value):
                text = "N/A"
            if nkey:
                n = cell.get(nkey)
                valid_key = nkey[:-2] + "_valid_n" if nkey.endswith("_n") else None
                valid = cell.get(valid_key, n)
                text += f"\nn={int(valid)}/{int(n)}" if n is not None and valid is not None else "\nn=0"
            if flagkey and number(cell.get(flagkey)) and cell[flagkey] > 0:
                text += " †"
            color = "black"
            if np.isfinite(value):
                rgba = palette(norm(value))
                luminance = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
                color = "white" if luminance < .46 else "black"
            axis.text(tau, noise_tau, text, ha="center", va="center", fontsize=8.5, color=color)
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=.047, pad=.035)
    colorbar.ax.tick_params(labelsize=8)
    if unit:
        colorbar.set_label(unit, fontsize=9)


def save_figure(fig, directory, name, title, note):
    fig.suptitle(title, fontsize=16, y=.995)
    fig.text(.5, .006, note, ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0., .045, 1., .97), h_pad=1.7, w_pad=1.3)
    fig.savefig(directory / f"{name}.png", dpi=175)
    path = directory / f"{name}.svg"
    fig.savefig(path, metadata={"Date": None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def plots(config, cells, directory, *, only_performance=False):
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    rms_keys = ["passive_position_rms", "teacher_position_rms", "performance_position_rms"]
    rms_max = max(row[f"{key}_mean"] for row in cells for key in rms_keys
                  if number(row.get(f"{key}_mean")))
    for axis, key, title in zip(axes[0], rms_keys, ("Passive position RMS", "Delay-aware teacher position RMS", "Transformer position RMS")):
        failure_key = "performance_task_failure_mean" if key.startswith("performance") else key.replace("position_rms", "task_failure") + "_mean"
        panel(axis, config, cells, f"{key}_mean", title, nkey=f"{key}_n", limit=(0., rms_max), unit="Position units", flagkey=failure_key)
    for axis, key, title in zip(axes[1], ("ratio_to_teacher", "ratio_to_passive", "performance_action_rms"),
                                ("Transformer / teacher RMS", "Transformer / passive RMS", "Transformer action RMS")):
        panel(axis, config, cells, f"{key}_mean", title, nkey=f"{key}_n", unit="Ratio" if "ratio" in key else "Command units",
              diverging=key == "ratio_to_teacher", center=1. if key == "ratio_to_teacher" else 0.)
    for axis, key, title in zip(axes[2, :2], ("performance_failure_percent", "performance_censored_percent"),
                                ("Transformer task failures (%)", "Transformer guard censoring (%)")):
        panel(axis, config, cells, f"{key}_mean", title, nkey=f"{key}_n", limit=(0., 100.))
    panel(axes[2, 2], config, cells, "performance_position_rms_range", "Transformer RMS range across seeds",
          nkey="performance_position_rms_n", unit="Max − min")
    save_figure(fig, directory, "performance", "Fixed 50 ms computation delay and update period · force disturbances",
                "Passive/teacher n counts noise tapes; transformer n counts model seeds (4 independent tapes per model). "
                "RMS uses 3–12 s.\nGrey = unavailable required metric; † = at least one task failure. Noise tapes are shared across models/cells; no interpolation.")
    if only_performance:
        return
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    for axis, checkpoint, title in zip(axes[0], ("initial", "epoch_025", "selected"),
                                       ("Common teacher histories: initialization", "Common teacher histories: epoch 25", "Common teacher histories: selected checkpoint")):
        key = f"common_{checkpoint}_eligible_percent"
        panel(axis, config, cells, f"{key}_mean", title, nkey=f"{key}_n", limit=(0., 100.), unit="Eligible heads (%)")
    panel(axes[1, 0], config, cells, "common_eligible_change_pp_mean", "Learned − initial eligible head fraction",
          nkey="common_eligible_change_pp_n", diverging=True, unit="Percentage points")
    panel(axes[1, 1], config, cells, "candidate_prevalence_percent_mean", "Models with a discovery-selected pathway",
          nkey="candidate_prevalence_percent_n", limit=(0., 100.), unit="Models (%)")
    panel(axes[1, 2], config, cells, "response_centered_weak_percent_mean", "Held-out response: centered weakening",
          nkey="response_centered_weak_percent_n", diverging=True, unit="Response change (%)")
    panel(axes[2, 0], config, cells, "response_centered_strong_percent_mean", "Held-out response: centered strengthening",
          nkey="response_centered_strong_percent_n", diverging=True, unit="Response change (%)")
    for axis, variant, title in zip(axes[2, 1:], ("centered_weak", "centered_strong"),
                                    ("Weakening: absolute response change", "Strengthening: absolute response change")):
        key = f"response_{variant}_absolute"
        panel(axis, config, cells, f"{key}_mean", title, nkey=f"{key}_n", diverging=True, unit="Command units")
    save_figure(fig, directory, "suppression", "Functional attention-pathway suppression · fixed-history assays",
                "Common histories: standardized force pulses, no background noise; checkpoint panels are categorical. "
                "Selected epoch varies by model.\nHeld-out response: discovery-selected candidates only; n = valid / required models. "
                "This assay does not establish biological E/I balance.")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    specs = [("effect_centered_weak_position_rms_percent", "Centered weakening: noise RMS"),
             ("effect_centered_strong_position_rms_percent", "Centered strengthening: noise RMS"),
             ("matched_effect_gain_matched_position_rms_percent", "Matched output gain: noise RMS"),
             ("matched_effect_alternative_matched_position_rms_percent", "Matched other pathway: noise RMS"),
             ("effect_centered_weak_action_rms_percent", "Centered weakening: action RMS"),
             ("effect_centered_weak_recovery_percent", "Centered weakening: paired recovery energy")]
    for axis, (key, title) in zip(axes.flat, specs):
        variant = next(variant for variant in VARIANTS if variant in key)
        flagkey = f"confirmation_{variant}_{'pulse_' if 'recovery' in key else ''}task_failure_mean"
        panel(axis, config, cells, f"{key}_mean", title, nkey=f"{key}_n", diverging=True, unit="Change from native (%)", flagkey=flagkey)
    save_figure(fig, directory, "causal_effects", "Own-history closed-loop interventions · held-out disturbances",
                "Positive noise-RMS or recovery change means worse control. Positive action-RMS change means more effort. "
                "Noise metrics use 1–4 s.\nCandidate interventions condition on discovery selection; control panels additionally require successful matching. "
                "n = valid / required models; 4 independent noise seeds per model, shared across conditions. † = task failure.")
    fig, axes = plt.subplots(3, 4, figsize=(20, 13))
    specs = [("gain_matched_matched_percent", "Output-gain calibration match rate", False, (0., 100.), "%"),
             ("alternative_matched_matched_percent", "Other-pathway calibration match rate", False, (0., 100.), "%"),
             ("gain_matched_match_error_percent", "Output-gain matching error", False, None, "% of target RMS"),
             ("alternative_matched_match_error_percent", "Other-pathway matching error", False, None, "% of target RMS"),
             ("effect_centered_weak_action_mean", "Weakening: mean-command drift", True, None, "Command units"),
             ("effect_centered_weak_position_mean", "Weakening: mean-position drift", True, None, "Position units"),
             ("effect_centered_strong_action_mean", "Strengthening: mean-command drift", True, None, "Command units"),
             ("effect_centered_strong_position_mean", "Strengthening: mean-position drift", True, None, "Position units"),
             ("confirmation_eligible_percent", "Candidates retaining held-out suppression", False, (0., 100.), "% of candidates"),
             ("both_controls_matched_percent", "Models with both controls matched", False, (0., 100.), "% of all models"),
             ("performance_saturation_percent", "Native actuator saturation time", False, (0., 100.), "% of time"),
             ("calibration_available_percent", "Candidates with usable calibration", False, (0., 100.), "% of candidates")]
    for axis, (key, title, diverging, limit, unit) in zip(axes.flat, specs):
        panel(axis, config, cells, f"{key}_mean", title, nkey=f"{key}_n", diverging=diverging, limit=limit, unit=unit)
    save_figure(fig, directory, "control_quality", "Interpretation checks and denominators",
                "Matching passes at ≤5% error on independent calibration histories. Held-out drift is measured, even after calibration centering.\n"
                "n = valid / required models within the stated population; unavailable candidates are not imputed as zero suppression.")


def summarize(config, models, cells, performance, *, full):
    neural = [row for row in performance if row["controller"] == "transformer"]
    summary = {
        "scope": config["scope"], "grid_cells": len(cells), "trained_models": len(models),
        "model_seeds_per_cell": len(config["seeds"]),
        "performance_rollouts": len(performance), "transformer_performance_rollouts": len(neural),
        "transformer_task_failures": sum(bool(row["task_failure"]) for row in neural),
        "transformer_censored_rollouts": sum(bool(row["censored"]) for row in neural),
        "selected_epoch_range": [int(min(row["best_epoch"] for row in models)), int(max(row["best_epoch"] for row in models))],
        "teacher_better_than_passive_cells": sum(row["teacher_position_rms_mean"] < row["passive_position_rms_mean"]
                                                 for row in cells if number(row["teacher_position_rms_mean"]) and number(row["passive_position_rms_mean"])),
        "transformer_better_than_passive_cells": sum(row["ratio_to_passive_mean"] < 1.
                                                     for row in cells if number(row["ratio_to_passive_mean"])),
        "aggregation": "Mean over independent noise tapes within each model, then equal model-seed weights. "
                       "Any missing numeric result in the required population makes that aggregate null. "
                       "Candidate effects condition on discovery selection; matched control panels additionally condition on matching success. "
                       "Noise tapes are shared across model seeds and grid cells; model seeds do not independently replicate the environment tapes.",
        "interpretation": "Exploratory teacher-imitation controllers with ordinary attention. Functional suppression is not evidence of biological E/I balance. "
                          "Three model seeds do not support a precise phase boundary or a universal architectural conclusion.",
    }
    if not full:
        return summary
    candidates = [row for row in models if row["candidate_selected"]]
    summary.update(selected_candidates=len(candidates), missing_candidates=len(models) - len(candidates),
                   discovery_unavailable_models=sum(not row["discovery_available"] for row in models),
                   discovery_unclassifiable_models=sum(not row["discovery_classifiable"] for row in models),
                   calibrated_candidates=sum(row["calibration_available"] for row in candidates),
                   confirmation_classifiable_candidates=sum(number(row["response_centered_weak_percent"]) for row in candidates),
                   confirmation_eligible_candidates=sum(bool(row["response_centered_weak_eligible"]) for row in candidates),
                   raw_confirmation_eligible_candidates=sum(bool(row["response_raw_weak_eligible"]) for row in candidates),
                   gain_matched_candidates=sum(bool(row.get("gain_matched_matched")) for row in candidates),
                   alternative_matched_candidates=sum(bool(row.get("alternative_matched_matched")) for row in candidates))
    summary["candidate_effects"] = {}
    summary["confirmation_rollout_counts"] = {}
    tape_count = len(config["diagnostics"]["confirmation_seeds"])
    amplitude_count = len(config["diagnostics"]["pulse_amplitudes"])
    for variant in ("native", *VARIANTS):
        pool = [row for row in models if f"confirmation_{variant}_task_failure" in row]
        counts = {"models": len(pool), "unique_sham_rollouts": len(pool) * tape_count,
                  "pulse_rollouts": len(pool) * tape_count * amplitude_count}
        for arm, multiplicity in (("", tape_count), ("pulse_", tape_count * amplitude_count)):
            for field in ("task_failure", "censored"):
                values = [row[f"confirmation_{variant}_{arm}{field}"] for row in pool]
                counts[f"{arm or 'sham_'}{field}_count"] = (
                    int(round(sum(values) * multiplicity)) if all(number(value) for value in values) else None)
        summary["confirmation_rollout_counts"][variant] = counts
    for variant in VARIANTS:
        pool = candidates
        if "matched" in variant:
            pool = [row for row in candidates if row.get(f"{variant}_matched")]
        fields = (f"effect_{variant}_position_rms_percent", f"effect_{variant}_action_rms_percent",
                  f"effect_{variant}_recovery_percent")
        summary["candidate_effects"][variant] = {
            field: strict_stats(row.get(field) for row in pool) for field in fields}
        rms = [row.get(fields[0]) for row in pool]
        summary["candidate_effects"][variant]["models_with_improved_noise_rms"] = sum(number(value) and value < 0 for value in rms)
        summary["candidate_effects"][variant]["models_with_worse_noise_rms"] = sum(number(value) and value > 0 for value in rms)
    summary["heldout_weak_response_percent"] = strict_stats(row["response_centered_weak_percent"] for row in candidates)
    summary["paired_control_comparisons"] = {}
    for control in ("gain_matched", "alternative_matched"):
        pool = [row for row in candidates if row.get(f"{control}_matched")]
        comparison = {"matched_models": len(pool),
                      "units": "Percentage points of change relative to the same model's native value",
                      "comparison": "Control effect minus centered-weak effect, paired within the same model. "
                                    "Negative noise RMS/recovery difference favors the control; negative action RMS means lower effort."}
        for field in ("position_rms_percent", "action_rms_percent", "recovery_percent"):
            deltas = [difference(row.get(f"effect_{control}_{field}"), row.get(f"effect_centered_weak_{field}"))
                      for row in pool]
            comparison[field] = {**strict_stats(deltas),
                                 "control_lower_count": sum(number(value) and value < 0 for value in deltas),
                                 "control_higher_count": sum(number(value) and value > 0 for value in deltas)}
        summary["paired_control_comparisons"][control] = comparison
    summary["common_checkpoint_eligible_percent"] = {
        checkpoint: strict_stats(row[f"common_{checkpoint}_eligible_percent"] for row in models)
        for checkpoint in ("initial", "epoch_025", "selected")}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/timescale_maps.json")
    parser.add_argument("--input", "--output", dest="output", type=Path, default=ROOT / "results/timescale_maps")
    parser.add_argument("--stage", choices=("all", "performance"), default="all")
    args = parser.parse_args()
    config = read_json(args.config)
    performance, training = read_csv(args.output / "performance.csv"), read_csv(args.output / "training.csv")
    expected = list(itertools.product(config["plant_taus"], config["noise_taus"], config["seeds"]))
    full = args.stage == "all"
    prepared = _indexed_json(args.output / "prepared", expected) if full else None
    confirmed = _indexed_json(args.output / "confirmation", expected) if full else None
    models = model_rows(config, performance, training, prepared, confirmed)
    cells = aggregate(config, models, performance)
    plots(config, cells, args.output / "plots", only_performance=not full)
    write_csv(args.output / "model_metrics.csv", models)
    write_csv(args.output / "map_cells.csv", cells)
    ranges = [{key: value for key, value in row.items()
               if key in {"tau", "noise_tau", "model_count"} or key.endswith(("_min", "_max", "_range", "_n"))}
              for row in cells]
    write_csv(args.output / "seed_ranges.csv", ranges)
    summary = summarize(config, models, cells, performance, full=full)
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
