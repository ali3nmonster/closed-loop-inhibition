#!/usr/bin/env python3
"""Paired, complete-population summaries of the frozen-controller delay sweep.

Every effect compares the same controller and disturbance streams. Endpoint
interactions pair the same controller at both delays before aggregation. Missing
numeric outcomes invalidate the declared population mean; failures remain
reported. Calibration-qualified control cohorts never change with delay.
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
from matplotlib.colors import TwoSlopeNorm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("native", "joint_weak", "joint_strong", "gain_matched", "alternative_matched")
CONTROLS = ("gain_matched", "alternative_matched")
COLORS = {"native": "#333333", "joint_weak": "#bc3a35", "joint_strong": "#2478ad",
          "gain_matched": "#b07912", "alternative_matched": "#7b5195", "passive": "#808080"}
LABELS = {"native": "Native", "joint_weak": "Weakened", "joint_strong": "Strengthened",
          "gain_matched": "Gain control", "alternative_matched": "Alternative group", "passive": "Passive"}
matplotlib.rcParams["svg.hashsalt"] = "frozen_controller_delay_v1"


def read(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value):
    return isinstance(value, (int, float, bool, np.number)) and bool(np.isfinite(value))


def stats(values):
    values = list(values)
    valid = [float(value) for value in values if number(value)]
    complete = bool(values) and len(valid) == len(values)
    return {"mean": float(np.mean(valid)) if complete else None,
            "min": float(np.min(valid)) if complete else None,
            "max": float(np.max(valid)) if complete else None,
            "n": len(values), "valid_n": len(valid)}


def subtract(a, b):
    return float(a - b) if number(a) and number(b) else None


def ratio(a, b, epsilon=1e-12):
    return float(a / b) if number(a) and number(b) and b > epsilon else None


def percent(a, b, epsilon=1e-12):
    value = ratio(a, b, epsilon)
    return 100. * (value - 1.) if value is not None else None


def identity(record):
    return record["tau"], record["noise_tau"], int(record["seed"])


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def assert_mean(values, reported, label):
    actual = stats(values)["mean"]
    if ((actual is None) != (reported is None)
            or actual is not None and not np.isclose(actual, reported, rtol=1e-11, atol=1e-14)):
        raise ValueError(f"Summary disagrees with required trial population: {label}")


def check_trials(record, seeds, amplitudes):
    """Check paired coverage, sham deduplication, and every numeric summary."""
    totals = {"unique_rollouts": 0, "task_failures": 0, "censored": 0}
    for variant, summary in record["summary"].items():
        rows = [row for row in record["rollouts"] if row["variant"] == variant]
        expected = set(itertools.product(seeds, amplitudes))
        keys = [(row["noise_seed"], row["amplitude"]) for row in rows]
        if set(keys) != expected or len(keys) != len(expected):
            raise ValueError("Missing or duplicate paired seed/amplitude trials")
        shams = []
        for seed in seeds:
            copies = [row["sham"] for row in rows if row["noise_seed"] == seed]
            if any(sham != copies[0] for sham in copies):
                raise ValueError("Pulse arms disagree about their shared sham")
            shams.append(copies[0])
        if summary["independent_noise_seeds"] != len(seeds):
            raise ValueError("Incorrect number of independent background tapes")
        arms = {"sham": shams, "pulse": [row["pulse"] for row in rows],
                "paired_recovery": [row["paired_recovery"] for row in rows]}
        for arm, observations in arms.items():
            for field, reported in summary[arm].items():
                if field != "failure_reason":
                    assert_mean([item.get(field) for item in observations], reported, f"{variant}/{arm}/{field}")
        outcomes = arms["sham"] + arms["pulse"]
        totals["unique_rollouts"] += len(outcomes)
        totals["task_failures"] += sum(bool(row["task_failure"]) for row in outcomes)
        totals["censored"] += sum(bool(row["censored"]) for row in outcomes)
    if {row["variant"] for row in record["rollouts"]} != set(record["summary"]):
        raise ValueError("Trial and summary variant sets differ")
    if "physical_rollouts" in record and record["physical_rollouts"] != totals["unique_rollouts"]:
        raise ValueError("Physical rollout accounting disagrees with paired trial records")
    if "timing" in record:
        if len(record["timing"]) != totals["unique_rollouts"] or not all(row["passed"] for row in record["timing"]):
            raise ValueError("A required physical timing audit is missing or failed")
    return totals


def qualified(setting):
    magnitude = setting.get("magnitude_matched", setting.get("matched"))
    direction = setting.get("response_direction_concordant", setting.get("direction_matched"))
    return bool(magnitude and direction)


def flatten(records, passive, seeds, amplitudes, epsilon=1e-12):
    rows = []
    frozen = {}
    for record in sorted(records, key=lambda item: (*identity(item), item["delay"])):
        model = identity(record)
        settings = record["frozen_settings"]
        if set(settings) != set(record["summary"]):
            raise ValueError("Frozen settings and evaluated variants differ")
        signature = json.dumps({"settings": settings, "group": record["group"]}, sort_keys=True)
        if model in frozen and frozen[model] != signature:
            raise ValueError("Controller intervention settings or group changed across delays")
        frozen[model] = signature
        row = {"tau": model[0], "noise_tau": model[1], "seed": model[2],
               "delay": record["delay"], "delay_over_tau": record["delay"] / model[0],
               "group_size": len(record["group"]) if record["group"] is not None else None,
               **check_trials(record, seeds, amplitudes)}
        native = record["summary"]["native"]
        p = passive[(model[0], model[1])]["summary"]["zero_policy"]
        for variant, summary in record["summary"].items():
            row[f"{variant}_recovery_energy"] = summary["paired_recovery"]["normalized_position_energy"]
            row[f"{variant}_recovery_absolute"] = subtract(row[f"{variant}_recovery_energy"], native["paired_recovery"]["normalized_position_energy"])
            row[f"{variant}_recovery_percent"] = percent(row[f"{variant}_recovery_energy"], native["paired_recovery"]["normalized_position_energy"], epsilon)
            row[f"{variant}_recovery_passive_ratio"] = ratio(row[f"{variant}_recovery_energy"], p["paired_recovery"]["normalized_position_energy"], epsilon)
            for field in ("position_rms", "action_rms", "position_mean", "action_mean", "saturation_fraction"):
                row[f"{variant}_{field}"] = summary["sham"][field]
                row[f"{variant}_{field}_absolute"] = subtract(summary["sham"][field], native["sham"][field])
            for field in ("position_rms", "action_rms"):
                row[f"{variant}_{field}_percent"] = percent(summary["sham"][field], native["sham"][field], epsilon)
            row[f"{variant}_position_passive_ratio"] = ratio(summary["sham"]["position_rms"], p["sham"]["position_rms"], epsilon)
            for arm in ("sham", "pulse"):
                for field in ("task_failure", "censored"):
                    row[f"{variant}_{arm}_{field}_percent"] = 100. * summary[arm][field]
            if variant in CONTROLS:
                row[f"{variant}_qualified"] = qualified(settings[variant])
                shifted = record.get("fixed_history", {}).get("variants", {}).get(variant, {})
                for field in ("immediate_output_rms", "weak_relative_magnitude_error", "response_direction_concordant",
                              "command_change_cosine", "matched_on_shifted_histories"):
                    row[f"{variant}_shifted_{field}"] = shifted.get(field)
        row["passive_recovery_energy"] = p["paired_recovery"]["normalized_position_energy"]
        row["passive_position_rms"] = p["sham"]["position_rms"]
        for control in CONTROLS:
            row.setdefault(f"{control}_qualified", False)
            for field in ("recovery_percent", "recovery_absolute", "position_rms_percent", "action_rms_percent"):
                row[f"{control}_minus_weak_{field}"] = subtract(row.get(f"{control}_{field}"), row.get(f"joint_weak_{field}"))
        rows.append(row)
    return rows


def endpoint_rows(rows, low, high):
    result = []
    models = sorted({identity(row) for row in rows})
    for model in models:
        pool = {row["delay"]: row for row in rows if identity(row) == model}
        if low not in pool or high not in pool:
            raise ValueError("Missing prespecified interaction endpoint")
        lo, hi = pool[low], pool[high]
        row = dict(zip(("tau", "noise_tau", "seed"), model))
        row.update(low_delay=low, high_delay=high)
        for variant in VARIANTS[1:]:
            for field in ("recovery_absolute", "recovery_percent", "position_rms_absolute", "position_rms_percent", "action_rms_percent"):
                row[f"{variant}_{field}_interaction"] = subtract(hi.get(f"{variant}_{field}"), lo.get(f"{variant}_{field}"))
            lval, hval = lo.get(f"{variant}_recovery_absolute"), hi.get(f"{variant}_recovery_absolute")
            available = number(lval) and number(hval)
            row[f"{variant}_helpful_to_harmful"] = bool(lval < 0 < hval) if available else None
            row[f"{variant}_harmful_to_helpful"] = bool(lval > 0 > hval) if available else None
            row[f"{variant}_low_effect_percent"] = lo.get(f"{variant}_recovery_percent")
            row[f"{variant}_high_effect_percent"] = hi.get(f"{variant}_recovery_percent")
        for control in CONTROLS:
            if lo[f"{control}_qualified"] != hi[f"{control}_qualified"]:
                raise ValueError("Control eligibility changed between interaction endpoints")
            row[f"{control}_qualified"] = lo[f"{control}_qualified"]
            for field in ("recovery_percent", "recovery_absolute", "position_rms_percent"):
                row[f"{control}_minus_weak_{field}_interaction"] = subtract(
                    hi.get(f"{control}_minus_weak_{field}"), lo.get(f"{control}_minus_weak_{field}"))
        result.append(row)
    return result


def cohort(rows, field):
    for control in CONTROLS:
        if field.startswith(control + "_") and ("minus_weak" in field or "shifted_" in field):
            return [row for row in rows if row[f"{control}_qualified"]]
    return rows


def aggregate(rows, by):
    result = []
    fields = sorted({field for row in rows for field in row if field not in {"tau", "noise_tau", "seed", "delay"}})
    for key in sorted({tuple(row[name] for name in by) for row in rows}):
        pool = [row for row in rows if tuple(row[name] for name in by) == key]
        row = dict(zip(by, key))
        for field in fields:
            values = stats(item.get(field) for item in cohort(pool, field))
            row.update({f"{field}_{name}": value for name, value in values.items()})
        result.append(row)
    return result


def signed_stats(values):
    values = list(values)
    return {**stats(values), "positive_models": sum(number(v) and v > 0 for v in values),
            "negative_models": sum(number(v) and v < 0 for v in values)}


def competence_stats(values):
    values = list(values)
    return {**stats(values), "better_than_passive_models": sum(number(v) and v < 1 for v in values),
            "worse_than_passive_models": sum(number(v) and v > 1 for v in values)}


def summarize(rows, endpoints, low, high):
    result = {"model_lineages": len(endpoints), "model_delay_records": len(rows),
              "primary_low_delay": low, "primary_high_delay": high,
              "own_loop_trials": sum(row["unique_rollouts"] for row in rows),
              "own_loop_task_failures": sum(row["task_failures"] for row in rows),
              "own_loop_censored": sum(row["censored"] for row in rows),
              "primary_interactions": {}, "all_model_control_interactions": {}, "controls": {},
              "by_plant": [], "by_delay": [], "by_plant_delay": []}
    for variant in VARIANTS[1:]:
        fields = ("recovery_absolute_interaction", "recovery_percent_interaction", "position_rms_percent_interaction",
                  "helpful_to_harmful", "harmful_to_helpful", "low_effect_percent", "high_effect_percent")
        target = "all_model_control_interactions" if variant in CONTROLS else "primary_interactions"
        result[target][variant] = {field: signed_stats(row.get(f"{variant}_{field}") for row in endpoints) for field in fields}
    for control in CONTROLS:
        pool = [row for row in endpoints if row[f"{control}_qualified"]]
        result["controls"][control] = {"fixed_qualified_models": len(pool), "control_minus_weak_interaction": {}}
        for field in ("recovery_percent", "recovery_absolute", "position_rms_percent"):
            result["controls"][control]["control_minus_weak_interaction"][field] = signed_stats(
                row.get(f"{control}_minus_weak_{field}_interaction") for row in pool)
    for tau in sorted({row["tau"] for row in rows}):
        result["by_plant"].append({"tau": tau, "interactions": {
            variant: {field: signed_stats(row.get(f"{variant}_{field}") for row in endpoints if row["tau"] == tau)
                      for field in ("recovery_absolute_interaction", "recovery_percent_interaction", "helpful_to_harmful", "harmful_to_helpful")}
            for variant in ("joint_weak", "joint_strong")}})
    for delay in sorted({row["delay"] for row in rows}):
        pool = [row for row in rows if row["delay"] == delay]
        result["by_delay"].append({"delay": delay, "variants": {
            variant: {field: signed_stats(row.get(f"{variant}_{field}") for row in pool)
                      for field in ("recovery_energy", "recovery_percent", "recovery_absolute", "recovery_passive_ratio",
                                    "position_rms", "position_passive_ratio", "position_rms_percent", "action_rms_percent", "saturation_fraction",
                                    "sham_task_failure_percent", "pulse_task_failure_percent")}
            for variant in VARIANTS}, "passive_competence": {
            variant: {field: competence_stats(row.get(f"{variant}_{field}") for row in pool)
                      for field in ("recovery_passive_ratio", "position_passive_ratio")}
            for variant in ("native", "joint_weak", "joint_strong")}, "controls": {
            control: {field: signed_stats(row.get(f"{control}_{field}") for row in pool if row[f"{control}_qualified"])
                      for field in ("minus_weak_recovery_percent", "minus_weak_position_rms_percent",
                                    "shifted_weak_relative_magnitude_error", "shifted_matched_on_shifted_histories",
                                    "shifted_response_direction_concordant")}
            for control in CONTROLS}})
        for tau in sorted({row["tau"] for row in pool}):
            plant = [row for row in pool if row["tau"] == tau]
            result["by_plant_delay"].append({"tau": tau, "delay": delay, "variants": {
                variant: {field: signed_stats(row.get(f"{variant}_{field}") for row in plant)
                          for field in ("recovery_energy", "recovery_percent", "recovery_passive_ratio",
                                        "position_rms", "position_passive_ratio", "position_rms_percent")}
                for variant in ("native", "joint_weak", "joint_strong")}})
    result["interpretation"] = ("Within-controller finite-horizon delay-by-intervention effects. Weights, groups and calibration settings are frozen; "
        "the explicit delay feature remains at training 50 ms. Reported seed ranges are descriptive, not confidence intervals. "
        "These are deployed already-trained controllers; the experiment does not estimate emergence during training or biological E/I balance.")
    return result


def save(fig, directory, name, title, note):
    fig.suptitle(title, fontsize=15, y=.995)
    fig.text(.5, .007, note, ha="center", va="bottom", fontsize=8.2)
    fig.tight_layout(rect=(0., .055, 1., .97), h_pad=1.9, w_pad=1.6)
    fig.savefig(directory / f"{name}.png", dpi=160)
    path = directory / f"{name}.svg"
    fig.savefig(path, metadata={"Date": None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def curve_plot(rows, cells, directory, name, fields, title, ylabel, baseline, *, reference=0., log=False):
    taus = sorted({row["tau"] for row in rows})
    nts = sorted({row["noise_tau"] for row in rows}, reverse=True)
    fig, axes = plt.subplots(len(nts), len(taus), figsize=(16, 12), squeeze=False)
    for iy, nt in enumerate(nts):
        for ix, tau in enumerate(taus):
            axis = axes[iy, ix]
            population = [row for row in rows if (row["tau"], row["noise_tau"]) == (tau, nt)]
            averaged = sorted((row for row in cells if (row["tau"], row["noise_tau"]) == (tau, nt)), key=lambda row: row["delay"])
            xs = np.asarray([row["delay"] * 1000 for row in averaged])
            for field, variant, label in fields:
                eligible = cohort(population, field)
                style = "--" if "_pulse_" in field else "-"
                for seed in sorted({row["seed"] for row in eligible}):
                    individual = sorted((row for row in eligible if row["seed"] == seed), key=lambda row: row["delay"])
                    axis.plot([row["delay"] * 1000 for row in individual],
                              [row.get(field) if number(row.get(field)) else np.nan for row in individual],
                              color=COLORS[variant], alpha=.18, linewidth=.7, linestyle=style)
                means = np.asarray([row.get(field + "_mean") if number(row.get(field + "_mean")) else np.nan for row in averaged])
                lower = np.asarray([row.get(field + "_min") if number(row.get(field + "_min")) else np.nan for row in averaged])
                upper = np.asarray([row.get(field + "_max") if number(row.get(field + "_max")) else np.nan for row in averaged])
                ns = sorted({row[field + "_n"] for row in averaged})
                valid = sorted({row[field + "_valid_n"] for row in averaged})
                denom = ",".join(map(str, ns))
                numer = str(valid[0]) if len(valid) == 1 else f"{valid[0]}–{valid[-1]}"
                legend = f"{label} (n={numer}/{denom})" if variant != "passive" else "Passive (shared reference)"
                axis.errorbar(xs, means, yerr=[np.maximum(0., means - lower), np.maximum(0., upper - means)],
                              color=COLORS[variant], label=legend, linewidth=1.25,
                              marker="o", markersize=3, capsize=2, linestyle=style)
            axis.axvline(baseline * 1000, color="black", linestyle=":", linewidth=.8)
            if reference is not None:
                axis.axhline(reference, color="black", linewidth=.5)
            if log:
                axis.set_yscale("log")
            if all("task_failure_percent" in field for field, _, _ in fields):
                axis.set_ylim(-3., 103.)
            if all("saturation_fraction" in field for field, _, _ in fields):
                axis.set_ylim(-.02, 1.02)
            axis.set_title(f"Plant τ={tau:g} s · noise τ={nt:g} s", fontsize=9.5)
            axis.set_xlabel("Physical computation delay (ms)", fontsize=8.5)
            axis.set_ylabel(ylabel, fontsize=8.5)
            axis.tick_params(labelsize=8)
            axis.grid(alpha=.15)
            axis.legend(fontsize=6.5, loc="best")
    save(fig, directory, name, title,
         "Fixed controller, group, offsets and explicit delay cue (50 ms); observation/update period 50 ms. Dotted line = training delay.\n"
         "Points/lines = means; bars = model-seed minimum–maximum; faint curves = individual models. Missing required outcomes break curves; no survivor averages.")


def interaction_plot(endpoints, directory, low, high):
    cells = aggregate(endpoints, ("tau", "noise_tau"))
    taus = sorted({row["tau"] for row in cells})
    nts = sorted({row["noise_tau"] for row in cells})
    lookup = {(row["tau"], row["noise_tau"]): row for row in cells}
    specs = [("joint_weak_recovery_absolute_interaction", "Weakening: raw energy interaction", "Energy / force²"),
             ("joint_weak_recovery_percent_interaction", "Weakening: relative interaction", "Percentage points"),
             ("joint_weak_helpful_to_harmful", "Weakening: helpful → harmful", "Fraction of models"),
             ("joint_strong_recovery_absolute_interaction", "Strengthening: raw energy interaction", "Energy / force²"),
             ("joint_strong_recovery_percent_interaction", "Strengthening: relative interaction", "Percentage points"),
             ("joint_strong_harmful_to_helpful", "Strengthening: harmful → helpful", "Fraction of models")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for axis, (field, title, units) in zip(axes.flat, specs):
        data = np.asarray([[lookup[(tau, nt)][field + "_mean"] if number(lookup[(tau, nt)][field + "_mean"]) else np.nan for tau in taus] for nt in nts])
        finite = data[np.isfinite(data)]
        fraction = "helpful" in field
        bound = max(float(np.max(np.abs(finite))) if len(finite) else 1., 1e-12)
        palette = plt.get_cmap("viridis" if fraction else "RdBu_r").copy()
        palette.set_bad("#d3d3d3")
        image = axis.imshow(data, origin="lower", aspect="auto", cmap=palette,
                            norm=None if fraction else TwoSlopeNorm(vmin=-bound, vcenter=0., vmax=bound),
                            vmin=0 if fraction else None, vmax=1 if fraction else None)
        for iy, nt in enumerate(nts):
            for ix, tau in enumerate(taus):
                row = lookup[(tau, nt)]
                value = data[iy, ix]
                count = f"n={row[field + '_valid_n']}/{row[field + '_n']}"
                if np.isfinite(value):
                    label = f"{value:.3g}\n[{row[field+'_min']:.3g},\n{row[field+'_max']:.3g}]\n{count}"
                    rgba = image.cmap(image.norm(value))
                    color = "white" if np.dot(rgba[:3], [.2126, .7152, .0722]) < .46 else "black"
                else:
                    label, color = f"undefined\n{count}", "black"
                axis.text(ix, iy, label, ha="center", va="center", fontsize=8., color=color)
        axis.set(xticks=np.arange(len(taus)), xticklabels=[f"{v:g}" for v in taus],
                 yticks=np.arange(len(nts)), yticklabels=[f"{v:g}" for v in nts],
                 xlabel="Plant τ (s)", ylabel="Noise τ (s)", title=title)
        fig.colorbar(image, ax=axis, fraction=.045, pad=.03, label=units)
    save(fig, directory, "delay_interactions", f"Paired within-controller interaction · {high*1000:g} ms minus {low*1000:g} ms",
         "Interaction = (intervention − native) at high delay minus (intervention − native) at low delay, computed per model before averaging.\n"
         "Positive weakening interaction means increasing its relative cost; sign-switch panels additionally require opposite endpoint signs. Brackets show model-seed ranges.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/delay_sweep")
    args = parser.parse_args()
    directory = args.input.resolve()
    config, base, manifest = [read(directory / name) for name in ("config.json", "base_config.json", "run_manifest.json")]
    if config != manifest["protocol_config"] or base != manifest["base_config"]:
        raise ValueError("Copied scientific configuration differs from frozen manifest")
    if not {"passive", "confirm"} <= set(manifest["stages"]):
        raise ValueError("Scientific run has not sealed every required stage")
    expected = list(itertools.product(base["plant_taus"], base["noise_taus"], base["seeds"]))
    if config.get("model_subset"):
        expected = [tuple(item) if isinstance(item, list) else (item["tau"], item["noise_tau"], item["seed"]) for item in config["model_subset"]]
    files = sorted((directory / "confirmation").glob("*.json"))
    records = [read(path) for path in files]
    wanted = {(*model, delay) for model in expected for delay in config["delays"]}
    observed = [(*identity(row), row["delay"]) for row in records]
    if set(observed) != wanted or len(observed) != len(wanted):
        raise ValueError("Incomplete or duplicate model-delay population")
    if any(row["delay_cue"] != config["delay_cue"] or row["period"] != base["period"] for row in records):
        raise ValueError("A record changed the fixed cue or update period")
    pfiles = sorted((directory / "passive").glob("*.json"))
    passive = {(row["tau"], row["noise_tau"]): row for row in map(read, pfiles)}
    if set(passive) != {model[:2] for model in expected} or len(passive) != len(pfiles):
        raise ValueError("Incomplete or duplicate passive reference population")
    for path in [directory / "config.json", directory / "base_config.json", *files, *pfiles]:
        if manifest["completed_sha256"].get(str(path.relative_to(ROOT))) != sha(path):
            raise ValueError(f"Unsealed or changed scientific input: {path}")
    seeds, amplitudes = config["confirmation_seeds"], config["probe"]["pulse_amplitudes"]
    passive_totals = [check_trials(record, seeds, amplitudes) for record in passive.values()]
    rows = flatten(records, passive, seeds, amplitudes, config["metric_denominator_epsilon"])
    low, high = config["primary_low_delay"], config["primary_high_delay"]
    endpoints = endpoint_rows(rows, low, high)
    cells = aggregate(rows, ("tau", "noise_tau", "delay"))
    interactions = aggregate(endpoints, ("tau", "noise_tau"))
    for filename, values in (("model_metrics.csv", rows), ("delay_cells.csv", cells),
                             ("model_interactions.csv", endpoints), ("interaction_cells.csv", interactions)):
        write_csv(directory / filename, values)
    summary = summarize(rows, endpoints, low, high)
    summary["passive_trials"] = sum(row["unique_rollouts"] for row in passive_totals)
    summary["passive_task_failures"] = sum(row["task_failures"] for row in passive_totals)
    summary["passive_censored"] = sum(row["censored"] for row in passive_totals)
    summary["total_trials"] = summary["own_loop_trials"] + summary["passive_trials"]
    write_json(directory / "summary.json", summary)
    plots = directory / "plots"
    plots.mkdir(exist_ok=True)
    baseline = config["baseline_delay"]
    specs = [("delay_recovery_effects", [(v + "_recovery_percent", v, LABELS[v]) for v in ("joint_weak", "joint_strong")],
              "Frozen pathways · delay-dependent recovery effects", "Recovery change from native (%)", 0., False),
             ("delay_recovery_energy", [(v + "_recovery_energy", v, LABELS[v]) for v in ("native", "joint_weak", "joint_strong", "passive")],
              "Absolute paired recovery energy · finite-horizon deployment", "Position energy / force²", None, True),
             ("delay_passive_ratios", [(v + "_recovery_passive_ratio", v, LABELS[v]) for v in ("native", "joint_weak", "joint_strong")],
              "Recovery relative to the passive plant", "Recovery energy / passive recovery energy", 1., True),
             ("delay_noise_effects", [(v + "_position_rms_percent", v, LABELS[v]) for v in ("joint_weak", "joint_strong")],
              "Noise regulation · delay-dependent effects", "Noise position RMS change (%)", 0., False),
             ("delay_noise_accuracy", [(v + "_position_rms", v, LABELS[v]) for v in ("native", "joint_weak", "joint_strong", "passive")],
              "Absolute noise-regulation error · passive comparator", "Noise position RMS", None, True),
             ("delay_noise_passive_ratios", [(v + "_position_passive_ratio", v, LABELS[v]) for v in ("native", "joint_weak", "joint_strong")],
              "Noise regulation relative to the passive plant", "Position RMS / passive position RMS", 1., True),
             ("delay_action_effort", [(v + "_action_rms", v, LABELS[v]) for v in ("native", "joint_weak", "joint_strong")],
              "Applied command effort · finite-horizon deployment", "Applied action RMS", None, True),
             ("delay_task_failures", [(v + "_" + arm + "_task_failure_percent", v, LABELS[v] + " " + arm)
                                       for v in ("native", "joint_weak", "joint_strong") for arm in ("sham", "pulse")],
              "Completed task failures · sham (solid) and pulse (dashed)", "Task failures (% of required trials)", 0., False),
             ("delay_saturation", [(v + "_saturation_fraction", v, LABELS[v]) for v in ("native", "joint_weak", "joint_strong")],
              "Actuator saturation · exact held-action time fraction", "Saturated fraction of scoring window", 0., False),
             ("delay_controls", [(v + "_minus_weak_recovery_percent", v, LABELS[v]) for v in CONTROLS],
              "Calibration-qualified controls minus group weakening", "Recovery-effect difference (percentage points)", 0., False),
             ("delay_match_drift", [(v + "_shifted_weak_relative_magnitude_error", v, LABELS[v]) for v in CONTROLS],
              "Fixed controls · matching drift on shifted native histories", "Magnitude error / weakening command effect", 0.05, False)]
    for name, fields, title, ylabel, reference, log in specs:
        curve_plot(rows, cells, plots, name, fields, title, ylabel, baseline, reference=reference, log=log)
    interaction_plot(endpoints, plots, low, high)
    inputs = [Path(__file__).resolve(), directory / "config.json", directory / "base_config.json", directory / "run_manifest.json", *files, *pfiles]
    outputs = [directory / name for name in ("model_metrics.csv", "delay_cells.csv", "model_interactions.csv", "interaction_cells.csv", "summary.json")]
    outputs += sorted(plots.glob("*.png")) + sorted(plots.glob("*.svg"))
    write_json(directory / "plot_manifest.json", {"inputs_sha256": {str(path.relative_to(ROOT)): sha(path) for path in inputs},
               "outputs_sha256": {str(path.relative_to(ROOT)): sha(path) for path in outputs},
               "numpy": np.__version__, "matplotlib": matplotlib.__version__})
    print(json.dumps({"models": len(endpoints), "delay_records": len(rows), "plots": len(specs) + 1, "output": str(directory)}))


if __name__ == "__main__":
    main()
