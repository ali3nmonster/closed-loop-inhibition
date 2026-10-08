"""Paired summaries of frozen local-response rescue and fresh training replication.

The inferential unit is an initialization/data seed block, averaged equally over
the four declared noise conditions. Missing required values invalidate the
whole population statistic. Local Jacobian identity is a construction check;
performance on independent trajectories is the empirical outcome.
"""

import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("_kernel_plot_helpers", ROOT / "experiments/plot_delay_sweep.py")
helpers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helpers)
VARIANTS = ("native", "joint_weak", "weak_equilibrium", "weak_scalar", "weak_kernel")
LABELS = {"native": "Native", "joint_weak": "Weakened", "weak_equilibrium": "+ Equilibrium",
          "weak_scalar": "+ Scalar gain", "weak_kernel": "+ Full local response"}
COLORS = {"native": "#333333", "joint_weak": "#c54737", "weak_equilibrium": "#ae7a16",
          "weak_scalar": "#398071", "weak_kernel": "#6252ae"}
CONTRASTS = (("equilibrium_minus_weak", "weak_equilibrium", "joint_weak"),
             ("scalar_minus_equilibrium", "weak_scalar", "weak_equilibrium"),
             ("kernel_minus_scalar", "weak_kernel", "weak_scalar"),
             ("kernel_minus_weak", "weak_kernel", "joint_weak"))
matplotlib.rcParams["svg.hashsalt"] = "frozen_kernel_rescue_v1"
read, write_json, sha = helpers.read, helpers.write_json, helpers.sha
stats, number, subtract, ratio, percent = helpers.stats, helpers.number, helpers.subtract, helpers.ratio, helpers.percent


def identity(record):
    return record["population"], record["tau"], record["noise_tau"], int(record["seed"])


def cell(record):
    return (*identity(record), record["delay"], record["duration"])


def aggregate(rows, by):
    """Strict numeric summaries with explicit complete and required counts."""
    identifiers = {"population", "tau", "noise_tau", "seed", "delay", "duration", "low_delay", "high_delay"}
    fields = sorted({key for row in rows for key in row if key not in identifiers})
    result = []
    for value in sorted({tuple(row[key] for key in by) for row in rows}):
        pool = [row for row in rows if tuple(row[key] for key in by) == value]
        summary = dict(zip(by, value))
        for field in fields:
            summary.update({f"{field}_{name}": entry for name, entry in stats(row.get(field) for row in pool).items()})
        result.append(summary)
    return result


def seed_bootstrap(values, *, resamples, seed, confidence, expected_n):
    """Percentile CI for a mean of complete independent seed-block means.

    Reuse the same resampling indices for every contrast, retaining pairing.
    This is descriptive uncertainty over training replicates conditional on the
    fixed environment grid and probe bank, not an equivalence or significance
    test. Missing blocks never become an available-case bootstrap.
    """
    if (isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1
            or isinstance(expected_n, bool) or not isinstance(expected_n, int) or expected_n < 2
            or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            or not 0 < confidence < 1):
        raise ValueError("Invalid bootstrap specification")
    values = list(values)
    summary = stats(values)
    result = {**summary, "ci_low": None, "ci_high": None, "confidence_level": confidence,
              "resamples": resamples, "bootstrap_seed": seed, "expected_seed_blocks": expected_n,
              "method": "paired_seed_block_percentile_bootstrap"}
    if len(values) != expected_n or summary["valid_n"] != expected_n:
        result["mean"] = result["min"] = result["max"] = None
        return result
    draws = np.random.default_rng(seed).integers(0, expected_n, size=(resamples, expected_n))
    means = np.asarray(values, dtype=float)[draws].mean(axis=1)
    tail = (1. - confidence) / 2.
    result["ci_low"], result["ci_high"] = map(float, np.quantile(means, (tail, 1. - tail)))
    return result


def interactions(rows, low, high):
    """Pair delay effects within a model and horizon before aggregation."""
    result = []
    keys = {(*identity(row), row["duration"]) for row in rows}
    for key in sorted(keys):
        pool = [row for row in rows if (*identity(row), row["duration"]) == key]
        lookup = {row["delay"]: row for row in pool}
        if len(lookup) != len(pool) or low not in lookup or high not in lookup:
            raise ValueError("Missing or duplicate required delay endpoints")
        lo, hi = lookup[low], lookup[high]
        row = dict(zip(("population", "tau", "noise_tau", "seed", "duration"), key))
        row.update(low_delay=low, high_delay=high)
        for name in VARIANTS[1:]:
            for metric in ("recovery_absolute", "recovery_percent", "position_rms_absolute", "action_rms_absolute"):
                row[f"{name}_{metric}_interaction"] = subtract(hi[f"{name}_{metric}"], lo[f"{name}_{metric}"])
            lv, hv = lo[f"{name}_recovery_absolute"], hi[f"{name}_recovery_absolute"]
            complete = number(lv) and number(hv)
            row[f"{name}_helpful_to_harmful"] = bool(lv < 0 < hv) if complete else None
            row[f"{name}_harmful_to_helpful"] = bool(lv > 0 > hv) if complete else None
        for label, after, before in CONTRASTS:
            for metric in ("recovery_absolute", "recovery_percent"):
                row[f"{label}_{metric}_interaction"] = subtract(
                    row[f"{after}_{metric}_interaction"], row[f"{before}_{metric}_interaction"])
        result.append(row)
    return result


def block_inference(endpoints, protocol):
    """Retain all seed blocks and bootstrap the fresh population only."""
    settings = protocol["statistics"]
    seed_rows = aggregate(endpoints, ("population", "tau", "seed", "duration", "low_delay", "high_delay"))
    result = []
    groups = sorted({(row["population"], row["tau"], row["duration"]) for row in endpoints})
    fields = sorted({key for row in endpoints for key in row if key.endswith("_interaction")})
    for population, tau, duration in groups:
        pool = sorted([row for row in seed_rows if (row["population"], row["tau"], row["duration"]) == (population, tau, duration)],
                      key=lambda row: row["seed"])
        declared_seeds = protocol["fresh_seeds"] if population == "fresh" else protocol["existing_seeds"]
        if {row["seed"] for row in pool} != set(declared_seeds):
            raise ValueError("Declared seed-block population is incomplete")
        for item in pool:
            original = [row for row in endpoints if (row["population"], row["tau"], row["duration"], row["seed"])
                        == (population, tau, duration, item["seed"])]
            if len(original) != len(protocol["noise_taus"]) or {row["noise_tau"] for row in original} != set(protocol["noise_taus"]):
                raise ValueError("A seed block lacks the declared complete noise-condition grid")
        entry = {"population": population, "tau": tau, "duration": duration,
                 "seed_blocks": len(pool), "noise_conditions_per_seed": len(protocol["noise_taus"]),
                 "primary_population_horizon": population == "fresh" and duration == protocol["primary_duration"],
                 "metrics": {}}
        for field in fields:
            values = [row[f"{field}_mean"] for row in pool]
            summary = (seed_bootstrap(values, resamples=settings["bootstrap_resamples"], seed=settings["bootstrap_seed"],
                                      confidence=settings["confidence_level"], expected_n=len(declared_seeds))
                       if population == "fresh" and len(declared_seeds) > 1 else {**stats(values), "ci_low": None, "ci_high": None,
                                                      "method": "existing_seed_blocks_descriptive_only"})
            entry["metrics"][field] = {**summary, "by_seed": [{"seed": row["seed"], "value": value}
                                                                  for row, value in zip(pool, values)]}
        result.append(entry)
    return seed_rows, result


def flatten(records, preparations, mechanisms, passive, protocol):
    """Independently validate trial means and frozen correction identity."""
    result, signatures = [], {}
    for record in sorted(records, key=cell):
        key = identity(record)
        prepared, mechanism = preparations[key], mechanisms[(*key, record["delay"])]
        settings = record["frozen_settings"]
        if settings != prepared["variants"] or record.get("group") != prepared.get("group"):
            raise ValueError("Confirmation correction differs from the single 50 ms preparation")
        signature = json.dumps({"settings": settings, "group": record.get("group")}, sort_keys=True)
        if key in signatures and signatures[key] != signature:
            raise ValueError("Corrections changed across delay or confirmation horizon")
        signatures[key] = signature
        available = {name for name, value in record["summary"].items() if value is not None}
        missing = set(VARIANTS) - available
        if (set(settings) != available or set(record["summary"]) != set(VARIANTS)
                or missing != set(record.get("unavailable_controls", {}))):
            raise ValueError("Required nested policy population or explicit unavailability is incomplete")
        if (mechanism["status"] not in ("complete", "upstream_unavailable")
                or set(mechanism.get("variants", {})) != available):
            raise ValueError("Numerical mechanism analysis is incomplete")
        if "prepared_sha256" in record:
            digest = hashlib.sha256(json.dumps(prepared, sort_keys=True, allow_nan=False).encode()).hexdigest()
            if record["prepared_sha256"] != digest or mechanism.get("prepared_sha256") != digest:
                raise ValueError("Confirmation and mechanism must bind the exact single preparation")
        checked = {**record, "summary": {name: record["summary"][name] for name in available}}
        totals = helpers.check_trials(checked, protocol["confirmation_seeds"], protocol["probe"]["pulse_amplitudes"])
        row = dict(zip(("population", "tau", "noise_tau", "seed"), key))
        row.update(delay=record["delay"], duration=record["duration"], group_size=len(record["group"]) if record.get("group") is not None else None, **totals)
        fit = prepared.get("anchor", {}).get("scalar_fit") or {}
        row["scalar_gain_at_bound"] = fit.get("gain_at_bound", False) if fit else None
        row["scalar_unconstrained_gain"] = fit.get("unconstrained_gain", fit.get("gain"))
        row["eligible_group"] = bool(record["group"]) if record.get("group") is not None else None
        native = record["summary"]["native"] or {}
        zero = passive[(record["tau"], record["noise_tau"], record["duration"])]["summary"]["zero_policy"]
        eps = protocol["metric_denominator_epsilon"]
        native_energy = native.get("paired_recovery", {}).get("normalized_position_energy")
        native_equilibrium = mechanism.get("variants", {}).get("native", {}).get("equilibrium", {})
        for name in VARIANTS:
            summary, mode = record["summary"][name] or {}, mechanism.get("variants", {}).get(name, {})
            if summary and (mode["status"] != "complete" or mode["settings"] != settings[name] or not mode["simulator_parity"]["passed"]):
                raise ValueError("Mechanism policy differs or its numerical parity failed")
            row[f"{name}_available"] = bool(summary)
            energy = summary.get("paired_recovery", {}).get("normalized_position_energy")
            row[f"{name}_recovery_energy"] = energy
            row[f"{name}_recovery_absolute"] = subtract(energy, native_energy)
            row[f"{name}_recovery_percent"] = percent(energy, native_energy, eps)
            row[f"{name}_recovery_passive_ratio"] = ratio(energy, zero["paired_recovery"]["normalized_position_energy"], eps)
            for field in ("position_rms", "action_rms", "position_mean", "action_mean", "saturation_fraction"):
                value = summary.get("sham", {}).get(field)
                row[f"{name}_{field}"] = value
                row[f"{name}_{field}_absolute"] = subtract(value, native.get("sham", {}).get(field))
                if field in ("position_rms", "action_rms"):
                    row[f"{name}_{field}_percent"] = percent(value, native.get("sham", {}).get(field), eps)
            row[f"{name}_position_passive_ratio"] = ratio(summary.get("sham", {}).get("position_rms"), zero["sham"]["position_rms"], eps)
            for arm in ("pulse", "sham"):
                for field in ("task_failure", "censored"):
                    row[f"{name}_{arm}_{field}"] = summary.get(arm, {}).get(field)
            for field in ("spectral_radius", "dominant_decay_rate_per_s", "dominant_frequency_hz", "transient_norm_max"):
                row[f"{name}_{field}"] = mode.get("linear", {}).get(field)
            row[f"{name}_locally_stable"] = mode.get("linear", {}).get("locally_asymptotically_stable")
            for field in ("position", "action"):
                value = mode.get("equilibrium", {}).get(field)
                row[f"{name}_equilibrium_{field}"] = value
                row[f"{name}_equilibrium_{field}_absolute"] = subtract(value, native_equilibrium.get(field))
            anchor = mechanism.get("anchor_identity", {}).get(name, {})
            for field in ("command_error", "jacobian_error_max", "full_map_error_max"):
                row[f"{name}_anchor_{field}"] = anchor.get(field)
            for phase, bank in (("calibration", prepared.get("calibration", {}).get("fixed_history", {})),
                                ("confirmation", record.get("fixed_history", {}))):
                fixed = bank.get("variants", {}).get(name) or {}
                wave = fixed.get("waveform", {})
                for field in ("residual_over_intervention", "relative_response_error", "response_cosine",
                              "sham_mean_change_from_native", "response_error_rms", "command_error_rms"):
                    row[f"{name}_{phase}_{field}"] = wave.get(field)
            correction = settings.get(name, {}).get("correction") or {}
            row[f"{name}_correction_gain"] = correction.get("gain", 1.) if summary else None
        for label, after, before in CONTRASTS:
            for metric in ("recovery_absolute", "recovery_percent"):
                row[f"{label}_{metric}"] = subtract(row[f"{after}_{metric}"], row[f"{before}_{metric}"])
        result.append(row)
    return result


def kernel_rows(preparations, period):
    """Signed local sensitivities in declared normalized physical coordinates."""
    result = []
    for key, record in sorted(preparations.items()):
        if "anchor" not in record or "weak_scalar" not in record["variants"] or "weak_kernel" not in record["variants"]:
            continue  # Explicitly counted as unavailable in population summaries.
        anchor = record["anchor"]
        native, weak = np.asarray(anchor["native_gradient"]), np.asarray(anchor["weak_gradient"])
        count = (len(native) - 1) // 3
        if native.shape != weak.shape or len(native) != 3 * count + 1:
            raise ValueError("Local response derivative does not match history triples plus held action")
        scalar = record["variants"]["weak_scalar"]["correction"]["gain"]
        kernel_setting = record["variants"]["weak_kernel"]["correction"]
        kernel = np.asarray(kernel_setting["kernel"])
        if kernel.shape != native.shape:
            raise ValueError("Kernel correction has the wrong coordinate dimension")
        for index in range(len(native)):
            held = index == len(native) - 1
            history_index = None if held else index // 3
            component = "held_action" if held else ("position", "velocity", "captured_action")[index % 3]
            row = dict(zip(("population", "tau", "noise_tau", "seed"), key))
            row.update(coordinate=index, component=component, history_index=history_index,
                       lag_seconds=0. if held else (count - 1 - history_index) * period,
                       native=float(native[index]), weak=float(weak[index]), scalar=float(scalar * weak[index]),
                       kernel=float(kernel_setting["gain"] * weak[index] + kernel[index]), correction=float(kernel[index]),
                       weak_minus_native=float(weak[index] - native[index]),
                       scalar_minus_native=float(scalar * weak[index] - native[index]))
            result.append(row)
    return result


def component_summaries(kernels, epsilon=1e-16):
    """Coordinate-dependent squared-norm allocation, never a causal share."""
    result = []
    for key in sorted({identity(row) for row in kernels}):
        pool = [row for row in kernels if identity(row) == key]
        selections = {
            "latest_position": [row for row in pool if row["component"] == "position" and row["lag_seconds"] == 0.],
            "older_positions": [row for row in pool if row["component"] == "position" and row["lag_seconds"] > 0.],
            "latest_velocity": [row for row in pool if row["component"] == "velocity" and row["lag_seconds"] == 0.],
            "older_velocities": [row for row in pool if row["component"] == "velocity" and row["lag_seconds"] > 0.],
            "captured_actions": [row for row in pool if row["component"] == "captured_action"],
            "held_action": [row for row in pool if row["component"] == "held_action"],
        }
        native = np.asarray([row["native"] for row in pool])
        weak = np.asarray([row["weak"] for row in pool])
        scalar = np.asarray([row["scalar"] for row in pool])
        total = float(np.sum((scalar - native) ** 2))
        row = dict(zip(("population", "tau", "noise_tau", "seed"), key))
        row.update(native_gradient_norm=float(np.linalg.norm(native)), weak_gradient_norm=float(np.linalg.norm(weak)),
                   weak_gradient_error_norm=float(np.linalg.norm(weak - native)),
                   scalar_gradient_error_norm=float(np.sqrt(total)),
                   scalar_residual_over_weak=ratio(float(np.linalg.norm(scalar - native)), float(np.linalg.norm(weak - native)), epsilon),
                   kernel_gradient_error_max=float(max(abs(item["kernel"] - item["native"]) for item in pool)))
        for component, values in selections.items():
            value = float(sum(item["scalar_minus_native"] ** 2 for item in values))
            row[f"{component}_residual_norm"] = float(np.sqrt(value))
            row[f"{component}_residual_squared_fraction"] = ratio(value, total, epsilon)
        result.append(row)
    return result


def save(fig, directory, name, title, note):
    fig.suptitle(title, fontsize=14)
    fig.text(.5, .014, note, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .065, 1, .94))
    fig.savefig(directory / f"{name}.png", dpi=170)
    path = directory / f"{name}.svg"
    fig.savefig(path, metadata={"Date": None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def curve(ax, rows, field, name, details=True):
    delays = sorted({row["delay"] for row in rows})
    x = np.asarray(delays) * 1000
    if details:
        for seed in sorted({row["seed"] for row in rows}):
            values = [stats(row[field] for row in rows if row["seed"] == seed and row["delay"] == delay)["mean"] for delay in delays]
            ax.plot(x, values, color=COLORS[name], alpha=.25, lw=.8)
    values = [stats(row[field] for row in rows if row["delay"] == delay)["mean"] for delay in delays]
    ax.plot(x, values, color=COLORS[name], lw=2., marker="o", ms=3, label=LABELS[name])
    ax.set_xticks(x)
    ax.set_xlabel("Physical delay (ms)")
    ax.grid(alpha=.2)


def panels(rows):
    return sorted({(row["population"], row["duration"]) for row in rows}, key=lambda x: (x[0] != "fresh", x[1]))


def recovery_plot(rows, directory):
    columns = panels(rows)
    fig, axes = plt.subplots(2, len(columns), figsize=(4.0 * len(columns), 7.4), squeeze=False)
    for column, (population, duration) in enumerate(columns):
        pool = [row for row in rows if (row["population"], row["duration"]) == (population, duration)]
        for index, metric in enumerate(("recovery_absolute", "recovery_percent")):
            ax = axes[index, column]
            for name in VARIANTS[1:]:
                curve(ax, pool, f"{name}_{metric}", name)
            ax.axhline(0, color="black", lw=.6)
            ax.set_ylabel("Absolute recovery-error effect" if index == 0 else "Recovery-error effect (%)")
            ax.set_title(f"{population.capitalize()} training; {duration:g} s horizon")
    axes[0, 0].legend(fontsize=7)
    save(fig, directory, "kernel_recovery", "Nested corrections tested on fresh disturbances",
         "Thin: initialization/data seed-block means. Thick: complete-population mean. Positive = worse than native. Fresh 4 s is primary; other panels secondary.")


def interaction_plot(inference, directory):
    ordered = sorted(inference, key=lambda r: (r["population"] != "fresh", r["duration"]))
    fig, axes = plt.subplots(1, len(ordered), figsize=(4.1 * len(ordered), 5.2), squeeze=False)
    for ax, record in zip(axes[0], ordered):
        for index, name in enumerate(VARIANTS[1:]):
            value = record["metrics"][f"{name}_recovery_absolute_interaction"]
            seeds = value["by_seed"]
            for si, item in enumerate(seeds):
                if number(item["value"]):
                    ax.scatter(index + (si - (len(seeds) - 1) / 2.) * .045, item["value"], s=22, color=COLORS[name], alpha=.7)
            if number(value["mean"]):
                ax.plot([index - .25, index + .25], [value["mean"]] * 2, color="black", lw=2.)
            if number(value["ci_low"]):
                ax.plot([index, index], [value["ci_low"], value["ci_high"]], color="black", lw=1.5)
                ax.plot([index - .06, index + .06], [value["ci_low"]] * 2, color="black", lw=1.)
                ax.plot([index - .06, index + .06], [value["ci_high"]] * 2, color="black", lw=1.)
        ax.set_xticks(range(4), [LABELS[name].replace(" ", "\n") for name in VARIANTS[1:]], fontsize=8)
        ax.set_title(f"{record['population'].capitalize()}; {record['duration']:g} s\n{record['seed_blocks']} seed blocks")
        ax.axhline(0, color="black", lw=.6)
        ax.set_ylabel("Effect at 100 ms − effect at 50 ms")
        ax.grid(axis="y", alpha=.2)
    save(fig, directory, "kernel_interactions", "Paired absolute recovery interactions",
         "Dots: seed means across four noise conditions. Black: mean; fresh-only whiskers: 95% percentile bootstrap over ten seed blocks. No trial/model-cell pseudoreplication.")


def principal_contrasts_plot(inference, directory):
    fresh = sorted([row for row in inference if row["population"] == "fresh"], key=lambda row: row["duration"])
    if not fresh:
        return
    fields = (("joint_weak_recovery_absolute_interaction", "Weakening timing interaction"),
              ("kernel_minus_scalar_recovery_absolute_interaction", "Full response − scalar timing interaction"))
    fig, axes = plt.subplots(len(fresh), 2, figsize=(12.4, 4.5 * len(fresh)), squeeze=False)
    colors = plt.get_cmap("tab10")
    for ri, record in enumerate(fresh):
        for ci, (field, title) in enumerate(fields):
            ax = axes[ri, ci]
            value = record["metrics"][field]
            blocks = value["by_seed"]
            for index, block in enumerate(blocks):
                if number(block["value"]):
                    ax.scatter(block["value"], index, color=colors(index % 10), s=28, zorder=3)
            mean_row = len(blocks) + .5
            if number(value["mean"]):
                ax.scatter(value["mean"], mean_row, color="black", marker="D", s=37, zorder=4)
            if number(value["ci_low"]):
                ax.plot([value["ci_low"], value["ci_high"]], [mean_row] * 2, color="black", lw=2.)
                ax.plot([value["ci_low"]] * 2, [mean_row - .15, mean_row + .15], color="black", lw=1.)
                ax.plot([value["ci_high"]] * 2, [mean_row - .15, mean_row + .15], color="black", lw=1.)
            ax.set_yticks([*range(len(blocks)), mean_row], [*[str(block["seed"]) for block in blocks], "Mean + CI"], fontsize=8)
            ax.invert_yaxis()
            ax.axvline(0., color="black", lw=.6)
            ax.axhline(len(blocks) - .15, color="gray", lw=.5, ls=":")
            ax.set_title(f"{record['duration']:g} s {'primary' if record['primary_population_horizon'] else 'secondary'}: {title}", fontsize=11)
            ax.set_ylabel("Initialization/data seed block")
            ax.set_xlabel("Absolute paired recovery-error interaction")
            ax.grid(axis="x", alpha=.2)
    save(fig, directory, "kernel_principal_contrasts", "Fresh training replication: the two principal paired contrasts",
         "Each colored point averages four noise conditions within one seed. Black: mean and 95% percentile CI from 10,000 seed-block resamples. Panels use separate scales; 12 s is secondary.")


def matching_plot(rows, directory):
    populations = sorted({row["population"] for row in rows}, key=lambda x: x != "fresh")
    duration = min(row["duration"] for row in rows)
    fig, axes = plt.subplots(2, len(populations), figsize=(6 * len(populations), 7.5), squeeze=False)
    for column, population in enumerate(populations):
        pool = [row for row in rows if row["population"] == population and row["duration"] == duration]
        for name in VARIANTS[1:]:
            curve(axes[0, column], pool, f"{name}_confirmation_residual_over_intervention", name)
            curve(axes[1, column], pool, f"{name}_equilibrium_position_absolute", name)
        axes[0, column].axhline(1, color="black", lw=.6)
        axes[1, column].axhline(0, color="black", lw=.6)
        axes[0, column].set_ylabel("Waveform residual / original weak effect")
        axes[1, column].set_ylabel("Equilibrium position shift from native")
        axes[0, column].set_title(population.capitalize())
    axes[0, 0].legend(fontsize=8)
    save(fig, directory, "kernel_matching", "What the frozen correction restores",
         "Waveforms use independent native confirmation histories. Equilibrium restoration and exact local response matching are construction checks, not independent performance evidence.")


def mode_plot(rows, directory):
    populations = sorted({row["population"] for row in rows}, key=lambda x: x != "fresh")
    duration = min(row["duration"] for row in rows)
    fig, axes = plt.subplots(2, len(populations), figsize=(6 * len(populations), 7.5), squeeze=False)
    for column, population in enumerate(populations):
        pool = [row for row in rows if row["population"] == population and row["duration"] == duration]
        for index, (metric, ylabel) in enumerate((("dominant_decay_rate_per_s", "Local dominant decay rate (1/s)"),
                                                ("spectral_radius", "Dominant pole magnitude"))):
            for name in VARIANTS:
                for model in sorted({identity(row) for row in pool}):
                    subset = sorted([row for row in pool if identity(row) == model], key=lambda row: row["delay"])
                    axes[index, column].plot([row["delay"] * 1000 for row in subset],
                                             [row[f"{name}_{metric}"] for row in subset],
                                             color=COLORS[name], lw=.5, alpha=.12)
                curve(axes[index, column], pool, f"{name}_{metric}", name)
            axes[index, column].set_ylabel(ylabel)
            axes[index, column].axhline(float(index), color="black", lw=.6)
        axes[0, column].set_title(population.capitalize())
        delays = sorted({row["delay"] for row in pool})
        population_size = len({identity(row) for row in pool})
        labels = []
        for name in VARIANTS:
            counts = [sum(row[f"{name}_locally_stable"] is False for row in pool if row["delay"] == delay) for delay in delays]
            labels.append(f"{LABELS[name]}: {' → '.join(map(str, counts))}/{population_size}")
        handles, _ = axes[0, column].get_legend_handles_labels()
        axes[0, column].legend(handles, labels, fontsize=7, title="Unstable models at 50 → 100 ms", title_fontsize=8)
    save(fig, directory, "kernel_local_validation", "Complete-loop local modes: validation of the correction",
         "Local matching is by construction. Faint: every model, including unstable equilibria; thin: seed-block means; thick: complete-population means. No global stability claim.")


def components_plot(kernels, directory):
    populations = sorted({row["population"] for row in kernels}, key=lambda x: x != "fresh")
    fig, axes = plt.subplots(len(populations), 4, figsize=(16, 3.7 * len(populations)), squeeze=False)
    for ri, population in enumerate(populations):
        for ci, component in enumerate(("position", "velocity", "captured_action", "held_action")):
            pool = [row for row in kernels if row["population"] == population and row["component"] == component]
            lags = sorted({row["lag_seconds"] for row in pool})
            for field, name in (("native", "native"), ("weak", "joint_weak"), ("scalar", "weak_scalar")):
                for seed in sorted({row["seed"] for row in pool}):
                    values = [stats(row[field] for row in pool if row["seed"] == seed and row["lag_seconds"] == lag)["mean"] for lag in lags]
                    axes[ri, ci].plot(np.asarray(lags) * 1000, values, color=COLORS[name], alpha=.2, lw=.7, marker="." if ci == 3 else None)
                values = [stats(row[field] for row in pool if row["lag_seconds"] == lag)["mean"] for lag in lags]
                axes[ri, ci].plot(np.asarray(lags) * 1000, values, color=COLORS[name], label=LABELS[name], lw=2, marker="o", ms=3)
            axes[ri, ci].set_title(f"{population.capitalize()}: {component.replace('_', ' ')}")
            axes[ri, ci].set_xlabel("Observation age (ms)" if ci < 3 else "Current command")
            axes[ri, ci].set_ylabel("Command derivative\n(normalized input)")
            if ci == 3:
                axes[ri, ci].set_xticks([0], ["held action"])
            axes[ri, ci].axhline(0, color="black", lw=.6)
            axes[ri, ci].grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    save(fig, directory, "kernel_components", "Which input sensitivities remain after scalar compensation?",
         "Frozen 50 ms native equilibrium. Coordinates: q/0.1, τv/0.1, captured action/0.1, held action/0.1. Full correction matches native; slopes are local, coordinate dependent.")


def tradeoff_plot(rows, directory):
    columns = panels(rows)
    fig, axes = plt.subplots(3, len(columns), figsize=(4 * len(columns), 9.8), squeeze=False)
    fields = (("position_rms_percent", "Sham position RMS effect (%)"),
              ("action_rms_percent", "Sham action RMS effect (%)"),
              ("recovery_passive_ratio", "Recovery error / passive"))
    for column, (population, duration) in enumerate(columns):
        pool = [row for row in rows if (row["population"], row["duration"]) == (population, duration)]
        for index, (field, ylabel) in enumerate(fields):
            for name in (VARIANTS if index == 2 else VARIANTS[1:]):
                curve(axes[index, column], pool, f"{name}_{field}", name)
            axes[index, column].set_ylabel(ylabel)
            axes[index, column].axhline(1. if index == 2 else 0., color="black", lw=.6)
        axes[0, column].set_title(f"{population.capitalize()}; {duration:g} s")
    axes[0, 0].legend(fontsize=7)
    save(fig, directory, "kernel_task_tradeoffs", "Recovery, background-noise regulation and action effort",
         "Each noisy variant is compared with its native controller; passive reference uses identical noise tapes. Complete failed trials remain included; censored means stay unavailable.")


def diagnostic_rows(records):
    result = []
    for record in sorted(records, key=cell):
        for name, values in record.get("correction_on_own_histories", {}).items():
            for trial in values["rollouts"]:
                row = dict(zip(("population", "tau", "noise_tau", "seed"), identity(record)))
                row.update(delay=record["delay"], duration=record["duration"], variant=name,
                           noise_seed=trial["noise_seed"], amplitude=trial["amplitude"],
                           decision_count=trial["decision_count"], warmup_decision_count=trial["warmup_decision_count"],
                           post_correction_clipping_fraction=trial["post_correction_clipping_fraction"])
                for component, moments in trial["components"].items():
                    row.update({f"{component}_{moment}": value for moment, value in moments.items()})
                correction = record["frozen_settings"][name].get("correction", {})
                if not correction or correction.get("identity", False):
                    for component in ("older_position", "latest_position", "older_velocity", "latest_velocity",
                                      "captured_action_history", "current_action"):
                        row.update({f"kernel_{component}_{moment}": 0. for moment in ("rms", "mean", "max_abs")})
                result.append(row)
    return result


def correction_plot(diagnostics, directory):
    """Observed correction terms, retaining cancellation between components."""
    primary = min(row["duration"] for row in diagnostics)
    panels = sorted({(row["population"], row["delay"]) for row in diagnostics}, key=lambda x: (x[0] != "fresh", x[1]))
    fields = (("offset", "Offset"), ("gain", "Scalar gain"),
              ("kernel_latest_position", "Latest q"), ("kernel_older_position", "Older q"),
              ("kernel_latest_velocity", "Latest v"), ("kernel_older_velocity", "Older v"),
              ("kernel_captured_action_history", "Past actions"), ("kernel_current_action", "Held action"),
              ("kernel", "Full kernel"), ("raw_command_change", "Total correction"))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.1 * len(panels), 5.4), squeeze=False)
    for ax, (population, delay) in zip(axes[0], panels):
        pool = [row for row in diagnostics if (row["population"], row["delay"], row["duration"], row["variant"])
                == (population, delay, primary, "weak_kernel")]
        seed_values = {}
        for seed in sorted({row["seed"] for row in pool}):
            seed_values[seed] = [stats(row.get(f"{field}_rms") for row in pool if row["seed"] == seed)["mean"] for field, _ in fields]
        means = [stats(values[index] for values in seed_values.values())["mean"] for index in range(len(fields))]
        x = np.arange(len(fields))
        valid = [number(value) for value in means]
        ax.bar(x[valid], np.asarray(means, dtype=object)[valid].astype(float), color="#7062af", alpha=.75)
        for values in seed_values.values():
            ax.plot(x, values, color="black", lw=.5, alpha=.2, marker=".", ms=2)
        ax.set_xticks(x, [label for _, label in fields], rotation=65, ha="right", fontsize=8)
        ax.set_ylabel("Physical command contribution RMS")
        ax.set_title(f"{population.capitalize()}; {delay * 1000:g} ms")
        ax.grid(axis="y", alpha=.2)
    save(fig, directory, "kernel_own_history_contributions", "How strongly does the correction act on its own trajectories?",
         "Four-second trials including startup. Mean per-trial RMS; thin lines: seed-block means. Components can cancel, so their RMS values do not add. Prefix diagnostics are not censored outcome estimates.")


def load_inputs(directory):
    """Reject changed artifacts, undeclared cells and altered configuration."""
    manifest = read(directory / "run_manifest.json")
    stage_names = ("discovery_bank", "fresh_train", "fresh_discovery", "prepare", "passive", "confirm", "mechanism")
    if set(manifest["stages"]) != set(stage_names):
        raise ValueError("All seven experiment stages must be sealed before plotting")
    for category in ("scientific_sources_sha256", "completed_sha256"):
        for relative, expected in manifest[category].items():
            if sha(ROOT / relative) != expected:
                raise ValueError(f"Sealed source or artifact changed: {relative}")
    protocol, base = read(directory / "config.json"), read(directory / "base_config.json")
    if protocol != manifest["protocol_config"] or base != manifest["base_config"]:
        raise ValueError("Configuration differs from frozen run manifest")
    models = ([tuple(row[key] for key in ("population", "tau", "noise_tau", "seed")) for row in protocol["model_subset"]]
              if protocol.get("model_subset") else [(population, tau, noise, seed)
              for population, seeds in (("existing", base["seeds"]), ("fresh", protocol["fresh_seeds"]))
              for tau, noise, seed in itertools.product(protocol["plant_taus"], base["noise_taus"], seeds)])
    fresh = [model for model in models if model[0] == "fresh"]
    expected = {
        "discovery_bank": {(model[1],) for model in fresh},
        "fresh_train": set(fresh), "fresh_discovery": set(fresh), "prepare": set(models),
        "confirm": {(*model, delay, duration) for model in models for delay in protocol["delays"] for duration in protocol["confirmation_durations"]},
        "mechanism": {(*model, delay) for model in models for delay in protocol["delays"]},
        "passive": {(model[1], model[2], duration) for model in models for duration in protocol["confirmation_durations"]},
    }
    stages = {}
    for stage in stage_names:
        paths = sorted((directory / stage).glob("*.json"))
        for path in paths:
            if manifest["completed_sha256"].get(str(path.relative_to(ROOT))) != sha(path):
                raise ValueError(f"Unsealed record in {stage}")
        values = [read(path) for path in paths]
        if stage == "discovery_bank":
            keys = [(row["tau"],) for row in values]
        elif stage == "passive":
            keys = [(row["tau"], row["noise_tau"], row["duration"]) for row in values]
        elif stage == "confirm":
            keys = [cell(row) for row in values]
        elif stage == "mechanism":
            keys = [(*identity(row), row["delay"]) for row in values]
        else:
            keys = [identity(row) for row in values]
        if len(set(keys)) != len(keys) or set(keys) != expected[stage]:
            raise ValueError(f"Declared population has missing, extra or duplicate records: {stage}")
        stages[stage] = dict(zip(keys, values))
    # Derived labels expose, rather than change, the frozen config population.
    protocol = {**protocol, "existing_seeds": sorted({model[3] for model in models if model[0] == "existing"}),
                "fresh_seeds": sorted({model[3] for model in fresh}),
                "noise_taus": sorted({model[2] for model in models}), "primary_duration": protocol["probe"]["duration"]}
    inputs = [directory / "run_manifest.json", *(ROOT / relative for relative in manifest["completed_sha256"])]
    return protocol, base, stages, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/kernel_rescue")
    args = parser.parse_args()
    directory = args.results.resolve()
    protocol, base, stages, inputs = load_inputs(directory)
    for record in stages["passive"].values():
        helpers.check_trials(record, protocol["confirmation_seeds"], protocol["probe"]["pulse_amplitudes"])
    for record in stages["prepare"].values():
        if "calibration" in record:
            helpers.check_trials(record["calibration"], protocol["calibration_seeds"], protocol["probe"]["pulse_amplitudes"])
    rows = flatten(stages["confirm"].values(), stages["prepare"], stages["mechanism"], stages["passive"], protocol)
    endpoints = interactions(rows, protocol["primary_low_delay"], protocol["primary_high_delay"])
    seed_interactions, inference = block_inference(endpoints, protocol)
    kernels = kernel_rows(stages["prepare"], base["period"])
    components = component_summaries(kernels)
    diagnostics = diagnostic_rows(stages["confirm"].values())
    tables = {"model_metrics.csv": rows, "model_interactions.csv": endpoints,
              "population_metrics.csv": aggregate(rows, ("population", "tau", "delay", "duration")),
              "seed_metrics.csv": aggregate(rows, ("population", "tau", "seed", "delay", "duration")),
              "seed_interactions.csv": seed_interactions, "kernel_coefficients.csv": kernels,
              "kernel_components.csv": components, "own_history_corrections.csv": diagnostics}
    for name, values in tables.items():
        helpers.write_csv(directory / name, values)
    summary = {"populations": {population: {"model_conditions": len({identity(row) for row in rows if row["population"] == population}),
                    "initialization_seeds": sorted({row["seed"] for row in rows if row["population"] == population}),
                    "eligible_groups": sum(bool(record.get("group")) for key, record in stages["prepare"].items() if key[0] == population),
                    "unavailable_preparations": sum(bool(record.get("unavailable_controls")) for key, record in stages["prepare"].items() if key[0] == population),
                    "preparation_status_counts": {status: sum(record.get("status") == status for key, record in stages["prepare"].items() if key[0] == population)
                        for status in sorted({record.get("status", "unknown") for key, record in stages["prepare"].items() if key[0] == population})}}
                    for population in sorted({row["population"] for row in rows})},
               "primary_duration": protocol["primary_duration"], "delays": protocol["delays"],
               "model_delay_horizon_records": len(rows), "population_metrics": tables["population_metrics.csv"],
               "inference": inference, "confirmation_physical_rollouts": sum(row["unique_rollouts"] for row in rows),
               "confirmation_task_failures": sum(row["task_failures"] for row in rows),
               "confirmation_censored": sum(row["censored"] for row in rows),
               "kernel_component_model_coverage": len(components),
               "coprimary_fresh_4s_metrics": ["joint_weak_recovery_absolute_interaction", "kernel_minus_scalar_recovery_absolute_interaction"],
               "interpretation": "Fresh 4 s training replication primary; existing models and 12 s secondary. Percentile bootstrap resamples entire initialization/data seed blocks, conditional on the fixed task grid and shared probe tapes. Missing required cells invalidate whole-population inference. Exact local matching is validation by construction."}
    write_json(directory / "summary.json", summary)
    plot_dir = directory / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for function, values in ((recovery_plot, rows), (interaction_plot, inference), (matching_plot, rows),
                             (mode_plot, rows), (tradeoff_plot, rows), (principal_contrasts_plot, inference)):
        function(values, plot_dir)
    if len(components) == len(stages["prepare"]):
        components_plot(kernels, plot_dir)
        correction_plot(diagnostics, plot_dir)
    outputs = [directory / name for name in (*tables.keys(), "summary.json") if (directory / name).exists()]
    outputs += sorted(plot_dir.glob("kernel_*.png")) + sorted(plot_dir.glob("kernel_*.svg"))
    manifest = {"plot_sources_sha256": {str(Path(__file__).resolve().relative_to(ROOT)): sha(__file__),
                                        "experiments/plot_delay_sweep.py": sha(ROOT / "experiments/plot_delay_sweep.py")},
                "inputs_sha256": {str(path.relative_to(ROOT)): sha(path) for path in inputs},
                "outputs_sha256": {str(path.relative_to(ROOT)): sha(path) for path in outputs}}
    write_json(directory / "plot_manifest.json", manifest)
    print(json.dumps({"model_conditions": len(stages["prepare"]), "confirmation_records": len(rows),
                      "inputs": len(inputs), "outputs": len(outputs), "results": str(directory)}))


if __name__ == "__main__":
    main()
