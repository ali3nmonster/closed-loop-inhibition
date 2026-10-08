"""Audited paired summaries of the targeted full-loop mechanism experiment."""

import argparse
from copy import deepcopy
import importlib.util
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("_delay_plot_helpers", ROOT / "experiments/plot_delay_sweep.py")
helpers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helpers)
VARIANTS = ("native", "joint_weak", "joint_strong", "gain_matched", "weak_rescue")
CONTROLS = ("gain_matched", "weak_rescue")
LABELS = {"native": "Native", "joint_weak": "Weakened", "joint_strong": "Strengthened",
          "gain_matched": "Gain matched", "weak_rescue": "Weak + gain rescue"}
COLORS = {"native": "#333333", "joint_weak": "#bb3936", "joint_strong": "#2478ad",
          "gain_matched": "#ac7611", "weak_rescue": "#7850aa"}
SEED_STYLES = ("--", ":", "-.")
matplotlib.rcParams["svg.hashsalt"] = "complete_loop_mechanism_v1"
read, write_json, sha = helpers.read, helpers.write_json, helpers.sha
stats, number, subtract, ratio, percent = helpers.stats, helpers.number, helpers.subtract, helpers.ratio, helpers.percent


def identity(record):
    return record["tau"], record["noise_tau"], int(record["seed"])


def cell(record):
    return (*identity(record), record["delay"])


def validate_trials(record, seeds, amplitudes):
    """Unavailable policies remain null while all available raw trials are checked."""
    cleaned = {**record, "summary": {key: value for key, value in record["summary"].items() if value is not None}}
    nulls = {key for key, value in record["summary"].items() if value is None}
    if nulls != set(record.get("unavailable_controls", {})):
        raise ValueError("Null summaries and unavailable controls disagree")
    return helpers.check_trials(cleaned, seeds, amplitudes)


def aggregate(rows, by):
    result = []
    keys = ("tau", "noise_tau", "seed", "delay", "comparison", "low_delay", "high_delay")
    fields = sorted({key for row in rows for key in row if key not in keys})
    for identity_value in sorted({tuple(row[key] for key in by) for row in rows}):
        pool = [row for row in rows if tuple(row[key] for key in by) == identity_value]
        grouped = dict(zip(by, identity_value))
        for field in fields:
            for name, value in stats(row.get(field) for row in pool).items():
                grouped[f"{field}_{name}"] = value
        result.append(grouped)
    return result


def flatten(confirmations, preparations, mechanisms, passive, protocol):
    rows = []
    invariant = {}
    for record in sorted(confirmations, key=cell):
        key, model = cell(record), identity(record)
        prepared, mechanism = preparations[key], mechanisms[key]
        settings = record["frozen_settings"]
        if settings != prepared["variants"] or record["group"] != prepared["group"]:
            raise ValueError("Confirmation settings differ from independently sealed calibration")
        signature = json.dumps({"variants": {name: settings[name] for name in VARIANTS[:3]},
                                "group": record["group"]}, sort_keys=True)
        if model in invariant and invariant[model] != signature:
            raise ValueError("Inherited native, weak or strong policy changed across delays")
        invariant[model] = signature
        if set(mechanism["variants"]) != set(settings):
            raise ValueError("Mechanism and confirmation policy populations differ")
        if mechanism["status"] != "complete":
            raise ValueError("Numerical mechanism failure prevents publication")
        for name, value in mechanism["variants"].items():
            if value["status"] != "complete" or not value["simulator_parity"]["passed"]:
                raise ValueError("Numerical mechanism or simulator-parity failure prevents publication")
            if value["settings"] != settings[name]:
                raise ValueError("Mechanism analyzed a different policy setting")
        row = dict(tau=model[0], noise_tau=model[1], seed=model[2], delay=record["delay"],
                   group_size=len(record["group"]),
                   **validate_trials(record, protocol["confirmation_seeds"], protocol["probe"]["pulse_amplitudes"]))
        native = record["summary"]["native"]
        zero = passive[model[:2]]["summary"]["zero_policy"]
        epsilon = protocol["metric_denominator_epsilon"]
        native_energy = native["paired_recovery"]["normalized_position_energy"]
        native_equilibrium = mechanism["variants"]["native"]["equilibrium"]
        for name in VARIANTS:
            summary = record["summary"][name]
            mode = mechanism["variants"].get(name, {})
            equilibrium, linear = mode.get("equilibrium", {}), mode.get("linear", {})
            energy = summary["paired_recovery"]["normalized_position_energy"] if summary else None
            row[f"{name}_available"] = summary is not None
            row[f"{name}_recovery_energy"] = energy
            row[f"{name}_recovery_absolute"] = subtract(energy, native_energy)
            row[f"{name}_recovery_percent"] = percent(energy, native_energy, epsilon)
            row[f"{name}_recovery_passive_ratio"] = ratio(energy, zero["paired_recovery"]["normalized_position_energy"], epsilon)
            for field in ("position_rms", "action_rms", "position_mean", "action_mean", "saturation_fraction"):
                value = summary["sham"][field] if summary else None
                row[f"{name}_{field}"] = value
                row[f"{name}_{field}_absolute"] = subtract(value, native["sham"][field])
            row[f"{name}_position_passive_ratio"] = ratio(row[f"{name}_position_rms"], zero["sham"]["position_rms"], epsilon)
            for arm in ("sham", "pulse"):
                for field in ("task_failure", "censored"):
                    row[f"{name}_{arm}_{field}"] = summary[arm][field] if summary else None
            for field in ("spectral_radius", "dominant_decay_rate_per_s", "dominant_frequency_hz", "dominant_damping_ratio", "transient_norm_max"):
                row[f"{name}_{field}"] = linear.get(field)
            row[f"{name}_locally_stable"] = linear.get("locally_asymptotically_stable")
            row[f"{name}_current_action_age"] = mode.get("current_action_age")
            row[f"{name}_equilibrium_state_residual_max"] = equilibrium.get("state_residual_max")
            row[f"{name}_equilibrium_clipped"] = equilibrium.get("clipped")
            impulse_data = mode.get("impulses", {})
            row[f"{name}_verification_relative_error_max"] = impulse_data.get("verification_relative_error_max")
            large = [item for item in impulse_data.get("records", []) if abs(item["amplitude"]) == .02]
            for field in ("sampled_recovery_integral_normalized", "relative_linear_prediction_error",
                          "post_pulse_sign_crossings", "settling_time_after_pulse_2pct_peak", "clipped_decision_count"):
                row[f"{name}_large_pulse_mean_{field}"] = stats(item[field] for item in large)["mean"]
            row[f"{name}_large_pulse_settled_fraction"] = stats(
                item["settling_time_after_pulse_2pct_peak"] is not None for item in large)["mean"]
            for field in ("position", "action"):
                row[f"{name}_equilibrium_{field}"] = equilibrium.get(field)
                row[f"{name}_equilibrium_{field}_absolute"] = subtract(equilibrium.get(field), native_equilibrium[field])
            for field in ("gain", "offset", "center"):
                row[f"{name}_{field}"] = settings.get(name, {}).get(field)
            for phase, bank in (("calibration", prepared["calibration"]["fixed_history"]),
                                ("confirmation", record["fixed_history"])):
                fixed = bank.get("variants", {}).get(name) or {}
                waveform = fixed.get("waveform", {})
                for field in ("residual_over_intervention", "relative_response_error", "response_cosine",
                              "sham_mean_change_from_native", "response_error_rms"):
                    row[f"{name}_{phase}_{field}"] = waveform.get(field)
            row[f"{name}_parity_error"] = mode.get("simulator_parity", {}).get("state_error_max_normalized")
        for name in CONTROLS:
            row[f"{name}_minus_weak_absolute"] = subtract(row[f"{name}_recovery_absolute"], row["joint_weak_recovery_absolute"])
        rows.append(row)
    return rows


def interaction_summary(endpoints):
    result = {}
    for label in dict.fromkeys(row["comparison"] for row in endpoints):
        pool = [row for row in endpoints if row["comparison"] == label]
        values = {"low_delay": pool[0]["low_delay"], "high_delay": pool[0]["high_delay"], "models": len(pool), "variants": {}}
        for name in VARIANTS[1:]:
            fields = {}
            for field in ("recovery_absolute_interaction", "recovery_percent_interaction"):
                data = [row[f"{name}_{field}"] for row in pool]
                fields[field] = {**stats(data), "positive_models": sum(number(v) and v > 0 for v in data),
                                 "negative_models": sum(number(v) and v < 0 for v in data)}
            for field in ("helpful_to_harmful", "harmful_to_helpful"):
                data = [row[f"{name}_{field}"] for row in pool]
                fields[field] = {"count": sum(value is True for value in data), "valid_n": sum(value is not None for value in data), "n": len(data)}
            if name in CONTROLS:
                fields["minus_weak_interaction"] = stats(row[f"{name}_minus_weak_interaction"] for row in pool)
            values["variants"][name] = fields
        result[label] = values
    return result


def interactions(rows, comparisons):
    result = []
    for label, low, high in comparisons:
        for model in sorted({identity(row) for row in rows}):
            lookup = {row["delay"]: row for row in rows if identity(row) == model}
            if low not in lookup or high not in lookup:
                raise ValueError("Required interaction endpoints are absent")
            lo, hi = lookup[low], lookup[high]
            row = dict(zip(("tau", "noise_tau", "seed"), model))
            row.update(comparison=label, low_delay=low, high_delay=high)
            for name in VARIANTS[1:]:
                for metric in ("recovery_absolute", "recovery_percent", "spectral_radius", "dominant_decay_rate_per_s"):
                    low_value, high_value = lo[f"{name}_{metric}"], hi[f"{name}_{metric}"]
                    if metric in ("spectral_radius", "dominant_decay_rate_per_s"):
                        low_value = subtract(low_value, lo[f"native_{metric}"])
                        high_value = subtract(high_value, hi[f"native_{metric}"])
                    row[f"{name}_{metric}_interaction"] = subtract(high_value, low_value)
                lval, hval = lo[f"{name}_recovery_absolute"], hi[f"{name}_recovery_absolute"]
                available = number(lval) and number(hval)
                row[f"{name}_helpful_to_harmful"] = bool(lval < 0 < hval) if available else None
                row[f"{name}_harmful_to_helpful"] = bool(lval > 0 > hval) if available else None
            for name in CONTROLS:
                row[f"{name}_minus_weak_interaction"] = subtract(
                    row[f"{name}_recovery_absolute_interaction"], row["joint_weak_recovery_absolute_interaction"])
            result.append(row)
    return result


def frequency_rows(mechanisms):
    rows = []
    for record in sorted(mechanisms.values(), key=cell):
        for name in VARIANTS:
            value = record["variants"].get(name)
            if value is None:
                continue
            for response in value["linear"]["frequency_response"]:
                rows.append({"tau": record["tau"], "noise_tau": record["noise_tau"], "seed": record["seed"],
                             "delay": record["delay"], "variant": name,
                             "model_locally_stable": value["linear"]["locally_asymptotically_stable"], **response})
    return rows


def frequency_curve(pool, population):
    """Formal resolvents include unstable models; no stability-based selection."""
    hz = sorted({row["frequency_hz"] for row in pool})
    geometric = []
    for frequency in hz:
        selected = [row for row in pool if row["frequency_hz"] == frequency]
        values = [row.get("magnitude") for row in selected]
        valid = len(values) == population and all(number(value) and value > 0 for value in values)
        geometric.append(float(np.exp(np.mean(np.log(values)))) if valid else None)
    unstable = len({identity(row) for row in pool if row["model_locally_stable"] is False})
    return hz, geometric, unstable


def save(fig, directory, name, title, note):
    fig.suptitle(title, fontsize=14)
    fig.text(.5, .015, note, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, .94))
    fig.savefig(directory / f"{name}.png", dpi=170)
    path = directory / f"{name}.svg"
    fig.savefig(path, metadata={"Date": None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def values_for(rows, field, delays):
    return [stats(row.get(field) for row in rows if row["delay"] == delay)["mean"] for delay in delays]


def plot_curve(ax, rows, field, name, *, details=True):
    delays = sorted({row["delay"] for row in rows})
    x = np.asarray(delays) * 1000
    if details:
        for model in sorted({identity(row) for row in rows}):
            pool = [row for row in rows if identity(row) == model]
            ax.plot(x, values_for(pool, field, delays), color=COLORS[name], alpha=.17, lw=.7)
        for index, seed in enumerate(sorted({row["seed"] for row in rows})):
            pool = [row for row in rows if row["seed"] == seed]
            ax.plot(x, values_for(pool, field, delays), color=COLORS[name], ls=SEED_STYLES[index % 3], lw=1.)
    ax.plot(x, values_for(rows, field, delays), color=COLORS[name], marker="o", ms=3, lw=2., label=LABELS[name])
    ax.set_xticks(x)
    ax.set_xlabel("Physical delay (ms)")
    ax.grid(alpha=.18)


def recovery_plot(rows, output):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
    for column, name in enumerate(VARIANTS[1:]):
        for index, metric in enumerate(("recovery_absolute", "recovery_percent")):
            ax = axes[index, column]
            plot_curve(ax, rows, f"{name}_{metric}", name)
            ax.axhline(0, color="black", lw=.6)
            ax.set_title(LABELS[name])
            ax.set_ylabel("Absolute error effect" if index == 0 else "Recovery effect (%)")
    save(fig, output, "loop_recovery", "Recovery relative to the native controller",
         "Thin: all model conditions. Dashed: averages within initialization seed. Thick: complete-population mean. Positive = worse.")


def modes_plot(rows, output):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    fields = (("spectral_radius", "Dominant pole magnitude", 1.),
              ("dominant_decay_rate_per_s", "Dominant decay rate (1/s)", 0.),
              ("dominant_frequency_hz", "Dominant frequency (Hz)", None))
    for ax, (field, ylabel, reference) in zip(axes, fields):
        for name in VARIANTS:
            plot_curve(ax, rows, f"{name}_{field}", name)
        ax.set_ylabel(ylabel)
        if reference is not None:
            ax.axhline(reference, color="black", lw=.7)
    axes[0].legend(fontsize=8)
    save(fig, output, "loop_modes", "Local modes of the complete controller–plant map",
         "Each policy is linearized at its own equilibrium. Thin: all conditions; dashed: seed means; thick: complete-population mean.")


def matching_plot(rows, output):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for index, phase in enumerate(("calibration", "confirmation")):
        for name in CONTROLS:
            plot_curve(axes[index], rows, f"{name}_{phase}_residual_over_intervention", name)
        axes[index].set_ylabel("Waveform residual / original weak effect")
        axes[index].set_title(phase.capitalize())
        axes[index].axhline(1, color="black", lw=.7)
    for name in CONTROLS:
        plot_curve(axes[2], rows, f"{name}_gain", name)
    axes[2].set_ylabel("Calibrated scalar gain")
    axes[2].axhline(1, color="black", lw=.7)
    axes[0].legend(fontsize=8)
    save(fig, output, "loop_gain_matching", "How much of the command waveform does scalar gain explain?",
         "Calibration and confirmation use separate native histories. Controls are frozen before confirmation; no post-confirmation selection.")


def interaction_plot(endpoints, output):
    labels = list(dict.fromkeys(row["comparison"] for row in endpoints))
    fig, axes = plt.subplots(1, len(labels), figsize=(5.1 * len(labels), 4.8), squeeze=False)
    seeds = sorted({row["seed"] for row in endpoints})
    noises = sorted({row["noise_tau"] for row in endpoints})
    seed_colors = ("#1b6c93", "#cb5c32", "#647e26")
    for ax, comparison in zip(axes[0], labels):
        pool = [row for row in endpoints if row["comparison"] == comparison]
        for index, name in enumerate(VARIANTS[1:]):
            field = f"{name}_recovery_absolute_interaction"
            for si, seed in enumerate(seeds):
                subset = [row for row in pool if row["seed"] == seed]
                for row in subset:
                    value = row[field]
                    if number(value):
                        offset = (si - 1) * .18 + (noises.index(row["noise_tau"]) - 1.5) * .025
                        ax.scatter(index + offset, value, s=20, color=seed_colors[si % 3], alpha=.6)
                mean = stats(row[field] for row in subset)["mean"]
                if mean is not None:
                    ax.scatter(index + (si - 1) * .18, mean, marker="D", s=40, color=seed_colors[si % 3], edgecolor="white", lw=.7)
            total = stats(row[field] for row in pool)["mean"]
            if total is not None:
                ax.plot([index - .36, index + .36], [total, total], color="black", lw=2)
        example = pool[0]
        title = "Primary comparison" if comparison == "primary" else "Secondary: same action-age phase"
        ax.set_title(f"{title}\n{example['low_delay'] * 1000:g} → {example['high_delay'] * 1000:g} ms", fontsize=11)
        ax.set_xticks(range(4), [LABELS[name].replace(" ", "\n") for name in VARIANTS[1:]], fontsize=8)
        ax.set_ylabel("High-delay effect − low-delay effect")
        ax.axhline(0, color="black", lw=.7)
        ax.grid(axis="y", alpha=.2)
    from matplotlib.lines import Line2D
    axes[0, 0].legend(handles=[Line2D([], [], color=seed_colors[index % 3], marker="D", ls="none",
                                     label=f"Initialization {seed}") for index, seed in enumerate(seeds)],
                      fontsize=7, loc="lower left")
    save(fig, output, "loop_interactions", "Paired absolute recovery interactions",
         "Dots: all model conditions. Colors and diamonds: initialization seeds and their means. Black: complete-population mean. Same-phase pairs are secondary.")


def means_plot(rows, output):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fields = (("equilibrium_position", "Equilibrium position"),
              ("equilibrium_position_absolute", "Equilibrium position shift from native"),
              ("position_mean_absolute", "Noisy-loop mean position shift from native"),
              ("action_mean_absolute", "Noisy-loop mean action shift from native"))
    for ax, (field, label) in zip(axes.flat, fields):
        for name in VARIANTS:
            plot_curve(ax, rows, f"{name}_{field}", name)
        ax.set_ylabel(label)
        ax.axhline(0, color="black", lw=.6)
    axes[0, 0].legend(fontsize=8)
    save(fig, output, "loop_means", "Equilibrium and own-loop mean shifts",
         "Own-loop means come from fresh sham trials. Delay phase changes the truthful current-action-age input, even with the explicit delay cue fixed.")


def ringdown_plot(mechanisms, output):
    selected = {key[-1]: value for key, value in mechanisms.items() if key[:3] == (.2, .01, 11) and key[-1] in (0., .1)}
    if set(selected) != {0., .1}:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for column, delay in enumerate((0., .1)):
        for index, amplitude in enumerate((-.02, .02)):
            ax = axes[index, column]
            for name in VARIANTS:
                value = selected[delay]["variants"].get(name)
                if value is None:
                    continue
                impulse = next(row for row in value["impulses"]["records"] if row["amplitude"] == amplitude)
                t = value["impulses"]["times"][:len(impulse["position_response"])]
                ax.plot(t, impulse["position_response"], color=COLORS[name], label=LABELS[name])
                ax.plot(t, impulse["linear_position_response"], color=COLORS[name], ls=":", alpha=.7)
            ax.axhline(0, color="black", lw=.5)
            ax.set_title(f"Delay {delay * 1000:g} ms; force pulse {amplitude:+g}")
            ax.set_xlabel("Time from pulse onset (s)")
            ax.set_ylabel("Position relative to own equilibrium")
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    save(fig, output, "loop_ringdown", "Illustrative example: training noise τ = 10 ms, initialization seed 11",
         "Solid: nonlinear complete-loop map; dotted: local linear prediction. Zero background, mature histories, 100 ms pulse. This is one model condition.")


def frequency_plot(frequencies, rows, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    population = len({identity(row) for row in rows})
    unstable_notes = []
    for column, delay in enumerate((0., .1)):
        for name in VARIANTS:
            pool = [row for row in frequencies if row["delay"] == delay and row["variant"] == name]
            hz, geometric, unstable = frequency_curve(pool, population)
            label = f"{LABELS[name]} ({unstable}/{population} unstable)"
            axes[0, column].plot(hz, geometric, color=COLORS[name], marker="o", ms=3, label=label)
            if unstable:
                unstable_notes.append(f"{LABELS[name]} {unstable}/{population} at {delay * 1000:g} ms")
            example = sorted([row for row in pool if identity(row) == (.2, .01, 11)], key=lambda row: row["frequency_hz"])
            if example:
                phases = [row.get("phase_degrees") for row in example]
                phases = np.rad2deg(np.unwrap(np.deg2rad(phases))).tolist() if all(number(v) for v in phases) else phases
                axes[1, column].plot([row["frequency_hz"] for row in example], phases, color=COLORS[name], marker="o")
        axes[0, column].set_title(f"Delay {delay * 1000:g} ms")
        axes[0, column].set_yscale("log")
        axes[0, column].set_ylabel("Resolvent magnitude (position / force)")
        axes[1, column].set_ylabel("Example phase (degrees, unwrapped)")
        axes[0, column].legend(fontsize=7)
        for ax in axes[:, column]:
            ax.set_xscale("log")
            ax.set_xlabel("Frequency (Hz)")
            ax.grid(alpha=.2)
    note = (f"Top: all {population} models, including unstable equilibria; geometric means are formal resolvents, not population steady-state responses.\n"
            + ("Unstable: " + "; ".join(unstable_notes) + ". " if unstable_notes else "All analyzed equilibria locally stable. ")
            + "Bottom: illustrative noise τ=10 ms, seed 11 example. Force held each decision interval.")
    save(fig, output, "loop_frequency", "Formal frequency resolvent of complete-loop linearizations", note)


def load_inputs(directory):
    manifest = read(directory / "run_manifest.json")
    if set(manifest["stages"]) != {"prepare", "passive", "confirm", "mechanism"}:
        raise ValueError("All four experiment stages must be sealed before plotting")
    for category in ("scientific_sources_sha256", "completed_sha256"):
        for relative, expected in manifest[category].items():
            if sha(ROOT / relative) != expected:
                raise ValueError(f"Sealed artifact or scientific source changed: {relative}")
    protocol, base = read(directory / "config.json"), read(directory / "base_config.json")
    if protocol != manifest["protocol_config"] or base != manifest["base_config"]:
        raise ValueError("Configuration differs from run manifest")
    models = ([tuple(item[key] for key in ("tau", "noise_tau", "seed")) for item in protocol["model_subset"]]
              if protocol.get("model_subset") else list(itertools.product(protocol["plant_taus"], base["noise_taus"], base["seeds"])))
    expected = {(*model, delay) for model in models for delay in protocol["delays"]}
    stages = {}
    inputs = [directory / name for name in ("run_manifest.json", "config.json", "base_config.json", "execution_audit.json")]
    for stage in ("prepare", "confirm", "mechanism", "passive"):
        paths = sorted((directory / stage).glob("*.json"))
        for path in paths:
            if manifest["completed_sha256"].get(str(path.relative_to(ROOT))) != sha(path):
                raise ValueError("Unsealed result appeared in a required stage")
        values = [read(path) for path in paths]
        keys = [(record["tau"], record["noise_tau"]) if stage == "passive" else cell(record) for record in values]
        expected_keys = {model[:2] for model in models} if stage == "passive" else expected
        if len(keys) != len(set(keys)) or set(keys) != expected_keys:
            raise ValueError("Declared model population has missing, extra or duplicate records")
        stages[stage] = dict(zip(keys, values))
        inputs.extend(paths)
    return protocol, stages, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/loop_mechanism")
    args = parser.parse_args()
    directory = args.results.resolve()
    previous_manifest = read(directory / "plot_manifest.json") if (directory / "plot_manifest.json").exists() else {}
    plot_directory = directory / "plots"
    plot_directory.mkdir(parents=True, exist_ok=True)
    protocol, stages, inputs = load_inputs(directory)
    counts = {}
    for stage, seeds in (("prepare", protocol["calibration_seeds"]), ("passive", protocol["confirmation_seeds"])):
        counts[stage] = [validate_trials(record["calibration"] if stage == "prepare" else record, seeds,
                                        protocol["probe"]["pulse_amplitudes"]) for record in stages[stage].values()]
    rows = flatten(stages["confirm"].values(), stages["prepare"], stages["mechanism"], stages["passive"], protocol)
    comparisons = [("primary", protocol["primary_low_delay"], protocol["primary_high_delay"])]
    comparisons += [(label, low, high) for label, low, high in (
        ("secondary_same_phase_25_75", .025, .075), ("secondary_same_phase_50_100", .05, .1))
        if low in protocol["delays"] and high in protocol["delays"]]
    endpoints = interactions(rows, comparisons)
    frequency = frequency_rows(stages["mechanism"])
    tables = {"model_metrics.csv": rows, "model_interactions.csv": endpoints,
              "delay_metrics.csv": aggregate(rows, ("delay",)),
              "seed_metrics.csv": aggregate(rows, ("seed", "delay")),
              "interaction_metrics.csv": aggregate(endpoints, ("comparison", "low_delay", "high_delay")),
              "seed_interactions.csv": aggregate(endpoints, ("comparison", "seed", "low_delay", "high_delay")),
              "frequency_metrics.csv": frequency}
    for name, values in tables.items():
        helpers.write_csv(directory / name, values)
    summary = {"models": len({identity(row) for row in rows}), "initialization_seeds": sorted({row["seed"] for row in rows}),
               "model_delay_records": len(rows), "delays": protocol["delays"], "comparisons": comparisons,
               "population_metrics": tables["delay_metrics.csv"], "population_interactions": tables["interaction_metrics.csv"],
               "interaction_summary": interaction_summary(endpoints),
               "by_seed_interactions": tables["seed_interactions.csv"],
               "physical_rollouts": {stage: sum(item["unique_rollouts"] for item in values) for stage, values in counts.items()},
               "confirmation_physical_rollouts": sum(row["unique_rollouts"] for row in rows),
               "confirmation_task_failures": sum(row["task_failures"] for row in rows),
               "confirmation_censored": sum(row["censored"] for row in rows),
               "interpretation": "Targeted exploratory follow-up on existing models; 3 initialization seeds, with 4 noise conditions each. Missing required values invalidate a full-population mean."}
    write_json(directory / "summary.json", summary)
    for function, values in ((recovery_plot, rows), (modes_plot, rows), (matching_plot, rows),
                             (interaction_plot, endpoints), (means_plot, rows), (ringdown_plot, stages["mechanism"])):
        function(values, plot_directory)
    frequency_plot(frequency, rows, plot_directory)
    outputs = [directory / name for name in (*tables.keys(), "summary.json")]
    figures = sorted(plot_directory.glob("loop_*.png")) + sorted(plot_directory.glob("loop_*.svg"))
    outputs += figures
    for figure in figures:
        previous = directory / figure.name
        if previous.exists() and previous_manifest.get("outputs_sha256", {}).get(str(previous.relative_to(ROOT))) == sha(previous):
            previous.unlink()  # Move only artifacts owned by the preceding plot manifest.
    manifest = {"plot_source_sha256": {str(Path(__file__).resolve().relative_to(ROOT)): sha(__file__),
                                       "experiments/plot_delay_sweep.py": sha(ROOT / "experiments/plot_delay_sweep.py")},
                "inputs_sha256": {str(path.relative_to(ROOT)): sha(path) for path in inputs},
                "outputs_sha256": {str(path.relative_to(ROOT)): sha(path) for path in outputs}}
    write_json(directory / "plot_manifest.json", manifest)
    print(json.dumps({"models": summary["models"], "inputs": len(inputs), "outputs": len(outputs), "results": str(directory)}))


if __name__ == "__main__":
    main()
