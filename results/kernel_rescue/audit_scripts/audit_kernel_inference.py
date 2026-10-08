"""Independent raw-trial reconstruction of the kernel-rescue population inference."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def complete_mean(values):
    return None if not values or any(v is None or not np.isfinite(v) for v in values) else float(sum(values) / len(values))


def sub(a, b):
    return None if a is None or b is None else a - b


def fractional(value, native, epsilon):
    return None if value is None or native is None or native <= epsilon else 100. * (value / native - 1.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("/home/ball/transformer-closed-loop-inhibition/results/kernel_rescue"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/kernel_inference_audit.json"))
    args = parser.parse_args()
    root = args.results
    protocol, base, reported = read(root / "config.json"), read(root / "base_config.json"), read(root / "summary.json")
    variants = ("native", "joint_weak", "weak_equilibrium", "weak_scalar", "weak_kernel")
    contrasts = (("equilibrium_minus_weak", "weak_equilibrium", "joint_weak"),
                 ("scalar_minus_equilibrium", "weak_scalar", "weak_equilibrium"),
                 ("kernel_minus_scalar", "weak_kernel", "weak_scalar"),
                 ("kernel_minus_weak", "weak_kernel", "joint_weak"))
    low, high = protocol["primary_low_delay"], protocol["primary_high_delay"]
    comparisons, maximum = 0, 0.

    def check(actual, expected, label):
        nonlocal comparisons, maximum
        comparisons += 1
        if actual is None or expected is None:
            if actual is not expected:
                raise AssertionError(f"Missingness mismatch: {label}: {actual} / {expected}")
        else:
            error = abs(actual - expected)
            maximum = max(maximum, error)
            if error > 1e-11 * max(1., abs(actual), abs(expected)):
                raise AssertionError(f"Arithmetic mismatch: {label}: {actual} / {expected}")

    values = {}
    physical = failed = censored = 0
    for path in sorted((root / "confirm").glob("*.json")):
        record = read(path)
        identity = (record["population"], record["tau"], record["noise_tau"], record["seed"], record["duration"])
        expected_pairs = {(seed, amplitude) for seed in protocol["confirmation_seeds"] for amplitude in protocol["probe"]["pulse_amplitudes"]}
        for name in variants:
            rows = [row for row in record["rollouts"] if row["variant"] == name]
            summary = record["summary"][name]
            if summary is None:
                if rows or name not in record["unavailable_controls"]:
                    raise AssertionError("Unavailable policy has trials or lacks its reason")
                energy = position = action = None
            else:
                if {(row["noise_seed"], row["amplitude"]) for row in rows} != expected_pairs or len(rows) != len(expected_pairs):
                    raise AssertionError("Incomplete paired raw-trial population")
                energy = complete_mean([row["paired_recovery"]["normalized_position_energy"] for row in rows])
                shams = {row["noise_seed"]: row["sham"] for row in rows}
                for row in rows:
                    if row["sham"] != shams[row["noise_seed"]]:
                        raise AssertionError("Pulse signs disagree about shared sham")
                position = complete_mean([row["position_rms"] for row in shams.values()])
                action = complete_mean([row["action_rms"] for row in shams.values()])
                check(energy, summary["paired_recovery"]["normalized_position_energy"], f"{path.name}/{name}/energy")
                check(position, summary["sham"]["position_rms"], f"{path.name}/{name}/position")
                check(action, summary["sham"]["action_rms"], f"{path.name}/{name}/action")
                unique_trials = list(shams.values()) + [row["pulse"] for row in rows]
                physical += len(unique_trials)
                failed += sum(row["task_failure"] for row in unique_trials)
                censored += sum(row["censored"] for row in unique_trials)
            values[(identity, record["delay"], name)] = (energy, position, action)

    interactions = defaultdict(dict)
    for identity in {key[0] for key in values}:
        for name in variants[1:]:
            for index, metric in enumerate(("recovery_absolute", "position_rms_absolute", "action_rms_absolute")):
                lo = sub(values[(identity, low, name)][index], values[(identity, low, "native")][index])
                hi = sub(values[(identity, high, name)][index], values[(identity, high, "native")][index])
                interactions[identity][f"{name}_{metric}_interaction"] = sub(hi, lo)
            lo = fractional(values[(identity, low, name)][0], values[(identity, low, "native")][0], protocol["metric_denominator_epsilon"])
            hi = fractional(values[(identity, high, name)][0], values[(identity, high, "native")][0], protocol["metric_denominator_epsilon"])
            interactions[identity][f"{name}_recovery_percent_interaction"] = sub(hi, lo)
        for label, after, before in contrasts:
            for metric in ("recovery_absolute", "recovery_percent"):
                interactions[identity][f"{label}_{metric}_interaction"] = sub(
                    interactions[identity][f"{after}_{metric}_interaction"], interactions[identity][f"{before}_{metric}_interaction"])

    for cohort in reported["inference"]:
        population, tau, duration = cohort["population"], cohort["tau"], cohort["duration"]
        seeds = sorted(protocol["fresh_seeds"] if population == "fresh" else base["seeds"])
        if protocol.get("model_subset"):
            seeds = sorted({row["seed"] for row in protocol["model_subset"] if row["population"] == population})
            noise_taus = sorted({row["noise_tau"] for row in protocol["model_subset"]})
        else:
            noise_taus = base["noise_taus"]
        for field, estimate in cohort["metrics"].items():
            per_seed = [complete_mean([interactions[(population, tau, noise, seed, duration)][field] for noise in noise_taus]) for seed in seeds]
            for actual, row in zip(per_seed, estimate["by_seed"]):
                check(actual, row["value"], f"{population}/{duration}/{field}/seed{row['seed']}")
            check(complete_mean(per_seed), estimate["mean"], f"{population}/{duration}/{field}/mean")
            if population == "fresh" and all(value is not None for value in per_seed) and len(per_seed) > 1:
                stat = protocol["statistics"]
                draws = np.random.default_rng(stat["bootstrap_seed"]).integers(0, len(seeds), (stat["bootstrap_resamples"], len(seeds)))
                replicates = np.asarray(per_seed)[draws].mean(axis=1)
                tail = (1. - stat["confidence_level"]) / 2.
                bounds = np.quantile(replicates, (tail, 1. - tail))
                check(float(bounds[0]), estimate["ci_low"], f"{population}/{duration}/{field}/ci_low")
                check(float(bounds[1]), estimate["ci_high"], f"{population}/{duration}/{field}/ci_high")
            else:
                check(None, estimate["ci_low"], f"{population}/{duration}/{field}/unavailable_low")
                check(None, estimate["ci_high"], f"{population}/{duration}/{field}/unavailable_high")
    check(physical, reported["confirmation_physical_rollouts"], "physical_count")
    check(failed, reported["confirmation_task_failures"], "task_failure_count")
    check(censored, reported["confirmation_censored"], "censor_count")
    result = {"passed": True, "arithmetic_comparisons": comparisons, "maximum_absolute_error": maximum,
              "confirmation_physical_rollouts": physical, "confirmation_task_failures": failed, "confirmation_censored": censored,
              "method": "Independent raw-trial means, per-model paired differences, equal-noise seed means and paired-seed bootstrap; no plotter or scientific summary helpers imported."}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
