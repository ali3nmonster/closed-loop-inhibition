"""Calibrate and confirm small, command-centered interventions in frozen heads.

All offsets and control strengths are fitted on native calibration histories.
Confirmation policies use constant settings on their own evolving histories.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from closed_loop_inhibition.centered_calibration import (
    centered_commands, fit_command_offset, weighted_mean,
)
from closed_loop_inhibition.centered_metrics import mean_applied_action, mean_position
from closed_loop_inhibition.controllers import PredictorPDController
from closed_loop_inhibition.imitation import SharedTeacher
from closed_loop_inhibition.interventions import scale_attention_head
from closed_loop_inhibition.plants import Oscillator
from closed_loop_inhibition.suppression_analysis import immediate_output_rms, pick_rms_match, response_statistics
from closed_loop_inhibition.suppression_tasks import make_scenarios, paired_rollouts, score_pair

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/CENTERED_SUPPRESSION_PILOT.md"
_spec = importlib.util.spec_from_file_location("_centered_prior_runner", ROOT / "experiments/run_suppression.py")
prior = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prior)

VARIANTS = ("native", "raw_weak", "raw_strong", "centered_weak", "centered_strong",
            "alternative_matched", "gain_matched")
SIGNED_FIELDS = ("pulse_mean_applied_action", "sham_mean_applied_action",
                 "pulse_mean_position", "sham_mean_position")


class CenteredPolicy(prior.Policy):
    """Transform physical raw commands once; the simulator performs final clipping."""

    def __init__(self, model, plant, encoding, *, center=0.0, gain=1.0, offset=0.0):
        super().__init__(model, plant, encoding)
        self.center, self.command_gain, self.offset = center, gain, offset

    def __call__(self, snapshot):
        raw = super().__call__(snapshot)
        return float(self.center + self.command_gain * (raw - self.center) + self.offset)


def verify_inherited_selection(config):
    """Check the inherited selection against both its original manifest and Git."""
    directory = ROOT / config["prior_suppression_results"]
    commit = config["prior_suppression_commit"]
    names = ("selection.json", "discovery_manifest.json", "scenarios.json", "config.json")
    for name in names:
        path = directory / name
        relative = str(path.relative_to(ROOT))
        committed = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        if path.read_bytes() != committed:
            raise ValueError(f"Inherited selection provenance changed: {relative}")
    discovery = json.loads((directory / "discovery_manifest.json").read_text())
    if prior.sha(directory / "selection.json") != discovery["output_sha256"]["selection.json"]:
        raise ValueError("Inherited selection disagrees with its discovery manifest")
    selection = json.loads((directory / "selection.json").read_text())
    if sorted(row["seed"] for row in selection) != sorted(config["model_seeds"]):
        raise ValueError("Inherited model seeds differ from requested seeds")
    if not all(row["selected"] for row in selection):
        raise ValueError("Every frozen model must have an inherited selected head")
    provenance = {"commit": commit, "files_sha256": {name: prior.sha(directory / name) for name in names}}
    return selection, provenance


def generate_scenarios(config):
    """Require independent streams and exogenous draws, including the prior pilot."""
    directory = ROOT / config["prior_suppression_results"]
    old_config = json.loads((directory / "config.json").read_text())
    old_scenarios = json.loads((directory / "scenarios.json").read_text())
    previous_seeds = {settings["seed"] for settings in old_config["splits"].values()}
    seeds = [settings["seed"] for settings in config["splits"].values()]
    if len(set(seeds)) != len(seeds) or previous_seeds.intersection(seeds):
        raise ValueError("Calibration and confirmation require fresh independent random streams")
    scenarios = {}
    for split, settings in config["splits"].items():
        scenarios[split] = make_scenarios(count=settings["count"], seed=settings["seed"],
                                          split=settings.get("prefix", f"centered_{split}"))
    signature = lambda row: json.dumps({key: value for key, value in row.items() if key != "id"}, sort_keys=True)
    seen = {signature(row) for rows in old_scenarios.values() for row in rows}
    identifiers = {row["id"] for rows in old_scenarios.values() for row in rows}
    for rows in scenarios.values():
        for row in rows:
            if signature(row) in seen or row["id"] in identifiers:
                raise ValueError("New scenarios duplicate prior or cross-stage exogenous scenarios")
            seen.add(signature(row))
            identifiers.add(row["id"])
    return scenarios


def grid_values(settings, name):
    explicit = settings.get(f"{name}s")
    if explicit is not None:
        return [float(value) for value in explicit]
    low, high, step = (settings[f"{name}_{part}"] for part in ("min", "max", "step"))
    return np.round(np.arange(low, high + step / 2, step), 9).tolist()


def stage_manifest(output, artifacts, checkpoints, outputs, datafiles=()):
    saved = prior.manifest(output, artifacts, checkpoints, outputs, datafiles)
    saved["protocol_sha256"] = prior.sha(PROTOCOL)
    return saved


def verify_stage(output, artifacts, name):
    saved = prior.verify_stage(output, artifacts, name)
    if saved["protocol_sha256"] != prior.sha(PROTOCOL):
        raise ValueError("Prospective protocol changed between centered suppression stages")
    return saved


def choose_match(grid, target, settings):
    """Use ascending-parameter ties and retain identity for unavailable targets."""
    epsilon = settings.get("target_epsilon", 1e-8)
    if target < epsilon:
        identity = next(row for row in grid if row["parameter"] == 1.0)
        return {**identity, "target_rms": float(target), "relative_error": None,
                "matched": False, "available": False, "reason": "target_below_epsilon"}
    selected = min(grid, key=lambda row: (abs(row["achieved_rms"] - target), row["parameter"]))
    matched = pick_rms_match([selected["parameter"]], [selected["achieved_rms"]], target,
                             tolerance=settings["matching_relative_tolerance"])
    return {**selected, **{key: value for key, value in matched.items() if key != "scale"},
            "available": True, "reason": None}


def fit_variant(native_raw, changed_raw, groups, limit, *, atol, center=0.0, gain=1.0):
    """Fit on sham only, then compute fixed-history pulse and sham effects."""
    transformed = tuple(center + gain * (raw - center) for raw in changed_raw)
    fitted = fit_command_offset(native_raw[1], transformed[1], groups, limit, atol=atol)
    outputs = tuple(centered_commands(raw, action_limit=limit, center=center, gain=gain,
                                      offset=fitted["offset"]) for raw in changed_raw)
    baseline = tuple(np.clip(raw, -limit, limit) for raw in native_raw)
    return {**fitted, "center": float(center), "gain": float(gain),
            "achieved_rms": immediate_output_rms(*baseline, *outputs, groups)}, outputs


def calibrate_cell(native_raw, candidate_raw, alternative_raw, groups, limit, config):
    """Return seven frozen settings and every matched-control grid diagnostic."""
    settings = config["calibration"]
    atol = settings.get("mean_atol", 1e-10)
    weak_scale, strong_scale = (config["candidate_scales"][key] for key in ("weak", "strong"))
    baseline = tuple(np.clip(raw, -limit, limit) for raw in native_raw)
    variants = {"native": {"head": "native", "head_scale": 1.0, "gain": 1.0, "center": 0.0, "offset": 0.0}}
    outputs = {"native": baseline}
    for label, scale in (("weak", weak_scale), ("strong", strong_scale)):
        raw = candidate_raw[scale]
        raw_outputs = tuple(np.clip(values, -limit, limit) for values in raw)
        fitted, centered = fit_variant(native_raw, raw, groups, limit, atol=atol)
        variants[f"raw_{label}"] = {"head": "candidate", "head_scale": scale, "gain": 1.0,
                                    "center": 0.0, "offset": 0.0,
                                    "achieved_rms": immediate_output_rms(*baseline, *raw_outputs, groups)}
        variants[f"centered_{label}"] = {"head": "candidate", "head_scale": scale, **fitted}
        outputs[f"raw_{label}"], outputs[f"centered_{label}"] = raw_outputs, centered
    target = variants["centered_weak"]["achieved_rms"]
    alternative_grid = []
    for scale, raw in sorted(alternative_raw.items()):
        fitted, _ = fit_variant(native_raw, raw, groups, limit, atol=atol)
        alternative_grid.append({"parameter": scale, **fitted})
    center = weighted_mean(baseline[1], groups)
    gain_grid = []
    for gain in grid_values(settings, "gain"):
        fitted, _ = fit_variant(native_raw, native_raw, groups, limit, atol=atol, center=center, gain=gain)
        gain_grid.append({"parameter": gain, **fitted})
    alternative = choose_match(alternative_grid, target, settings)
    gain = choose_match(gain_grid, target, settings)
    variants["alternative_matched"] = {"head": "alternative", "head_scale": alternative["parameter"], **alternative}
    variants["gain_matched"] = {"head": "native", "head_scale": 1.0, **gain}
    return variants, {"alternative_matched": alternative_grid, "gain_matched": gain_grid}, outputs


def make_policy(model, selection, settings, plant, encoding):
    if settings["head"] == "native":
        changed = model
    else:
        head = selection[settings["head"]]
        changed = scale_attention_head(model, head["layer"], head["head"], settings["head_scale"])
    return CenteredPolicy(changed, plant, encoding, center=settings["center"],
                          gain=settings["gain"], offset=settings["offset"])


def evaluate_policy(plant, policy_factory, scenarios, config, seed, variant):
    """Use one immutable per-condition policy on both arms and all scenarios."""
    rows = []
    for schedule, delay in prior.conditions(config):
        policy = policy_factory(schedule, delay)
        for scenario in scenarios:
            pulse, sham = paired_rollouts(plant, policy, scenario, prior.timing(config, schedule, delay))
            scores = score_pair(pulse, sham, scenario, recovery_seconds=config["recovery_seconds"])
            jobs = [job for job in pulse.jobs if job["status"] == "applied"]
            intervals = np.diff([job["apply_time"] for job in jobs])
            ages = [job["apply_time"] - job["capture_time"] for job in jobs]
            extra = {}
            for label, run in (("pulse", pulse), ("sham", sham)):
                applied = [job for job in run.jobs if job["status"] == "applied"]
                extra[f"{label}_mean_applied_action"] = mean_applied_action(run)
                extra[f"{label}_mean_position"] = mean_position(run)
                extra[f"{label}_clipped_actions"] = sum(job["raw_action"] != job["action"] for job in applied)
                extra[f"{label}_applied_actions"] = len(applied)
            rows.append({"seed": seed, "variant": variant, "scenario_id": scenario["id"],
                         "schedule": schedule, "compute_duration": delay, "amplitude": scenario["amplitude"],
                         "mean_application_interval": float(np.mean(intervals)) if len(intervals) else None,
                         "mean_observation_age_at_application": float(np.mean(ages)) if ages else None,
                         **scores, **extra, "absolute_competence": prior.absolute_competence(scores, config)})
    return rows


def drift_checks(rows, config):
    checks = []
    tolerance = config["competence"]["maximum_sham_rmse_gap"]
    for seed in config["model_seeds"]:
        for schedule, delay in prior.conditions(config):
            native = {row["scenario_id"]: row for row in rows if row["seed"] == seed and row["variant"] == "native"
                      and row["schedule"] == schedule and row["compute_duration"] == delay}
            for variant in sorted({row["variant"] for row in rows if row["seed"] == seed} - {"native"}):
                changed = [row for row in rows if row["seed"] == seed and row["variant"] == variant
                           and row["schedule"] == schedule and row["compute_duration"] == delay]
                complete = bool(changed) and len(changed) == len(native) and all(
                    not row["censored"] and not native[row["scenario_id"]]["censored"] for row in changed)
                difference = float(np.mean([row["sham_tracking_rmse"] - native[row["scenario_id"]]["sham_tracking_rmse"]
                                            for row in changed])) if complete else None
                checks.append({"seed": seed, "schedule": schedule, "compute_duration": delay, "variant": variant,
                               "mean_sham_rmse_gap": difference, "maximum_gap": tolerance,
                               "passed": bool(complete and difference <= tolerance),
                               "absolute_competence": bool(complete and all(row["absolute_competence"] for row in changed))})
    return checks


def augment_summary(rows, config):
    aggregates, effects = prior.summarize_results(rows, config)
    for aggregate in aggregates:
        chosen = [row for row in rows if all(row[key] == aggregate[key] for key in
                  ("seed", "variant", "schedule", "compute_duration")) and not row["censored"]]
        for field in SIGNED_FIELDS:
            aggregate[f"mean_{field}"] = float(np.mean([row[field] for row in chosen])) if chosen else None
    paired_contrasts = []
    for seed in config["model_seeds"]:
        for schedule in config["schedules"]:
            indexed = {(row["variant"], row["scenario_id"], row["compute_duration"]): row
                       for row in rows if row["seed"] == seed and row["schedule"] == schedule}
            identifiers = sorted({row["scenario_id"] for row in rows if row["seed"] == seed})
            for scenario_id in identifiers:
                interactions = {}
                for variant in VARIANTS[1:]:
                    low, high = config["primary"]["low_delay"], config["primary"]["high_delay"]
                    quartet = [indexed[(variant, scenario_id, high)], indexed[("native", scenario_id, high)],
                               indexed[(variant, scenario_id, low)], indexed[("native", scenario_id, low)]]
                    interactions[variant] = None if any(row["censored"] for row in quartet) else float(
                        quartet[0]["J_response"] - quartet[1]["J_response"] - quartet[2]["J_response"] + quartet[3]["J_response"])
                for control in ("raw_weak", "alternative_matched", "gain_matched"):
                    left, right = interactions["centered_weak"], interactions[control]
                    paired_contrasts.append({"seed": seed, "schedule": schedule, "scenario_id": scenario_id,
                                             "control": control, "candidate_interaction": left, "control_interaction": right,
                                             "candidate_minus_control": None if left is None or right is None else left - right})
    return aggregates, effects, paired_contrasts


def make_plots(output, aggregates, config):
    colors = ("#222222", "#df9d96", "#98c8b8", "#b64135", "#35876f", "#cd9231", "#3d75b4")
    fig, axes = prior.plt.subplots(3, len(config["schedules"]), figsize=(12, 11), squeeze=False)
    metrics = (("mean_J_response", "Normalized response J (s)"),
               ("mean_sham_tracking_rmse", "Own-sham position RMSE"),
               ("mean_pulse_action_effort", "Pulse action effort"))
    for column, schedule in enumerate(config["schedules"]):
        for variant, color in zip(VARIANTS, colors):
            for row_index, (field, label) in enumerate(metrics):
                values, low, high = [], [], []
                for delay in config["compute_durations"]:
                    cell = [row[field] for row in aggregates if row["variant"] == variant
                            and row["schedule"] == schedule and row["compute_duration"] == delay and row[field] is not None]
                    values.append(np.mean(cell) if cell else np.nan)
                    low.append(min(cell) if cell else np.nan)
                    high.append(max(cell) if cell else np.nan)
                axis = axes[row_index, column]
                axis.plot(config["compute_durations"], values, "o-", color=color, label=variant)
                axis.fill_between(config["compute_durations"], low, high, color=color, alpha=.07)
                axis.set(ylabel=label, xlabel="Virtual computation duration (s)")
                axis.grid(alpha=.2)
        axes[0, column].set_title(schedule.replace("_", " "))
        axes[0, column].legend(fontsize=7)
    fig.suptitle("Centered intervention pilot: mean and range across three frozen models", fontsize=11)
    fig.tight_layout()
    fig.savefig(output / "centered_results.png", dpi=150)
    path = output / "centered_results.svg"
    fig.savefig(path, metadata={"Date": None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    prior.plt.close(fig)


def calibrate(config, pretrained, models, plant, scenarios, selection, output, artifacts, checkpoints):
    cells, grids, directions, matching, datafiles = [], [], [], [], []
    limit = pretrained["encoding"]["action_limit"]
    head_scales = grid_values(config["calibration"], "alternative_scale")
    for item in selection:
        seed, model = item["seed"], models[item["seed"]]
        data = prior.collect(model, plant, pretrained["encoding"], scenarios, config)
        filename = f"calibration_{seed}.npz"
        np.savez_compressed(artifacts / filename, **data)
        datafiles.append(filename)
        native_raw = prior.predict(model, data, limit, clip=False)
        candidate_raw, alternative_raw = {}, {}
        for scale in config["candidate_scales"].values():
            head = item["candidate"]
            changed = scale_attention_head(model, head["layer"], head["head"], scale)
            candidate_raw[scale] = prior.predict(changed, data, limit, clip=False)
        for scale in head_scales:
            head = item["alternative"]
            changed = scale_attention_head(model, head["layer"], head["head"], scale)
            alternative_raw[scale] = prior.predict(changed, data, limit, clip=False)
        print(json.dumps({"stage": "calibration_predictions", "seed": seed, "alternative_grid": len(head_scales)}), flush=True)
        for schedule, delay in prior.conditions(config):
            mask = (data["schedules"] == schedule) & (data["delays"] == delay)
            sliced = lambda pair: tuple(values[mask] for values in pair)
            settings, cell_grids, outputs = calibrate_cell(sliced(native_raw),
                {scale: sliced(pair) for scale, pair in candidate_raw.items()},
                {scale: sliced(pair) for scale, pair in alternative_raw.items()}, data["groups"][mask], limit, config)
            identity = {"seed": seed, "schedule": schedule, "compute_duration": delay}
            cells.append({**identity, "variants": settings})
            for variant, grid in cell_grids.items():
                grids.extend({**identity, "variant": variant, **row} for row in grid)
                matching.append({**identity, "variant": variant, **settings[variant]})
            if schedule == config["discovery"]["schedule"] and delay == config["discovery"]["compute_duration"]:
                for variant in ("raw_weak", "centered_weak", "raw_strong", "centered_strong"):
                    score = response_statistics(*outputs["native"], *outputs[variant], data["groups"][mask],
                        data["amplitudes"][mask], min_fractional_increase=0.0,
                        min_positive_fraction=config["calibration"]["direction_minimum_positive_fraction"],
                        baseline_epsilon=config["calibration"]["baseline_response_epsilon"])
                    directions.append({**identity, "variant": variant, **score})
        prior.write_json(output / "calibration.json", cells)
        prior.write_json(output / "calibration_direction.json", directions)
        prior.write_csv(output / "calibration_matching.csv", matching)
        print(json.dumps({"stage": "calibration_settings", "seed": seed, "cells": len(prior.conditions(config))}), flush=True)
    prior.write_csv(output / "calibration_grid.csv", grids)
    teacher = SharedTeacher(PredictorPDController(plant, **pretrained["teacher"]), **pretrained["encoding"])
    rows = evaluate_policy(plant, lambda schedule, delay: teacher, scenarios, config, -1, "teacher")
    for item in selection:
        seed, model = item["seed"], models[item["seed"]]
        lookup = {(row["schedule"], row["compute_duration"]): row["variants"] for row in cells if row["seed"] == seed}
        for variant in ("native", "centered_weak", "centered_strong"):
            factory = lambda schedule, delay: make_policy(model, item, lookup[(schedule, delay)][variant], plant, pretrained["encoding"])
            rows.extend(evaluate_policy(plant, factory, scenarios, config, seed, variant))
            prior.write_csv(output / "calibration_rollouts.csv", rows)
            print(json.dumps({"stage": "calibration_closed_loop", "seed": seed, "variant": variant}), flush=True)
    readiness = {"native_competence": prior.check_competence(rows, config), "drift": drift_checks(rows, config),
                 "direction": [{key: row[key] for key in ("seed", "variant", "eligible", "median_fractional_increase", "positive_fraction")}
                               for row in directions],
                 "matching_checks": len(matching), "matching_passed": sum(row["matched"] for row in matching)}
    prior.write_json(output / "calibration_readiness.json", readiness)
    verify_stage(output, artifacts, "calibration_start")
    outputs = ["calibration_start_manifest.json", "config.json", "scenarios.json", "selection.json", "inherited_provenance.json", "calibration.json",
               "calibration_direction.json", "calibration_matching.csv", "calibration_grid.csv",
               "calibration_rollouts.csv", "calibration_readiness.json"]
    saved = stage_manifest(output, artifacts, checkpoints, outputs, datafiles)
    saved.update({"confirmation_evaluated": False, "probe_rollouts": 2 * len(scenarios) * len(models) * len(prior.conditions(config)),
                  "calibration_rollouts": 2 * len(rows), "pretrained_commit": config["pretrained_commit"]})
    prior.write_json(output / "calibration_manifest.json", saved)
    print(json.dumps({"stage": "calibration_complete", "rollouts": saved["probe_rollouts"] + saved["calibration_rollouts"],
                      "matching_passed": readiness["matching_passed"], "matching_checks": len(matching)}), flush=True)


def confirm(config, pretrained, models, plant, scenarios, selection, output, artifacts, checkpoints):
    calibrated = json.loads((output / "calibration.json").read_text())
    teacher = SharedTeacher(PredictorPDController(plant, **pretrained["teacher"]), **pretrained["encoding"])
    rows = evaluate_policy(plant, lambda schedule, delay: teacher, scenarios, config, -1, "teacher")
    for item in selection:
        seed, model = item["seed"], models[item["seed"]]
        lookup = {(row["schedule"], row["compute_duration"]): row["variants"] for row in calibrated if row["seed"] == seed}
        for variant in VARIANTS:
            factory = lambda schedule, delay: make_policy(model, item, lookup[(schedule, delay)][variant], plant, pretrained["encoding"])
            rows.extend(evaluate_policy(plant, factory, scenarios, config, seed, variant))
            prior.write_csv(output / "confirmation_pairs.csv", rows)
            print(json.dumps({"stage": "confirmation", "seed": seed, "variant": variant, "completed_rollouts": 2 * len(rows)}), flush=True)
    aggregate, effects, contrasts = augment_summary(rows, config)
    prior.write_csv(output / "condition_summary.csv", aggregate)
    prior.write_json(output / "interactions.json", effects)
    prior.write_csv(output / "paired_interaction_contrasts.csv", contrasts)
    prior.write_json(output / "confirmation_competence.json", {"native": prior.check_competence(rows, config), "drift": drift_checks(rows, config)})
    make_plots(output, aggregate, config)
    outputs = ["calibration_manifest.json", "config.json", "scenarios.json", "selection.json", "inherited_provenance.json",
               "calibration.json", "confirmation_pairs.csv", "condition_summary.csv", "interactions.json",
               "paired_interaction_contrasts.csv", "confirmation_competence.json", "centered_results.png", "centered_results.svg"]
    verify_stage(output, artifacts, "calibration")
    saved = stage_manifest(output, artifacts, checkpoints, outputs)
    saved.update({"confirmation_evaluated": True, "confirmation_pairs": len(rows), "confirmation_rollouts": 2 * len(rows),
                  "censored_pairs": sum(row["censored"] for row in rows), "pretrained_commit": config["pretrained_commit"]})
    prior.write_json(output / "confirmation_manifest.json", saved)
    print(json.dumps({"stage": "complete", "rollouts": 2 * len(rows), "censored_pairs": saved["censored_pairs"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/centered_suppression_pilot.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/centered_suppression_pilot")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "runs/centered_suppression_pilot")
    parser.add_argument("--stage", choices=("calibrate", "evaluate"), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    torch.set_num_threads(config["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    selection, provenance = verify_inherited_selection(config)
    scenarios = generate_scenarios(config)
    pretrained, models, checkpoints = prior.load_frozen_models(config)
    plant = Oscillator(**pretrained["plant"])
    args.output.mkdir(parents=True, exist_ok=True)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    if args.stage == "calibrate":
        if any(args.output.iterdir()) or any(args.artifacts.iterdir()):
            raise ValueError("Calibration requires fresh output and artifact directories")
        prior.write_json(args.output / "config.json", config)
        prior.write_json(args.output / "scenarios.json", scenarios)
        prior.write_json(args.output / "selection.json", selection)
        prior.write_json(args.output / "inherited_provenance.json", provenance)
        initial = stage_manifest(args.output, args.artifacts, checkpoints,
                                 ["config.json", "scenarios.json", "selection.json", "inherited_provenance.json"])
        initial["confirmation_evaluated"] = False
        prior.write_json(args.output / "calibration_start_manifest.json", initial)
        calibrate(config, pretrained, models, plant, scenarios["calibration"], selection,
                  args.output, args.artifacts, checkpoints)
    else:
        if (args.output / "confirmation_manifest.json").exists() or (args.output / "confirmation_pairs.csv").exists():
            raise ValueError("Confirmation has already been started; preserve its outcomes and choose a new experiment")
        verify_stage(args.output, args.artifacts, "calibration")
        for filename, expected in (("config.json", config), ("scenarios.json", scenarios),
                                   ("selection.json", selection), ("inherited_provenance.json", provenance)):
            if json.loads((args.output / filename).read_text()) != expected:
                raise ValueError(f"Frozen calibration metadata differs: {filename}")
        confirm(config, pretrained, models, plant, scenarios["confirmation"], selection,
                args.output, args.artifacts, checkpoints)


if __name__ == "__main__":
    main()
