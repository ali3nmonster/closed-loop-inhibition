"""Discover, calibrate, and test attention-head suppression on frozen policies.

Run separate stages to keep candidate selection and control calibration apart
from held-out closed-loop outcomes. No controller weights are trained here.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from closed_loop_inhibition.controllers import PredictorPDController
from closed_loop_inhibition.imitation import SharedTeacher, encode_snapshot
from closed_loop_inhibition.interventions import enumerate_attention_heads, head_residuals, scale_attention_head
from closed_loop_inhibition.neural import make_model
from closed_loop_inhibition.plants import Oscillator
from closed_loop_inhibition.suppression_analysis import (
    immediate_output_rms, norm_match_head, pick_rms_match, response_statistics,
)
from closed_loop_inhibition.suppression_tasks import (
    collect_paired_probes, make_scenarios, paired_rollouts, score_pair,
)
from closed_loop_inhibition.timing import TimingConfig

ROOT = Path(__file__).resolve().parents[1]
matplotlib.rcParams["svg.hashsalt"] = "closed-loop-inhibition-suppression"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    return {str(path.relative_to(ROOT)): sha(path)
            for directory, suffix in (("src", "py"), ("experiments", "py"), ("configs", "json"))
            for path in sorted((ROOT / directory).rglob(f"*.{suffix}"))}


class Policy:
    def __init__(self, model, plant, encoding, gain=1.0):
        self.model, self.plant, self.encoding, self.gain = model.eval(), plant, encoding, gain

    def __call__(self, snapshot):
        tokens, valid = encode_snapshot(snapshot, tau=self.plant.tau, **self.encoding)
        with torch.inference_mode():
            normalized = self.model(torch.from_numpy(tokens[None]), torch.from_numpy(valid[None])).item()
        return float(normalized * self.encoding["action_limit"] * self.gain)


def conditions(config, discovery=False):
    if discovery:
        return [(config["discovery"]["schedule"], config["discovery"]["compute_duration"])]
    return [(schedule, delay) for schedule in config["schedules"] for delay in config["compute_durations"]]


def timing(config, schedule, delay):
    return TimingConfig(**config["timing"], schedule=schedule, compute_duration=delay)


def load_frozen_models(config):
    """Verify original implementation and checkpoint bytes, allowing new modules."""
    source_commit = config["pretrained_commit"]
    original = ["neural.py", "imitation.py", "timing.py", "records.py", "plants.py", "signals.py", "metrics.py", "controllers.py"]
    for name in original:
        relative = f"src/closed_loop_inhibition/{name}"
        old = subprocess.check_output(["git", "show", f"{source_commit}:{relative}"], cwd=ROOT)
        if (ROOT / relative).read_bytes() != old:
            raise ValueError(f"Frozen controller/simulator implementation changed: {relative}")
    results = ROOT / config["pretrained_results"]
    old_manifest = json.loads((results / "training_manifest.json").read_text())
    for name in ("config.json", "training_summary.json"):
        if sha(results / name) != old_manifest["output_sha256"][name]:
            raise ValueError(f"Pretrained metadata changed: {name}")
    pretrained = json.loads((results / "config.json").read_text())
    summaries = json.loads((results / "training_summary.json").read_text())
    models, checkpoint_hashes = {}, {}
    for seed in config["model_seeds"]:
        row = next(row for row in summaries if row["model"] == "transformer" and row["seed"] == seed)
        path = ROOT / config["pretrained_artifacts"] / "checkpoints" / f"transformer_{seed}.pt"
        if sha(path) != row["checkpoint_sha256"]:
            raise ValueError(f"Pretrained checkpoint changed: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["model_config"] != pretrained["model"] or payload["encoding"] != pretrained["encoding"]:
            raise ValueError("Checkpoint model/encoding metadata mismatch")
        model = make_model("transformer", **payload["model_config"])
        model.load_state_dict(payload["state_dict"])
        model.eval().requires_grad_(False)
        models[seed] = model
        checkpoint_hashes[str(path.relative_to(ROOT))] = sha(path)
    return pretrained, models, checkpoint_hashes


def verify_stage(output, artifacts, name):
    manifest = json.loads((output / f"{name}_manifest.json").read_text())
    if manifest["source_sha256"] != source_hashes():
        raise ValueError("Sources changed between suppression stages; start a separate run")
    for path, expected in manifest["checkpoint_sha256"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"Frozen checkpoint changed: {path}")
    for location, key in ((output, "output_sha256"), (artifacts, "artifact_sha256")):
        for path, expected in manifest[key].items():
            if sha(location / path) != expected:
                raise ValueError(f"Stage artifact changed: {location / path}")
    return manifest


def manifest(output, artifacts, checkpoint_hashes, outputs, datafiles=()):
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "torch": torch.__version__, "numpy": np.__version__, "device": "cpu",
        "cpu_threads": torch.get_num_threads(), "source_sha256": source_hashes(),
        "source_commit_at_run": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_tree_dirty_at_run": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "checkpoint_sha256": checkpoint_hashes,
        "output_sha256": {name: sha(output / name) for name in outputs},
        "artifact_sha256": {name: sha(artifacts / name) for name in datafiles},
    }


def collect(model, plant, encoding, scenarios, config, discovery=False):
    parts = []
    for scenario in scenarios:
        for schedule, delay in conditions(config, discovery):
            data = collect_paired_probes(plant, Policy(model, plant, encoding), scenario,
                                        timing(config, schedule, delay), encoding,
                                        window_seconds=config["discovery"]["window_seconds"])
            count = len(data["times"])
            group = f"{scenario['id']}:{schedule}:{delay:g}"
            data.update({"groups": np.repeat(group, count), "scenario_ids": np.repeat(scenario["id"], count),
                         "amplitudes": np.repeat(scenario["amplitude"], count),
                         "schedules": np.repeat(schedule, count), "delays": np.repeat(delay, count)})
            parts.append(data)
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def predict(model, data, action_limit, gain=1.0, clip=True):
    result = []
    with torch.inference_mode():
        for arm in ("pulse", "sham"):
            values = []
            for start in range(0, len(data["valid"]), 256):
                x = torch.from_numpy(data[f"{arm}_tokens"][start:start + 256])
                mask = torch.from_numpy(data["valid"][start:start + 256])
                values.append(model(x, mask).numpy().astype(np.float64) * action_limit * gain)
            joined = np.concatenate(values)
            result.append(np.clip(joined, -action_limit, action_limit) if clip else joined)
    return tuple(result)


def response(base, changed, data, config, calibration=False):
    settings = config["discovery"]
    return response_statistics(*base, *changed, data["groups"], data["amplitudes"],
                               min_fractional_increase=0.0 if calibration else settings["minimum_fractional_increase"],
                               min_positive_fraction=settings["minimum_positive_fraction"],
                               baseline_epsilon=settings["baseline_epsilon"])


def residual_rms(model, data):
    sums = None
    count = 0
    with torch.inference_mode():
        for arm in ("pulse", "sham"):
            for start in range(0, len(data["valid"]), 128):
                mask = torch.from_numpy(data["valid"][start:start + 128])
                contributions = head_residuals(model, torch.from_numpy(data[f"{arm}_tokens"][start:start + 128]), mask)
                reduced = contributions.double().square().sum(dim=(0, 1, 4)).numpy()
                sums = reduced if sums is None else sums + reduced
                count += int(mask.sum())
    return np.sqrt(sums / count)


def discover(config, pretrained, models, plant, scenarios, output, artifacts, checkpoints):
    selections, all_heads = [], []
    restoration = []
    files = []
    for seed, model in models.items():
        data = collect(model, plant, pretrained["encoding"], scenarios, config, discovery=True)
        filename = f"discovery_{seed}.npz"
        np.savez_compressed(artifacts / filename, **data)
        files.append(filename)
        baseline = predict(model, data, pretrained["encoding"]["action_limit"])
        norms = residual_rms(model, data)
        heads = []
        for layer, head in enumerate_attention_heads(model):
            changed_model = scale_attention_head(model, layer, head, config["discovery"]["attenuation_scale"])
            scores = response(baseline, predict(changed_model, data, pretrained["encoding"]["action_limit"]), data, config)
            row = {"seed": seed, "layer": layer, "head": head, "residual_rms": float(norms[layer, head]), **scores}
            heads.append(row)
            all_heads.append(row)
        eligible = [row for row in heads if row["eligible"]]
        if eligible:
            candidate = sorted(eligible, key=lambda r: (-r["median_fractional_increase"], r["layer"], r["head"]))[0]
            alternative = norm_match_head(heads, candidate["layer"], candidate["head"])
            selected = {"seed": seed, "selected": True, "candidate": candidate, "alternative": alternative}
        else:
            selected = {"seed": seed, "selected": False, "reason": "No head met the frozen discovery rule"}
        selections.append(selected)
        check_head = selected["candidate"] if selected["selected"] else {"layer": 0, "head": 0}
        restored = scale_attention_head(model, check_head["layer"], check_head["head"], .5)
        restored.load_state_dict(model.state_dict())
        schedule, delay = conditions(config, discovery=True)[0]
        native_runs = paired_rollouts(plant, Policy(model, plant, pretrained["encoding"]), scenarios[0], timing(config, schedule, delay))
        restored_runs = paired_rollouts(plant, Policy(restored, plant, pretrained["encoding"]), scenarios[0], timing(config, schedule, delay))
        action_error, state_error = 0.0, 0.0
        for native_run, restored_run in zip(native_runs, restored_runs):
            if len(native_run.applied_actions) != len(restored_run.applied_actions):
                raise RuntimeError("Restored policy changed action timing")
            action_error = max(action_error, max(abs(a.value - b.value) for a, b in zip(native_run.applied_actions, restored_run.applied_actions)))
            state_error = max(state_error, max(abs(a["position"] - b["position"]) for a, b in zip(native_run.samples, restored_run.samples)))
        if action_error != 0.0 or state_error != 0.0:
            raise RuntimeError("Restored model did not reproduce the original paired trajectories")
        restoration.append({"seed": seed, "scenario_id": scenarios[0]["id"], "schedule": schedule,
                            "compute_duration": delay, "maximum_action_difference": action_error,
                            "maximum_position_difference": state_error})
        print(json.dumps({"stage": "discovery", "seed": seed, "selected": selected["selected"],
                          "eligible_heads": len(eligible)}), flush=True)
    write_json(output / "head_discovery.json", all_heads)
    write_json(output / "selection.json", selections)
    write_json(output / "restoration.json", restoration)
    saved = manifest(output, artifacts, checkpoints,
                     ["config.json", "scenarios.json", "head_discovery.json", "selection.json", "restoration.json"], files)
    saved["confirmation_evaluated"] = False
    write_json(output / "discovery_manifest.json", saved)


def absolute_competence(scores, config):
    limits = config["competence"]
    return (not scores["censored"]
            and max(scores["pulse_peak_abs_error"], scores["sham_peak_abs_error"]) < limits["maximum_peak_error"]
            and max(scores["pulse_last_second_rmse"], scores["sham_last_second_rmse"]) <= limits["maximum_last_second_rmse"])


def evaluate_policy(plant, policy, scenarios, config, seed, variant):
    rows = []
    for scenario in scenarios:
        for schedule, delay in conditions(config):
            pulse, sham = paired_rollouts(plant, policy, scenario, timing(config, schedule, delay))
            scores = score_pair(pulse, sham, scenario, recovery_seconds=config["recovery_seconds"])
            jobs = [job for job in pulse.jobs if job["status"] == "applied"]
            intervals = np.diff([job["apply_time"] for job in jobs])
            ages = [job["apply_time"] - job["capture_time"] for job in jobs]
            rows.append({"seed": seed, "variant": variant, "scenario_id": scenario["id"],
                         "schedule": schedule, "compute_duration": delay, "amplitude": scenario["amplitude"],
                         "mean_application_interval": float(np.mean(intervals)) if len(intervals) else None,
                         "mean_observation_age_at_application": float(np.mean(ages)) if ages else None,
                         **scores, "absolute_competence": absolute_competence(scores, config)})
    return rows


def check_competence(rows, config):
    checks = []
    teacher = [r for r in rows if r["variant"] == "teacher"]
    for seed in config["model_seeds"]:
        for schedule, delay in conditions(config):
            native = [r for r in rows if r["seed"] == seed and r["variant"] == "native"
                      and r["schedule"] == schedule and r["compute_duration"] == delay]
            reference = [r for r in teacher if r["schedule"] == schedule and r["compute_duration"] == delay]
            complete = bool(native) and all(not r["censored"] for r in native + reference)
            ratio = float(np.mean([r["J_response"] for r in native]) / np.mean([r["J_response"] for r in reference])) if complete else None
            checks.append({"seed": seed, "schedule": schedule, "compute_duration": delay,
                           "response_ratio_to_teacher": ratio,
                           "absolute_competence": all(r["absolute_competence"] for r in native),
                           "passed": complete and all(r["absolute_competence"] for r in native)
                                     and ratio <= config["competence"]["maximum_teacher_response_ratio"]})
    return checks


def subset(data, mask):
    return {key: value[mask] for key, value in data.items()}


def calibrate(config, pretrained, models, plant, scenarios, output, artifacts, checkpoints):
    selection = json.loads((output / "selection.json").read_text())
    results, files, matching_rows = [], [], []
    limit = pretrained["encoding"]["action_limit"]
    for item in selection:
        seed, model = item["seed"], models[item["seed"]]
        if not item["selected"]:
            results.append({"seed": seed, "selected": False})
            continue
        data = collect(model, plant, pretrained["encoding"], scenarios, config)
        filename = f"calibration_{seed}.npz"
        np.savez_compressed(artifacts / filename, **data)
        files.append(filename)
        baseline_raw = predict(model, data, limit, clip=False)
        baseline = tuple(np.clip(values, -limit, limit) for values in baseline_raw)
        candidate, alternative = item["candidate"], item["alternative"]
        half = scale_attention_head(model, candidate["layer"], candidate["head"], .5)
        changed = predict(half, data, limit)
        target = immediate_output_rms(*baseline, *changed, data["groups"])
        mask = ((data["schedules"] == config["discovery"]["schedule"])
                & (data["delays"] == config["discovery"]["compute_duration"]))
        replication = response(tuple(x[mask] for x in baseline), tuple(x[mask] for x in changed), subset(data, mask), config, calibration=True)
        head_scales, head_outputs, head_rms = config["calibration"]["head_scales"], [], []
        for scale in head_scales:
            variant = scale_attention_head(model, alternative["layer"], alternative["head"], scale)
            predictions = predict(variant, data, limit)
            head_outputs.append(predictions)
            head_rms.append(immediate_output_rms(*baseline, *predictions, data["groups"]))
        alternative_match = pick_rms_match(head_scales, head_rms, target, tolerance=config["calibration"]["matching_relative_tolerance"])
        settings = config["calibration"]
        gains = np.round(np.arange(settings["gain_min"], settings["gain_max"] + settings["gain_step"] / 2, settings["gain_step"]), 6).tolist()
        # Retain raw neural outputs when scaling, then apply the shared actuator clip.
        gain_outputs = [tuple(np.clip(values * gain, -limit, limit) for values in baseline_raw) for gain in gains]
        gain_rms = [immediate_output_rms(*baseline, *predictions, data["groups"]) for predictions in gain_outputs]
        gain_match = pick_rms_match(gains, gain_rms, target, tolerance=settings["matching_relative_tolerance"])
        chosen_alternative = head_outputs[head_scales.index(alternative_match["scale"])]
        chosen_gain = gain_outputs[gains.index(gain_match["scale"])]
        for schedule, delay in conditions(config):
            selected = (data["schedules"] == schedule) & (data["delays"] == delay)
            native_cell = tuple(x[selected] for x in baseline)
            target_cell = immediate_output_rms(*native_cell, *(x[selected] for x in changed), data["groups"][selected])
            for name, predictions in (("alternative_matched", chosen_alternative), ("gain_matched", chosen_gain)):
                achieved = immediate_output_rms(*native_cell, *(x[selected] for x in predictions), data["groups"][selected])
                relative = abs(achieved - target_cell) / target_cell if target_cell > 1e-12 else None
                matching_rows.append({"seed": seed, "variant": name, "schedule": schedule, "compute_duration": delay,
                                      "target_rms": target_cell, "achieved_rms": achieved, "relative_error": relative,
                                      "matched": relative is not None and relative <= settings["matching_relative_tolerance"]})
        results.append({"seed": seed, "selected": True, "replication": replication,
                        "alternative_match": alternative_match, "gain_match": gain_match,
                        "head_match_grid": [{"scale": scale, "rms": value} for scale, value in zip(head_scales, head_rms)],
                        "gain_match_grid": [{"scale": scale, "rms": value} for scale, value in zip(gains, gain_rms)]})
        print(json.dumps({"stage": "calibration", "seed": seed, "replicated": replication["eligible"],
                          "alternative_match": alternative_match, "gain_match": gain_match}), flush=True)
    teacher = SharedTeacher(PredictorPDController(plant, **pretrained["teacher"]), **pretrained["encoding"])
    rows = evaluate_policy(plant, teacher, scenarios, config, -1, "teacher")
    for seed, model in models.items():
        rows.extend(evaluate_policy(plant, Policy(model, plant, pretrained["encoding"]), scenarios, config, seed, "native"))
    readiness = check_competence(rows, config)
    write_json(output / "calibration.json", results)
    write_json(output / "calibration_readiness.json", readiness)
    write_csv(output / "calibration_matching.csv", matching_rows)
    write_csv(output / "calibration_rollouts.csv", rows)
    outputs = ["discovery_manifest.json", "config.json", "scenarios.json", "selection.json", "calibration.json",
               "calibration_readiness.json", "calibration_rollouts.csv"]
    if matching_rows:
        outputs.append("calibration_matching.csv")
    saved = manifest(output, artifacts, checkpoints, outputs, files)
    saved["confirmation_evaluated"] = False
    write_json(output / "calibration_manifest.json", saved)
    print(json.dumps({"stage": "calibration_complete", "native_competence_checks_passed": sum(r["passed"] for r in readiness),
                      "native_competence_checks": len(readiness)}), flush=True)


def variants(model, selection, calibration):
    result = [("native", model, 1.0)]
    if selection["selected"]:
        candidate, alternative = selection["candidate"], selection["alternative"]
        result.extend([
            ("candidate_half", scale_attention_head(model, candidate["layer"], candidate["head"], .5), 1.0),
            ("candidate_stronger", scale_attention_head(model, candidate["layer"], candidate["head"], 1.5), 1.0),
            ("alternative_half", scale_attention_head(model, alternative["layer"], alternative["head"], .5), 1.0),
            ("alternative_matched", scale_attention_head(model, alternative["layer"], alternative["head"], calibration["alternative_match"]["scale"]), 1.0),
            ("gain_matched", model, calibration["gain_match"]["scale"]),
        ])
    return result


def summarize_results(rows, config):
    aggregates, effects = [], []
    for seed in sorted({r["seed"] for r in rows}):
        for variant in sorted({r["variant"] for r in rows if r["seed"] == seed}):
            for schedule, delay in conditions(config):
                chosen = [r for r in rows if r["seed"] == seed and r["variant"] == variant
                          and r["schedule"] == schedule and r["compute_duration"] == delay]
                valid = [r for r in chosen if not r["censored"]]
                aggregates.append({"seed": seed, "variant": variant, "schedule": schedule, "compute_duration": delay,
                                   "pairs": len(chosen), "censored_pairs": len(chosen) - len(valid),
                                   "absolute_competence_failures": sum(not r["absolute_competence"] for r in chosen),
                                   **{f"mean_{key}": float(np.mean([r[key] for r in valid])) if valid else None
                                      for key in ("J_response", "peak_response", "sham_tracking_rmse", "pulse_tracking_rmse", "pulse_action_effort", "sham_action_effort", "pulse_action_variation", "sham_action_variation")}})
    for seed in config["model_seeds"]:
        for schedule in config["schedules"]:
            native = {(r["scenario_id"], r["compute_duration"]): r for r in rows
                      if r["seed"] == seed and r["variant"] == "native" and r["schedule"] == schedule}
            for variant in sorted({r["variant"] for r in rows if r["seed"] == seed} - {"native"}):
                changed = {(r["scenario_id"], r["compute_duration"]): r for r in rows
                           if r["seed"] == seed and r["variant"] == variant and r["schedule"] == schedule}
                values = []
                for scenario in sorted({key[0] for key in native}):
                    low, high = config["primary"]["low_delay"], config["primary"]["high_delay"]
                    quartet = [changed[(scenario, high)], native[(scenario, high)], changed[(scenario, low)], native[(scenario, low)]]
                    if any(r["censored"] for r in quartet):
                        continue
                    interaction = quartet[0]["J_response"] - quartet[1]["J_response"] - quartet[2]["J_response"] + quartet[3]["J_response"]
                    values.append(float(interaction))
                effects.append({"seed": seed, "schedule": schedule, "variant": variant,
                                "complete_paired_scenarios": len(values),
                                "mean_interaction": float(np.mean(values)) if values else None,
                                "scenario_interactions": values})
    return aggregates, effects


def make_plots(output, aggregates, effects, config):
    colors = {"native": "#222222", "candidate_half": "#b64135", "candidate_stronger": "#35876f",
              "alternative_half": "#9674b6", "alternative_matched": "#cd9231", "gain_matched": "#3d75b4"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for column, schedule in enumerate(config["schedules"]):
        for variant, color in colors.items():
            available = [r for r in aggregates if r["variant"] == variant and r["schedule"] == schedule]
            if not available:
                continue
            means, low, high, drift = [], [], [], []
            for delay in config["compute_durations"]:
                selected = [r for r in available if r["compute_duration"] == delay and r["mean_J_response"] is not None]
                values = [r["mean_J_response"] for r in selected]
                means.append(np.mean(values) if values else np.nan)
                low.append(min(values) if values else np.nan)
                high.append(max(values) if values else np.nan)
                drift.append(np.mean([r["mean_sham_tracking_rmse"] for r in selected]) if selected else np.nan)
            axes[0, column].plot(config["compute_durations"], means, "o-", color=color, label=variant)
            axes[0, column].fill_between(config["compute_durations"], low, high, color=color, alpha=.1)
            axes[1, column].plot(config["compute_durations"], drift, "o-", color=color, label=variant)
        axes[0, column].set(title=schedule.replace("_", " "), ylabel="Normalized disturbance-response ISE")
        axes[1, column].set(ylabel="No-disturbance tracking RMSE", xlabel="Virtual computation duration (s)")
    for axis in axes.flat:
        axis.grid(alpha=.2)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "suppression_results.png", dpi=150)
    svg = output / "suppression_results.svg"
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def confirm(config, pretrained, models, plant, scenarios, output, artifacts, checkpoints):
    selections = json.loads((output / "selection.json").read_text())
    calibration = json.loads((output / "calibration.json").read_text())
    teacher = SharedTeacher(PredictorPDController(plant, **pretrained["teacher"]), **pretrained["encoding"])
    rows = evaluate_policy(plant, teacher, scenarios, config, -1, "teacher")
    for seed, model in models.items():
        selection = next(r for r in selections if r["seed"] == seed)
        calibrated = next(r for r in calibration if r["seed"] == seed)
        for name, variant, gain in variants(model, selection, calibrated):
            rows.extend(evaluate_policy(plant, Policy(variant, plant, pretrained["encoding"], gain), scenarios, config, seed, name))
            print(json.dumps({"stage": "confirmation", "seed": seed, "variant": name}), flush=True)
    aggregate, effects = summarize_results(rows, config)
    write_csv(output / "confirmation_pairs.csv", rows)
    write_csv(output / "condition_summary.csv", aggregate)
    write_json(output / "interactions.json", effects)
    write_json(output / "confirmation_competence.json", check_competence(rows, config))
    make_plots(output, aggregate, effects, config)
    saved = manifest(output, artifacts, checkpoints,
                     ["calibration_manifest.json", "config.json", "scenarios.json", "confirmation_pairs.csv",
                      "condition_summary.csv", "interactions.json", "confirmation_competence.json", "suppression_results.png", "suppression_results.svg"])
    saved.update({"confirmation_pairs": len(rows), "confirmation_rollouts": 2 * len(rows),
                  "censored_pairs": sum(r["censored"] for r in rows), "confirmation_evaluated": True})
    write_json(output / "confirmation_manifest.json", saved)
    print(json.dumps({"stage": "complete", "pairs": len(rows), "rollouts": 2 * len(rows)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/suppression_pilot.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/suppression_pilot")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "runs/suppression_pilot")
    parser.add_argument("--stage", choices=["discover", "calibrate", "evaluate"], required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    torch.set_num_threads(config["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    pretrained, models, checkpoint_hashes = load_frozen_models(config)
    plant = Oscillator(**pretrained["plant"])
    args.output.mkdir(parents=True, exist_ok=True)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    scenarios = {split: make_scenarios(**settings, split=split) for split, settings in config["splits"].items()}
    stage_manifest = {"discover": "discovery", "calibrate": "calibration", "evaluate": "confirmation"}[args.stage]
    if (args.output / f"{stage_manifest}_manifest.json").exists():
        raise ValueError("Completed stage exists; choose fresh output/artifact directories")
    if args.stage == "discover":
        write_json(args.output / "config.json", config)
        write_json(args.output / "scenarios.json", scenarios)
        discover(config, pretrained, models, plant, scenarios["discovery"], args.output, args.artifacts, checkpoint_hashes)
    else:
        verify_stage(args.output, args.artifacts, "discovery")
        if json.loads((args.output / "config.json").read_text()) != config:
            raise ValueError("Configuration differs from discovery")
        if args.stage == "calibrate":
            calibrate(config, pretrained, models, plant, scenarios["calibration"], args.output, args.artifacts, checkpoint_hashes)
        else:
            verify_stage(args.output, args.artifacts, "calibration")
            confirm(config, pretrained, models, plant, scenarios["confirmation"], args.output, args.artifacts, checkpoint_hashes)


if __name__ == "__main__":
    main()
