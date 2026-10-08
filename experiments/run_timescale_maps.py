"""Run the prospectively specified force-only two-timescale map experiment.

Stages persist complete model-level records. A resumed run requires unchanged
scientific sources/configuration/protocol and unchanged completed artifacts.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from closed_loop_inhibition.timescale_maps import (
    NeuralPolicy, cell_id, load_checkpoint, make_plant, metrics, run_episode,
    teacher_policy, train_cell,
)
from closed_loop_inhibition.timescale_diagnostics import prepare_model, confirm_model

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/TIMESCALE_MAPS_PILOT.md"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def source_fingerprints(config_path):
    paths = sorted((ROOT / "src").rglob("*.py")) + [Path(__file__).resolve(), PROTOCOL,
                                                               config_path.resolve()]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def initialize(output, artifacts, config_path, config):
    output.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    path = output / "run_manifest.json"
    current = source_fingerprints(config_path)
    if path.exists():
        saved = read_json(path)
        if saved["scientific_sources_sha256"] != current or saved["config"] != config:
            raise ValueError("Scientific sources, configuration or protocol changed; use a new run directory")
        if saved["artifacts_directory"] != str(artifacts):
            raise ValueError("Artifact directory differs from the original run")
        for relative, expected in saved["completed_sha256"].items():
            target = ROOT / relative
            if not target.exists() or sha(target) != expected:
                raise ValueError(f"Completed artifact changed or disappeared: {relative}")
        return saved
    if any(output.iterdir()) or any(artifacts.iterdir()):
        raise ValueError("A new run requires empty result and artifact directories")
    saved = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scientific_sources_sha256": current, "config": config,
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "artifacts_directory": str(artifacts), "completed_sha256": {},
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(), "stages": [],
    }
    write_json(path, saved)
    write_json(output / "config.json", config)
    return saved


def seal(output, manifest, paths, stage=None):
    for path in paths:
        manifest["completed_sha256"][str(Path(path).relative_to(ROOT))] = sha(path)
    if stage and stage not in manifest["stages"]:
        manifest["stages"].append(stage)
    write_json(output / "run_manifest.json", manifest)


def check_unchanged(manifest, config_path):
    if manifest["scientific_sources_sha256"] != source_fingerprints(config_path):
        raise ValueError("Scientific implementation changed during execution")


def completed(path, manifest):
    """Only sealed results can be reused; interrupted writes need inspection."""
    path = Path(path)
    if not path.exists():
        return False
    relative = str(path.relative_to(ROOT))
    expected = manifest["completed_sha256"].get(relative)
    if expected is None or sha(path) != expected:
        raise ValueError(f"Existing result is unsealed or changed: {relative}")
    return True


def cells(config):
    for tau in config["plant_taus"]:
        for noise_tau in config["noise_taus"]:
            yield tau, noise_tau


def evaluate_policy(config, tau, noise_tau, policy, kind, model_seed=None):
    rows = []
    for background_seed in config["evaluation"]["seeds"]:
        run = run_episode(config, tau, noise_tau, policy, background_seed)
        rows.append({"tau": tau, "noise_tau": noise_tau, "controller": kind,
                     "model_seed": model_seed, "background_seed": background_seed,
                     **metrics(config, run)})
    return rows


def baseline_stage(config, output, artifacts, manifest):
    path = output / "baseline.json"
    if completed(path, manifest):
        return read_json(path)
    rows = []
    for tau, noise_tau in cells(config):
        teacher = teacher_policy(config, make_plant(config, tau))
        for name, policy in (("passive", lambda snapshot: 0.), ("teacher", teacher)):
            rows.extend(evaluate_policy(config, tau, noise_tau, policy, name))
        print(f"baseline {cell_id(tau, noise_tau)}", flush=True)
    write_json(path, rows)
    write_csv(output / "baseline.csv", rows)
    seal(output, manifest, [path, output / "baseline.csv"], "baseline")
    return rows


def training_stage(config, output, artifacts, manifest):
    summaries, history = [], []
    for tau, noise_tau in cells(config):
        for seed in config["seeds"]:
            target = output / "training" / f"{cell_id(tau, noise_tau)}_seed_{seed}.json"
            if completed(target, manifest):
                record = read_json(target)
            else:
                _, summary, rows = train_cell(config, tau, noise_tau, seed, artifacts)
                record = {"summary": summary, "history": rows}
                write_json(target, record)
                directory = artifacts / "checkpoints" / cell_id(tau, noise_tau) / f"seed_{seed}"
                data = artifacts / "data" / cell_id(tau, noise_tau)
                seal(output, manifest, [target, *directory.glob("*.pt"), *data.glob("*.npz")])
            summaries.append(record["summary"])
            history.extend(record["history"])
            print(f"trained {len(summaries)}/{len(config['plant_taus'])*len(config['noise_taus'])*len(config['seeds'])} "
                  f"{cell_id(tau, noise_tau)} seed={seed} val={record['summary']['validation_mse']:.6g}", flush=True)
    write_csv(output / "training.csv", summaries)
    write_csv(output / "learning_curves.csv", history)
    seal(output, manifest, [output / "training.csv", output / "learning_curves.csv"], "training")


def selected_model(config, tau, noise_tau, seed, artifacts):
    directory = artifacts / "checkpoints" / cell_id(tau, noise_tau) / f"seed_{seed}"
    return load_checkpoint(directory / "selected.pt", config), directory


def performance_stage(config, output, artifacts, manifest):
    all_rows = read_json(output / "baseline.json")
    for tau, noise_tau in cells(config):
        for seed in config["seeds"]:
            target = output / "performance" / f"{cell_id(tau, noise_tau)}_seed_{seed}.json"
            if completed(target, manifest):
                rows = read_json(target)
            else:
                model, _ = selected_model(config, tau, noise_tau, seed, artifacts)
                policy = NeuralPolicy(model, make_plant(config, tau), config)
                rows = evaluate_policy(config, tau, noise_tau, policy, "transformer", seed)
                write_json(target, rows)
                seal(output, manifest, [target])
            all_rows.extend(rows)
        print(f"performance {cell_id(tau, noise_tau)}", flush=True)
    write_csv(output / "performance.csv", all_rows)
    seal(output, manifest, [output / "performance.csv"], "performance")


def prepare_stage(config, output, artifacts, manifest):
    for tau, noise_tau in cells(config):
        for seed in config["seeds"]:
            target = output / "prepared" / f"{cell_id(tau, noise_tau)}_seed_{seed}.json"
            if not completed(target, manifest):
                model, directory = selected_model(config, tau, noise_tau, seed, artifacts)
                prepared = prepare_model(config, tau, noise_tau, seed, model, directory)
                write_json(target, prepared)
                seal(output, manifest, [target])
            print(f"prepared {cell_id(tau, noise_tau)} seed={seed}", flush=True)
    seal(output, manifest, [], "prepare")


def confirmation_stage(config, output, artifacts, manifest):
    for tau, noise_tau in cells(config):
        for seed in config["seeds"]:
            name = f"{cell_id(tau, noise_tau)}_seed_{seed}.json"
            prepared_path = output / "prepared" / name
            expected = manifest["completed_sha256"][str(prepared_path.relative_to(ROOT))]
            if sha(prepared_path) != expected:
                raise ValueError("Discovery/calibration record changed before confirmation")
            target = output / "confirmation" / name
            if not completed(target, manifest):
                model, _ = selected_model(config, tau, noise_tau, seed, artifacts)
                confirmed = confirm_model(config, tau, noise_tau, seed, model, read_json(prepared_path))
                confirmed["prepared_sha256"] = expected
                write_json(target, confirmed)
                seal(output, manifest, [target])
            print(f"confirmed {cell_id(tau, noise_tau)} seed={seed}", flush=True)
    seal(output, manifest, [], "confirm")


def convergence_stage(config, output, artifacts, manifest):
    target = output / "reporting_convergence.json"
    if completed(target, manifest):
        return
    tau = min(config["plant_taus"])
    seed = config["seeds"][0]
    background_seed = config["evaluation"]["seeds"][0]
    results = []
    for noise_tau in (min(config["noise_taus"]), max(config["noise_taus"])):
        model, _ = selected_model(config, tau, noise_tau, seed, artifacts)
        plant = make_plant(config, tau)
        for name, policy in (("passive", lambda snapshot: 0.),
                             ("teacher", teacher_policy(config, plant)),
                             ("transformer", NeuralPolicy(model, plant, config))):
            old_path = output / ("baseline.json" if name != "transformer" else
                                f"performance/{cell_id(tau, noise_tau)}_seed_{seed}.json")
            coarse = next(row for row in read_json(old_path)
                          if row["tau"] == tau and row["noise_tau"] == noise_tau
                          and row["controller"] == name and row["background_seed"] == background_seed)
            fine_config = {**config, "sample_interval": .005}
            run = run_episode(fine_config, tau, noise_tau, policy, background_seed)
            fine = metrics(fine_config, run)
            a, b = coarse["position_rms"], fine["position_rms"]
            difference = None if a is None or b is None else abs(a - b) / max(abs(b), 1e-12)
            results.append({"tau": tau, "noise_tau": noise_tau, "controller": name,
                            "model_seed": seed if name == "transformer" else None,
                            "background_seed": background_seed, "coarse": coarse, "fine": fine,
                            "position_rms_relative_difference": difference})
    write_json(target, results)
    seal(output, manifest, [target], "convergence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/timescale_maps.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/timescale_maps")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "runs/timescale_maps")
    parser.add_argument("--stage", choices=["all", "baseline", "training", "performance", "prepare", "confirm", "convergence"], default="all")
    args = parser.parse_args()
    config = read_json(args.config)
    torch.set_num_threads(config["training"]["cpu_threads"])
    torch.set_num_interop_threads(1)
    output, artifacts = args.output.resolve(), args.artifacts.resolve()
    manifest = initialize(output, artifacts, args.config, config)
    stages = [("baseline", baseline_stage), ("training", training_stage),
              ("performance", performance_stage), ("prepare", prepare_stage),
              ("confirm", confirmation_stage), ("convergence", convergence_stage)]
    for name, function in stages:
        if args.stage in ("all", name):
            check_unchanged(manifest, args.config)
            print(f"stage_start {name}", flush=True)
            function(config, output, artifacts, manifest)
            check_unchanged(manifest, args.config)
            print(f"stage_complete {name}", flush=True)
    print(json.dumps({"complete_stages": manifest["stages"], "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
