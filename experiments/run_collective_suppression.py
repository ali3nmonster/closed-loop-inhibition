"""Execute frozen-checkpoint collective suppression and dynamics assays."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from closed_loop_inhibition.collective_suppression import (
    build_bank, discover_model, confirm_model,
)
from closed_loop_inhibition.collective_dynamics import (
    prepare_dynamics, confirm_dynamics, frequency_assay,
)
from closed_loop_inhibition.timescale_maps import cell_id, load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/COLLECTIVE_SUPPRESSION.md"
PRIOR_RUNNER = ROOT / "experiments/run_timescale_maps.py"
_spec = importlib.util.spec_from_file_location("_collective_prior_runner", PRIOR_RUNNER)
prior = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prior)
sha, read_json, write_json = prior.sha, prior.read_json, prior.write_json
seal, completed = prior.seal, prior.completed


def scientific_sources(config_path):
    paths = [*sorted((ROOT / "src").rglob("*.py")), Path(__file__).resolve(),
             PRIOR_RUNNER, PROTOCOL, config_path.resolve()]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def verify_parent(protocol):
    """Check the committed parent contract and each explicitly recorded byte hash.

    New modules are permitted; historical scientific sources are not rewritten.
    """
    directory = ROOT / protocol["parent_results"]
    path = directory / "run_manifest.json"
    committed = subprocess.check_output(
        ["git", "show", f"{protocol['parent_commit']}:{path.relative_to(ROOT)}"], cwd=ROOT)
    if path.read_bytes() != committed:
        raise ValueError("Parent run manifest differs from its committed record")
    manifest = read_json(path)
    base = read_json(directory / "config.json")
    if base != manifest["config"]:
        raise ValueError("Parent configuration differs from the frozen manifest")
    for category in ("scientific_sources_sha256", "completed_sha256"):
        for relative, expected in manifest[category].items():
            target = ROOT / relative
            if not target.exists() or sha(target) != expected:
                raise ValueError(f"Parent scientific artifact changed: {relative}")
    inherited = {"parent_commit": protocol["parent_commit"],
                 "parent_manifest": str(path.relative_to(ROOT)),
                 "parent_manifest_sha256": sha(path),
                 "parent_config_sha256": sha(directory / "config.json"),
                 "source_hashes_checked": len(manifest["scientific_sources_sha256"]),
                 "artifact_hashes_checked": len(manifest["completed_sha256"]),
                 "checkpoint_root": str(Path(manifest["artifacts_directory"]) / "checkpoints")}
    return base, inherited


def validate_streams(base, protocol):
    streams = [protocol["discovery_seeds"], protocol["confirmation_seeds"],
               protocol["dynamics"]["calibration_seeds"], protocol["dynamics"]["confirmation_seeds"]]
    if any(not values or len(values) != len(set(values)) for values in streams):
        raise ValueError("Assay seed streams require unique nonempty seeds")
    if any(set(streams[a]) & set(streams[b]) for a in range(len(streams)) for b in range(a + 1, len(streams))):
        raise ValueError("Discovery, calibration and confirmation streams overlap")
    old = set(base["evaluation"]["seeds"])
    for key in ("discovery_seeds", "calibration_seeds", "confirmation_seeds"):
        old.update(base["diagnostics"][key])
    for start, count in ((base["training"]["train_seed"], base["training"]["episodes"]),
                         (base["training"]["validation_seed"], base["training"]["validation_episodes"])):
        old.update(range(start, start + count))
    if any(old.intersection(values) for values in streams):
        raise ValueError("New probe stream overlaps inherited training or assay seeds")


def models_to_run(base, protocol):
    expected = [(tau, noise_tau, seed) for tau in base["plant_taus"]
                for noise_tau in base["noise_taus"] for seed in base["seeds"]]
    subset = protocol.get("model_subset")
    if subset is None:
        return expected
    values = [(item["tau"], item["noise_tau"], item["seed"]) for item in subset]
    if not values or len(values) != len(set(values)) or not set(values).issubset(expected):
        raise ValueError("model_subset must identify unique inherited models")
    return values


def initialize(output, artifacts, config_path, base, protocol, inherited):
    output.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    path = output / "run_manifest.json"
    current = scientific_sources(config_path)
    if path.exists():
        saved = read_json(path)
        if (saved["scientific_sources_sha256"] != current or saved["protocol_config"] != protocol
                or saved["base_config"] != base or saved["inherited"] != inherited
                or saved["artifacts_directory"] != str(artifacts)):
            raise ValueError("Frozen scientific inputs changed; use a fresh experiment directory")
        for relative, expected in saved["completed_sha256"].items():
            target = ROOT / relative
            if not target.exists() or sha(target) != expected:
                raise ValueError(f"Completed artifact changed or disappeared: {relative}")
        return saved
    if any(output.iterdir()) or any(artifacts.iterdir()):
        raise ValueError("A new experiment requires empty result and artifact directories")
    manifest = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "scientific_sources_sha256": current, "protocol_config": protocol,
                "base_config": base, "inherited": inherited, "completed_sha256": {},
                "artifacts_directory": str(artifacts), "stages": [],
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
                "python": sys.version, "platform": platform.platform(),
                "numpy": np.__version__, "torch": torch.__version__,
                "torch_threads": torch.get_num_threads()}
    write_json(path, manifest)
    write_json(output / "config.json", protocol)
    write_json(output / "base_config.json", base)
    seal(output, manifest, [output / "config.json", output / "base_config.json"])
    return manifest


def checked_record(output, directory, name, manifest):
    path = output / directory / name
    if not completed(path, manifest):
        raise ValueError(f"Required earlier-stage record is absent: {path}")
    return read_json(path), sha(path)


def load_models(base, protocol, inherited, tau, noise_tau, seed):
    directory = Path(inherited["checkpoint_root"]) / cell_id(tau, noise_tau) / f"seed_{seed}"
    models, metadata = {}, {}
    for name, filename in protocol["checkpoints"].items():
        path = directory / filename
        models[name] = load_checkpoint(path, base)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (payload["tau"], payload["noise_tau"], payload["seed"]) != (tau, noise_tau, seed):
            raise ValueError("Checkpoint does not match requested model cell")
        metadata[name] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "epoch": payload["epoch"]}
    return models, metadata


def bank_path(artifacts, tau, split):
    return artifacts / "banks" / f"tau_{tau:g}_{split}.npz"


def save_bank(path, bank):
    arrays = {key: value for key, value in bank.items() if key not in {"pulse", "sham"}}
    for arm in ("pulse", "sham"):
        arrays[f"{arm}_tokens"], arrays[f"{arm}_valid"] = bank[arm]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_bank(path):
    with np.load(path, allow_pickle=False) as saved:
        bank = {key: saved[key].copy() for key in saved.files
                if key not in {"pulse_tokens", "pulse_valid", "sham_tokens", "sham_valid"}}
        for arm in ("pulse", "sham"):
            bank[arm] = (saved[f"{arm}_tokens"].copy(), saved[f"{arm}_valid"].copy())
    return bank


def bank_stage(base, protocol, output, artifacts, manifest):
    for tau in sorted({item[0] for item in models_to_run(base, protocol)}):
        for split in ("discovery", "confirmation"):
            target = bank_path(artifacts, tau, split)
            if not completed(target, manifest):
                save_bank(target, build_bank(base, protocol, tau, protocol[f"{split}_seeds"]))
                seal(output, manifest, [target])
            print(f"bank {tau:g} {split}", flush=True)
    seal(output, manifest, [], "banks")


def model_stage(stage, base, protocol, output, artifacts, manifest):
    common_banks = {}
    selected_models = models_to_run(base, protocol)
    for index, (tau, noise_tau, seed) in enumerate(selected_models, 1):
        name = f"{cell_id(tau, noise_tau)}_seed_{seed}.json"
        target = output / stage / name
        if completed(target, manifest):
            print(f"reused {stage} {index}/{len(selected_models)} {name}", flush=True)
            continue
        models, metadata = load_models(base, protocol, manifest["inherited"], tau, noise_tau, seed)
        dependencies = {}
        if stage in ("discovery", "common_confirmation"):
            split = "discovery" if stage == "discovery" else "confirmation"
            path = bank_path(artifacts, tau, split)
            if not completed(path, manifest):
                raise ValueError("Common bank was not sealed before use")
            if (tau, split) not in common_banks:
                common_banks[(tau, split)] = load_bank(path)
            bank = common_banks[(tau, split)]
            dependencies["bank"] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
            if stage == "discovery":
                result = discover_model(base, protocol, tau, noise_tau, seed, models, bank=bank)
            else:
                discovery, digest = checked_record(output, "discovery", name, manifest)
                dependencies["discovery_sha256"] = digest
                result = confirm_model(base, protocol, tau, noise_tau, seed, models, discovery, bank=bank)
        elif stage in ("dynamics_prepared", "frequency"):
            discovery, digest = checked_record(output, "discovery", name, manifest)
            dependencies["discovery_sha256"] = digest
            groups = {key: discovery["checkpoints"][key]["selected"] for key in models}
            if stage == "dynamics_prepared":
                result = prepare_dynamics(base, protocol, tau, noise_tau, seed, models["selected"], groups["selected"])
            else:
                result = frequency_assay(base, protocol, tau, noise_tau, seed, models, groups)
        elif stage == "dynamics_confirmation":
            prepared, digest = checked_record(output, "dynamics_prepared", name, manifest)
            dependencies["prepared_sha256"] = digest
            result = confirm_dynamics(base, protocol, tau, noise_tau, seed, models["selected"], prepared)
        else:
            raise ValueError(f"Unknown stage: {stage}")
        result["checkpoint_metadata"] = metadata
        result["dependencies"] = dependencies
        write_json(target, result)
        seal(output, manifest, [target])
        print(f"completed {stage} {index}/{len(selected_models)} {name}", flush=True)
    seal(output, manifest, [], stage)


def main():
    stages = ("banks", "discovery", "common_confirmation", "dynamics_prepared", "frequency", "dynamics_confirmation")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/collective_suppression.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/collective_suppression")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "runs/collective_suppression")
    parser.add_argument("--stage", choices=("all", *stages), default="all")
    args = parser.parse_args()
    protocol = read_json(args.config)
    torch.set_num_threads(protocol["cpu_threads"])
    torch.set_num_interop_threads(1)
    base, inherited = verify_parent(protocol)
    validate_streams(base, protocol)
    output, artifacts = args.output.resolve(), args.artifacts.resolve()
    manifest = initialize(output, artifacts, args.config, base, protocol, inherited)
    for stage in stages:
        if args.stage not in ("all", stage):
            continue
        if scientific_sources(args.config) != manifest["scientific_sources_sha256"]:
            raise ValueError("Scientific inputs changed during the experiment")
        print(f"stage_start {stage}", flush=True)
        if stage == "banks":
            bank_stage(base, protocol, output, artifacts, manifest)
        else:
            model_stage(stage, base, protocol, output, artifacts, manifest)
        if scientific_sources(args.config) != manifest["scientific_sources_sha256"]:
            raise ValueError("Scientific inputs changed during the experiment")
        print(f"stage_complete {stage}", flush=True)
    print(json.dumps({"complete_stages": manifest["stages"], "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
