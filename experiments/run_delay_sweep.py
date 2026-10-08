"""Frozen-policy command-latency experiment with resumable, sealed outcomes."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
import math
import multiprocessing
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from closed_loop_inhibition.timescale_maps import cell_id, load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/DELAY_SWEEP.md"
INHERITED_RUNNERS = [ROOT / "experiments/run_timescale_maps.py",
                     ROOT / "experiments/run_collective_suppression.py"]
PARENT_STAGES = {"banks", "discovery", "common_confirmation", "dynamics_prepared",
                 "frequency", "dynamics_confirmation"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def scientific_sources(config_path):
    paths = [*sorted((ROOT / "src").rglob("*.py")), Path(__file__).resolve(),
             *INHERITED_RUNNERS, PROTOCOL, config_path.resolve()]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def models_to_run(base, protocol):
    full = list(itertools.product(base["plant_taus"], base["noise_taus"], base["seeds"]))
    subset = protocol.get("model_subset")
    if subset is None:
        return full
    chosen = [(r["tau"], r["noise_tau"], r["seed"]) for r in subset]
    if not chosen or len(set(chosen)) != len(chosen) or not set(chosen) <= set(full):
        raise ValueError("model_subset must contain unique inherited model cells")
    return chosen


def jobs(base, protocol):
    return [(*cell, delay) for cell in models_to_run(base, protocol) for delay in protocol["delays"]]


def _verify_hashes(manifest):
    for category in ("scientific_sources_sha256", "completed_sha256"):
        for relative, expected in manifest[category].items():
            path = ROOT / relative
            if not path.exists() or sha(path) != expected:
                raise ValueError(f"Inherited artifact or source changed: {relative}")


def checked_record(path, manifest):
    path = Path(path)
    relative = str(path.relative_to(ROOT))
    if not path.exists() or manifest["completed_sha256"].get(relative) != sha(path):
        raise ValueError(f"Required dependency is absent, unsealed or changed: {relative}")
    return read_json(path), sha(path)


def verify_parent(protocol):
    directory = ROOT / protocol["parent_results"]
    path = directory / "run_manifest.json"
    committed = subprocess.check_output(
        ["git", "show", f"{protocol['parent_commit']}:{path.relative_to(ROOT)}"], cwd=ROOT)
    if path.read_bytes() != committed:
        raise ValueError("Collective parent manifest differs from its committed record")
    parent = read_json(path)
    if set(parent["stages"]) != PARENT_STAGES:
        raise ValueError("Collective parent experiment is incomplete")
    _verify_hashes(parent)
    base = read_json(directory / "base_config.json")
    if base != parent["base_config"] or read_json(directory / "config.json") != parent["protocol_config"]:
        raise ValueError("Parent configuration differs from its frozen manifest")
    ancestor_path = ROOT / parent["inherited"]["parent_manifest"]
    if sha(ancestor_path) != parent["inherited"]["parent_manifest_sha256"]:
        raise ValueError("Original training manifest changed")
    ancestor = read_json(ancestor_path)
    if ancestor["config"] != base:
        raise ValueError("Inherited training configuration differs")
    _verify_hashes(ancestor)
    records = {}
    for tau, noise_tau, seed in models_to_run(base, protocol):
        name = f"{cell_id(tau, noise_tau)}_seed_{seed}"
        discovery_path = directory / "discovery" / f"{name}.json"
        prepared_path = directory / "dynamics_prepared" / f"{name}.json"
        discovery, discovery_sha = checked_record(discovery_path, parent)
        prepared, prepared_sha = checked_record(prepared_path, parent)
        if any((row["tau"], row["noise_tau"], row["seed"]) != (tau, noise_tau, seed)
               for row in (discovery, prepared)):
            raise ValueError("Inherited record belongs to a different model cell")
        selected = discovery["checkpoints"]["selected"]["selected"]
        if (prepared["group"] != selected
                or prepared["dependencies"]["discovery_sha256"] != discovery_sha):
            raise ValueError("Inherited group or discovery dependency changed")
        metadata = discovery["checkpoint_metadata"]["selected"]
        checkpoint = ROOT / metadata["path"]
        if (prepared["checkpoint_metadata"]["selected"] != metadata
                or checkpoint.name != protocol["checkpoint"]
                or ancestor["completed_sha256"].get(metadata["path"]) != metadata["sha256"]
                or sha(checkpoint) != metadata["sha256"]):
            raise ValueError("Inherited selected checkpoint binding changed")
        records[name] = {"prepared_path": str(prepared_path.relative_to(ROOT)),
                         "prepared_sha256": prepared_sha,
                         "discovery_path": str(discovery_path.relative_to(ROOT)),
                         "discovery_sha256": discovery_sha, "checkpoint": metadata}
    return base, {"parent_commit": protocol["parent_commit"],
                  "parent_manifest": str(path.relative_to(ROOT)), "parent_manifest_sha256": sha(path),
                  "parent_source_hashes_checked": len(parent["scientific_sources_sha256"]),
                  "parent_artifact_hashes_checked": len(parent["completed_sha256"]),
                  "ancestor_manifest": str(ancestor_path.relative_to(ROOT)),
                  "ancestor_manifest_sha256": sha(ancestor_path),
                  "ancestor_source_hashes_checked": len(ancestor["scientific_sources_sha256"]),
                  "ancestor_artifact_hashes_checked": len(ancestor["completed_sha256"]),
                  "parent_protocol": parent["protocol_config"], "model_records": records}


def validate_streams(base, protocol, parent_protocol):
    seeds = protocol["confirmation_seeds"]
    if (not seeds or len(seeds) != len(set(seeds))
            or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in seeds)):
        raise ValueError("Confirmation streams must contain unique nonnegative integer seeds")
    old = set(base["evaluation"]["seeds"])
    for name in ("discovery_seeds", "calibration_seeds", "confirmation_seeds"):
        old.update(base["diagnostics"][name])
    for start, count in ((base["training"]["train_seed"], base["training"]["episodes"]),
                         (base["training"]["validation_seed"], base["training"]["validation_episodes"])):
        old.update(range(start, start + count))
    for name in ("discovery_seeds", "confirmation_seeds"):
        old.update(parent_protocol[name])
    for name in ("calibration_seeds", "confirmation_seeds"):
        old.update(parent_protocol["dynamics"][name])
    if old.intersection(seeds):
        raise ValueError("New confirmation streams overlap inherited streams")


def validate_protocol(base, protocol, parent_protocol):
    validate_streams(base, protocol, parent_protocol)
    delays = protocol["delays"]
    if (not delays or any(isinstance(d, bool) or not isinstance(d, (int, float))
                          or not math.isfinite(d) or d < 0 for d in delays)
            or delays != sorted(set(delays))):
        raise ValueError("Delays must be finite, nonnegative, unique and ordered")
    if len({round(d * 1e9) for d in delays}) != len(delays):
        raise ValueError("Delays must remain distinct at nanosecond clock resolution")
    if protocol["delay_cue"] != base["delay"] or protocol["baseline_delay"] != base["delay"]:
        raise ValueError("Delay cue and baseline must equal the inherited training delay")
    if any(protocol[key] not in delays for key in ("baseline_delay", "primary_low_delay", "primary_high_delay")):
        raise ValueError("Baseline and primary endpoints must be in the delay grid")
    if protocol["primary_low_delay"] >= protocol["primary_high_delay"]:
        raise ValueError("Primary endpoints must be ordered")
    if base["period"] != 0.05:
        raise ValueError("This protocol requires the inherited 50 ms cadence")
    for key in ("workers", "cpu_threads"):
        if isinstance(protocol[key], bool) or not isinstance(protocol[key], int) or protocol[key] < 1:
            raise ValueError("Worker and thread counts must be positive integers")
    probe = protocol["probe"]
    if (any(not isinstance(probe[key], (int, float)) or isinstance(probe[key], bool)
            or not math.isfinite(probe[key]) for key in ("pulse_onset", "pulse_width", "duration"))
            or not (0 < probe["pulse_onset"] < probe["pulse_onset"] + probe["pulse_width"] < probe["duration"])):
        raise ValueError("Pulse and scoring window must lie within the horizon")
    amplitudes = probe["pulse_amplitudes"]
    if (not amplitudes or len(amplitudes) != len(set(amplitudes))
            or any(not isinstance(a, (int, float)) or isinstance(a, bool)
                   or not math.isfinite(a) or a == 0 for a in amplitudes)):
        raise ValueError("Pulse amplitudes must be distinct finite nonzero values")
    if not math.isfinite(protocol["metric_denominator_epsilon"]) or protocol["metric_denominator_epsilon"] <= 0:
        raise ValueError("Metric denominator floor must be positive")
    models_to_run(base, protocol)


def completed(path, manifest):
    path = Path(path)
    if not path.exists():
        return False
    if manifest["completed_sha256"].get(str(path.relative_to(ROOT))) != sha(path):
        raise ValueError(f"Existing result is unsealed or changed: {path}")
    return True


def seal(output, manifest, paths, stage=None):
    for path in paths:
        manifest["completed_sha256"][str(Path(path).relative_to(ROOT))] = sha(path)
    if stage and stage not in manifest["stages"]:
        manifest["stages"].append(stage)
    write_json(output / "run_manifest.json", manifest)


def initialize(output, artifacts, config_path, base, protocol, inherited):
    output.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    path = output / "run_manifest.json"
    sources = scientific_sources(config_path)
    if path.exists():
        manifest = read_json(path)
        if (manifest["scientific_sources_sha256"] != sources or manifest["protocol_config"] != protocol
                or manifest["base_config"] != base or manifest["inherited"] != inherited
                or manifest["artifacts_directory"] != str(artifacts)):
            raise ValueError("Frozen scientific inputs changed; use a fresh run directory")
        for relative, expected in manifest["completed_sha256"].items():
            if not (ROOT / relative).exists() or sha(ROOT / relative) != expected:
                raise ValueError(f"Completed artifact changed or disappeared: {relative}")
        return manifest
    if any(output.iterdir()) or any(artifacts.iterdir()):
        raise ValueError("A new experiment requires empty result and artifact directories")
    manifest = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "scientific_sources_sha256": sources, "protocol_config": protocol,
                "base_config": base, "inherited": inherited, "completed_sha256": {}, "stages": [],
                "artifacts_directory": str(artifacts),
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
                "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                "torch": torch.__version__, "workers": protocol["workers"],
                "threads_per_worker": protocol["cpu_threads"]}
    write_json(output / "config.json", protocol)
    write_json(output / "base_config.json", base)
    seal(output, manifest, [output / "config.json", output / "base_config.json"])
    return manifest


def _worker_init(threads):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)


def _compute(stage, job, base, protocol, inherited):
    from closed_loop_inhibition.delay_sweep import confirm_delay, passive_cell
    if stage == "passive":
        tau, noise_tau = job
        return passive_cell(base, protocol, tau, noise_tau)
    tau, noise_tau, seed, delay = job
    name = f"{cell_id(tau, noise_tau)}_seed_{seed}"
    metadata = inherited["model_records"][name]
    for key, digest in (("prepared_path", "prepared_sha256"), ("discovery_path", "discovery_sha256")):
        if sha(ROOT / metadata[key]) != metadata[digest]:
            raise ValueError("Frozen controller dependency changed during execution")
    checkpoint = ROOT / metadata["checkpoint"]["path"]
    if sha(checkpoint) != metadata["checkpoint"]["sha256"]:
        raise ValueError("Selected checkpoint changed during execution")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (payload["tau"], payload["noise_tau"], payload["seed"]) != (tau, noise_tau, seed):
        raise ValueError("Selected checkpoint belongs to another model cell")
    model = load_checkpoint(checkpoint, base)
    prepared = read_json(ROOT / metadata["prepared_path"])
    result = confirm_delay(base, protocol, tau, noise_tau, seed, model, prepared, delay)
    result["dependencies"] = metadata
    return result


def result_path(output, stage, job):
    if stage == "passive":
        return output / stage / f"{cell_id(*job)}.json"
    tau, noise_tau, seed, delay = job
    delay_label = f"{delay:.9f}".rstrip("0").rstrip(".")
    return output / "confirmation" / f"{cell_id(tau, noise_tau)}_seed_{seed}_delay_{delay_label}.json"


def validate_result(stage, job, record):
    actual = (record["tau"], record["noise_tau"])
    if stage == "confirm":
        actual += (record["seed"], record["delay"])
    if actual != tuple(job):
        raise ValueError("Worker result does not match its submitted job identity")


def run_stage(stage, base, protocol, inherited, output, manifest):
    expected = (sorted({(tau, noise_tau) for tau, noise_tau, _ in models_to_run(base, protocol)})
                if stage == "passive" else jobs(base, protocol))
    pending = [job for job in expected if not completed(result_path(output, stage, job), manifest)]
    done = len(expected) - len(pending)
    with ProcessPoolExecutor(max_workers=protocol["workers"], mp_context=multiprocessing.get_context("spawn"),
                             initializer=_worker_init, initargs=(protocol["cpu_threads"],)) as executor:
        futures = {executor.submit(_compute, stage, job, base, protocol, inherited): job for job in pending}
        for future in as_completed(futures):
            job = futures[future]
            result = future.result()
            validate_result(stage, job, result)
            path = result_path(output, stage, job)
            write_json(path, result)
            seal(output, manifest, [path])
            done += 1
            print(f"completed {stage} {done}/{len(expected)} {path.name}", flush=True)
    if not all(completed(result_path(output, stage, job), manifest) for job in expected):
        raise ValueError("Stage cannot complete with missing required results")
    seal(output, manifest, [], stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/delay_sweep.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/delay_sweep")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "runs/delay_sweep")
    parser.add_argument("--stage", choices=("all", "passive", "confirm"), default="all")
    args = parser.parse_args()
    protocol = read_json(args.config)
    base, inherited = verify_parent(protocol)
    validate_protocol(base, protocol, inherited["parent_protocol"])
    output, artifacts = args.output.resolve(), args.artifacts.resolve()
    manifest = initialize(output, artifacts, args.config, base, protocol, inherited)
    for stage in ("passive", "confirm"):
        if args.stage not in ("all", stage):
            continue
        if scientific_sources(args.config) != manifest["scientific_sources_sha256"]:
            raise ValueError("Scientific inputs changed during the experiment")
        print(f"stage_start {stage}", flush=True)
        run_stage(stage, base, protocol, inherited, output, manifest)
        if scientific_sources(args.config) != manifest["scientific_sources_sha256"]:
            raise ValueError("Scientific inputs changed during the experiment")
        print(f"stage_complete {stage}", flush=True)
    print(json.dumps({"complete_stages": manifest["stages"], "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
