"""Sealed calibration, confirmation and complete-loop mechanism experiment."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import importlib.util
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
PROTOCOL = ROOT / "docs/LOOP_MECHANISM.md"
INHERITED_RUNNERS = [ROOT / "experiments" / name for name in (
    "run_timescale_maps.py", "run_collective_suppression.py", "run_delay_sweep.py")]
STAGES = ("prepare", "passive", "confirm", "mechanism")
_spec = importlib.util.spec_from_file_location("_inherited_delay_runner", INHERITED_RUNNERS[-1])
_delay_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_delay_runner)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    # Serialize before opening anything: non-finite data cannot leave partial output.
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(data)
    temporary.replace(path)


def scientific_sources(config_path):
    paths = [*sorted((ROOT / "src").rglob("*.py")), Path(__file__).resolve(),
             *INHERITED_RUNNERS, PROTOCOL, Path(config_path).resolve()]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def models_to_run(base, protocol):
    full = list(itertools.product(protocol["plant_taus"], base["noise_taus"], base["seeds"]))
    if protocol["plant_taus"] != [.2]:
        raise ValueError("Mechanism experiment requires the 200 ms inherited plant")
    subset = protocol.get("model_subset")
    if subset is None:
        return full
    if not protocol.get("development", False):
        raise ValueError("A model subset requires an explicitly declared development run")
    chosen = [(row["tau"], row["noise_tau"], row["seed"]) for row in subset]
    if not chosen or len(set(chosen)) != len(chosen) or not set(chosen) <= set(full):
        raise ValueError("model_subset must contain unique inherited model cells")
    return chosen


def jobs(base, protocol):
    return [(*cell, delay) for cell in models_to_run(base, protocol) for delay in protocol["delays"]]


def stage_jobs(stage, base, protocol):
    if stage == "passive":
        return sorted({(tau, noise) for tau, noise, _ in models_to_run(base, protocol)})
    if stage not in STAGES:
        raise ValueError("Unknown experiment stage")
    return jobs(base, protocol)


def _verify_declared_hashes(manifest):
    for category in ("scientific_sources_sha256", "completed_sha256"):
        for relative, expected in manifest[category].items():
            path = ROOT / relative
            if not path.exists() or sha(path) != expected:
                raise ValueError(f"Inherited artifact or source changed: {relative}")


def verify_parent(protocol):
    # Reuse the frozen checkpoint/discovery binding implementation, supplying an
    # explicit subset because the earlier runner otherwise selects all plants.
    original_base = read_json(ROOT / protocol["parent_results"] / "base_config.json")
    legacy_protocol = deepcopy(protocol)
    legacy_protocol["model_subset"] = [dict(tau=tau, noise_tau=noise, seed=seed)
                                        for tau, noise, seed in models_to_run(original_base, protocol)]
    base, inherited = _delay_runner.verify_parent(legacy_protocol)
    directory = ROOT / protocol["delay_results"]
    path = directory / "run_manifest.json"
    committed = subprocess.check_output(
        ["git", "show", f"{protocol['delay_commit']}:{path.relative_to(ROOT)}"], cwd=ROOT)
    if path.read_bytes() != committed:
        raise ValueError("Delay parent manifest differs from its committed record")
    delay_parent = read_json(path)
    if set(delay_parent["stages"]) != {"passive", "confirm"}:
        raise ValueError("Delay parent experiment is incomplete")
    _verify_declared_hashes(delay_parent)
    if (read_json(directory / "config.json") != delay_parent["protocol_config"]
            or read_json(directory / "base_config.json") != base
            or delay_parent["base_config"] != base):
        raise ValueError("Delay parent frozen configuration differs")
    for key in ("parent_commit", "parent_manifest", "parent_manifest_sha256", "ancestor_manifest_sha256"):
        if delay_parent["inherited"][key] != inherited[key]:
            raise ValueError("Delay and collective parent lineage differ")
    for name, metadata in inherited["model_records"].items():
        if delay_parent["inherited"]["model_records"].get(name) != metadata:
            raise ValueError("Delay parent controller binding differs")
    inherited.update(delay_commit=protocol["delay_commit"],
                     delay_manifest=str(path.relative_to(ROOT)), delay_manifest_sha256=sha(path),
                     delay_source_hashes_checked=len(delay_parent["scientific_sources_sha256"]),
                     delay_artifact_hashes_checked=len(delay_parent["completed_sha256"]),
                     delay_protocol=delay_parent["protocol_config"])
    return base, inherited


def validate_streams(base, protocol, parent_protocol, delay_protocol):
    for name in ("calibration_seeds", "confirmation_seeds"):
        values = protocol[name]
        if (not values or len(values) != len(set(values))
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values)):
            raise ValueError("Streams must contain unique nonnegative integer seeds")
        # This checks every original training, validation, discovery, calibration
        # and evaluation stream family, including collective own-loop probes.
        _delay_runner.validate_streams(base, {"confirmation_seeds": values}, parent_protocol)
        if set(values).intersection(delay_protocol["confirmation_seeds"]):
            raise ValueError("New streams overlap inherited delay-sweep streams")
    if set(protocol["calibration_seeds"]).intersection(protocol["confirmation_seeds"]):
        raise ValueError("Calibration and confirmation streams must be disjoint")
    production_streams = set(range(5010001, 5010003)) | set(range(5030001, 5030005))
    if protocol.get("development", False) and production_streams.intersection(
            protocol["calibration_seeds"] + protocol["confirmation_seeds"]):
        raise ValueError("Development must not consume reserved production streams")


def _finite(value, *, positive=False, nonnegative=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            and (not positive or value > 0) and (not nonnegative or value >= 0))


def validate_protocol(base, protocol, parent_protocol, delay_protocol):
    validate_streams(base, protocol, parent_protocol, delay_protocol)
    if not isinstance(protocol.get("development", False), bool):
        raise ValueError("development must be a Boolean")
    delays = protocol["delays"]
    allowed = [0., .025, .05, .075, .1]
    if (not delays or any(not _finite(value, nonnegative=True) for value in delays)
            or delays != sorted(set(delays)) or not set(delays) <= set(allowed)):
        raise ValueError("Delays must be an ordered subset of the frozen 0–100 ms grid")
    if not protocol.get("development", False) and delays != allowed:
        raise ValueError("Production requires all five delays")
    if base["period"] != .05 or protocol["delay_cue"] != base["delay"]:
        raise ValueError("Cadence and delay cue must retain the inherited 50 ms timing")
    if protocol["baseline_delay"] != base["delay"]:
        raise ValueError("Baseline delay must retain the inherited training delay")
    low, high = protocol["primary_low_delay"], protocol["primary_high_delay"]
    if (not _finite(low, nonnegative=True) or not _finite(high, nonnegative=True)
            or low >= high or low not in delays or high not in delays):
        raise ValueError("Primary endpoints must be ordered and present in the delay grid")
    for key in ("workers", "cpu_threads"):
        value = protocol[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("Worker and thread counts must be positive integers")
    probe = protocol["probe"]
    if (any(not _finite(probe[key], positive=True) for key in ("duration", "pulse_onset", "pulse_width"))
            or probe["pulse_onset"] + probe["pulse_width"] >= probe["duration"]):
        raise ValueError("Pulse and scoring window must lie within the horizon")
    amplitudes = probe["pulse_amplitudes"]
    if (not amplitudes or len(set(amplitudes)) != len(amplitudes)
            or any(not _finite(value) or value == 0 for value in amplitudes)):
        raise ValueError("Pulse amplitudes must be distinct finite nonzero values")
    for key in ("baseline_epsilon", "metric_denominator_epsilon", "response_energy_floor", "match_tolerance"):
        if not _finite(protocol[key], positive=True):
            raise ValueError(f"{key} must be finite and positive")
    bounds = protocol["gain_bounds"]
    if len(bounds) != 2 or any(not _finite(value, positive=True) for value in bounds) or not bounds[0] < 1 < bounds[1]:
        raise ValueError("Gain bounds must be positive and strictly bracket one")
    for key in ("min_fractional_increase", "response_direction_epsilon"):
        if not _finite(protocol[key], nonnegative=True):
            raise ValueError(f"{key} must be finite and nonnegative")
    if not _finite(protocol["min_positive_fraction"], positive=True) or protocol["min_positive_fraction"] > 1:
        raise ValueError("min_positive_fraction must lie in (0, 1]")
    mechanism = protocol["mechanism"]
    for key in ("duration", "pulse_width", "parity_duration", "equilibrium_tolerance",
                "parity_state_tolerance", "parity_command_tolerance"):
        if not _finite(mechanism[key], positive=True):
            raise ValueError(f"mechanism.{key} must be finite and positive")
    if (mechanism["pulse_width"] >= mechanism["duration"]
            or mechanism["parity_duration"] <= base["history_seconds"]):
        raise ValueError("Mechanism horizons must contain the pulse and mature history")
    for key in ("duration", "pulse_width", "parity_duration"):
        periods = mechanism[key] / base["period"]
        if not math.isclose(periods, round(periods), abs_tol=1e-9, rel_tol=0.):
            raise ValueError("Mechanism horizons must be integer multiples of the decision period")
    frequencies = mechanism["frequencies"]
    if (not frequencies or any(not _finite(value, positive=True) or value >= .5 / base["period"] for value in frequencies)
            or frequencies != sorted(set(frequencies))):
        raise ValueError("Mechanism frequencies must be unique, ordered and below Nyquist")
    amplitudes = mechanism["pulse_amplitudes"]
    if (not amplitudes or len(amplitudes) != len(set(amplitudes))
            or any(not _finite(value) or value == 0 for value in amplitudes)
            or not any(abs(value) <= 1e-4 for value in amplitudes)):
        raise ValueError("Mechanism pulses must be distinct finite values including a small local probe")
    fraction = mechanism["settling_fraction"]
    if not _finite(fraction, positive=True) or fraction >= 1:
        raise ValueError("Settling fraction must lie in (0, 1)")
    models_to_run(base, protocol)


def result_path(output, stage, job):
    if stage == "passive":
        label = cell_id(*job)
    else:
        tau, noise, seed, delay = job
        delay_label = f"{delay:.9f}".rstrip("0").rstrip(".")
        label = f"{cell_id(tau, noise)}_seed_{seed}_delay_{delay_label}"
    return Path(output) / stage / f"{label}.json"


def completed(path, manifest):
    path = Path(path)
    if not path.exists():
        return False
    if manifest["completed_sha256"].get(str(path.relative_to(ROOT))) != sha(path):
        raise ValueError(f"Existing result is unsealed or changed: {path}")
    return True


def checked_record(path, manifest):
    if not completed(path, manifest):
        raise ValueError(f"Required dependency is absent: {path}")
    return read_json(path), sha(path)


def seal(output, manifest, paths, stage=None):
    for path in paths:
        manifest["completed_sha256"][str(Path(path).relative_to(ROOT))] = sha(path)
    if stage and stage not in manifest["stages"]:
        manifest["stages"].append(stage)
    write_json(Path(output) / "run_manifest.json", manifest)


def initialize(output, config_path, base, protocol, inherited):
    output = Path(output)
    output.relative_to(ROOT)  # Provenance keys and resume checks must remain rooted.
    output.mkdir(parents=True, exist_ok=True)
    path = output / "run_manifest.json"
    sources = scientific_sources(config_path)
    if path.exists():
        manifest = read_json(path)
        if (manifest["scientific_sources_sha256"] != sources or manifest["protocol_config"] != protocol
                or manifest["base_config"] != base or manifest["inherited"] != inherited):
            raise ValueError("Frozen scientific inputs changed; use a fresh run directory")
        for relative, expected in manifest["completed_sha256"].items():
            if not (ROOT / relative).exists() or sha(ROOT / relative) != expected:
                raise ValueError(f"Completed artifact changed or disappeared: {relative}")
        return manifest
    if any(output.iterdir()):
        raise ValueError("A new experiment requires an empty result directory")
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))
    if dirty and not protocol.get("development", False):
        raise ValueError("Production requires committed, clean scientific inputs before launch")
    manifest = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "scientific_sources_sha256": sources, "protocol_config": protocol,
                "base_config": base, "inherited": inherited, "completed_sha256": {}, "stages": [],
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "source_dirty": dirty, "python": sys.version, "platform": platform.platform(),
                "numpy": np.__version__, "torch": torch.__version__, "workers": protocol["workers"],
                "threads_per_worker": protocol["cpu_threads"], "stage_elapsed_seconds": {}}
    write_json(output / "config.json", protocol)
    write_json(output / "base_config.json", base)
    seal(output, manifest, [output / "config.json", output / "base_config.json"])
    return manifest


def require_all_prepared(base, protocol, output, manifest):
    if "prepare" not in manifest["stages"]:
        raise ValueError("All calibration records must be sealed before any confirmation or mechanism work")
    for job in jobs(base, protocol):
        if not completed(result_path(output, "prepare", job), manifest):
            raise ValueError("All calibration records must be sealed before any confirmation or mechanism work")


def _worker_init(threads):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)


def _compute(stage, job, base, protocol, inherited, preparation=None):
    start = time.perf_counter()
    if stage == "passive":
        from closed_loop_inhibition.delay_sweep import passive_cell
        result = passive_cell(base, protocol, *job)
    else:
        tau, noise, seed, delay = job
        metadata = inherited["model_records"][f"{cell_id(tau, noise)}_seed_{seed}"]
        for key, digest in (("prepared_path", "prepared_sha256"), ("discovery_path", "discovery_sha256")):
            if sha(ROOT / metadata[key]) != metadata[digest]:
                raise ValueError("Frozen controller dependency changed during execution")
        checkpoint = ROOT / metadata["checkpoint"]["path"]
        if sha(checkpoint) != metadata["checkpoint"]["sha256"]:
            raise ValueError("Selected checkpoint changed during execution")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (payload["tau"], payload["noise_tau"], payload["seed"]) != (tau, noise, seed):
            raise ValueError("Selected checkpoint belongs to another model cell")
        model = load_checkpoint(checkpoint, base)
        if stage == "prepare":
            from closed_loop_inhibition.gain_rescue import prepare_rescue
            result = prepare_rescue(base, protocol, tau, noise, seed, model,
                                    read_json(ROOT / metadata["prepared_path"]), delay)
        else:
            if preparation is None or sha(ROOT / preparation["path"]) != preparation["sha256"]:
                raise ValueError("Sealed calibration dependency changed during execution")
            prepared = read_json(ROOT / preparation["path"])
            if stage == "confirm":
                from closed_loop_inhibition.gain_rescue import confirm_rescue
                result = confirm_rescue(base, protocol, tau, noise, seed, model, prepared, delay)
            elif stage == "mechanism":
                from closed_loop_inhibition.loop_mechanism import analyze_model
                result = analyze_model(base, protocol, tau, noise, seed, model, prepared, delay)
            else:
                raise ValueError("Unknown computation stage")
        result["dependencies"] = deepcopy(metadata)
        if preparation is not None:
            result["dependencies"]["calibration"] = preparation
    result["elapsed_seconds"] = time.perf_counter() - start
    return result


def validate_result(stage, job, record, protocol=None):
    actual = (record["tau"], record["noise_tau"])
    if stage != "passive":
        actual += (record["seed"], record["delay"])
    if actual != tuple(job):
        raise ValueError("Worker result does not match its submitted job identity")
    count = record["physical_rollouts"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("Worker must declare a nonnegative integer physical rollout count")
    json.dumps(record, allow_nan=False)
    if stage != "mechanism":
        rollout_counts(record)
    if protocol is None or stage == "mechanism":
        return
    if stage in ("prepare", "confirm"):
        available = record["variants" if stage == "prepare" else "frozen_settings"]
        missing = record["unavailable_controls"]
        required = {"native", "joint_weak", "joint_strong"}
        controls = {"gain_matched", "weak_rescue"}
        if (not required <= set(available) or not set(missing) <= controls
                or set(available).intersection(missing)
                or set(available) | set(missing) != required | controls
                or any(not reason for reason in missing.values())):
            raise ValueError("Available and unavailable variants must account for all five policies")
        if stage == "confirm":
            if set(record["summary"]) != required | controls:
                raise ValueError("Confirmation summaries must explicitly account for every variant")
            if any((record["summary"][name] is None) != (name in missing) for name in required | controls):
                raise ValueError("Unavailable controls require null summaries, available controls numeric records")
    data = record.get("calibration", record)
    labels = {"native"} if stage == "prepare" else ({"zero_policy"} if stage == "passive" else set(available))
    seeds = protocol["calibration_seeds" if stage == "prepare" else "confirmation_seeds"]
    expected = set(itertools.product(labels, seeds, protocol["probe"]["pulse_amplitudes"]))
    observed = {(row["variant"], row["noise_seed"], row["amplitude"]) for row in data["rollouts"]}
    if observed != expected:
        raise ValueError("Physical records do not cover every declared variant, stream and pulse")
    timing = [(row["variant"], row["noise_seed"], row["amplitude"]) for row in data["timing"]]
    expected_timing = expected | set(itertools.product(labels, seeds, [0.]))
    if len(timing) != len(set(timing)) or set(timing) != expected_timing:
        raise ValueError("Timing audits must cover each unique physical trial exactly once")
    if data["physical_rollouts"] != count or count != len(expected_timing):
        raise ValueError("Physical rollout count differs from the declared protocol coverage")


def run_stage(stage, base, protocol, inherited, output, manifest):
    if stage != "prepare":
        require_all_prepared(base, protocol, output, manifest)
    expected = stage_jobs(stage, base, protocol)
    pending = [job for job in expected if not completed(result_path(output, stage, job), manifest)]
    done = len(expected) - len(pending)
    start = time.perf_counter()
    if pending:
        with ProcessPoolExecutor(max_workers=protocol["workers"], mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_worker_init, initargs=(protocol["cpu_threads"],)) as executor:
            futures = {}
            for job in pending:
                preparation = None
                if stage in ("confirm", "mechanism"):
                    path = result_path(output, "prepare", job)
                    _, digest = checked_record(path, manifest)
                    preparation = {"path": str(path.relative_to(ROOT)), "sha256": digest}
                future = executor.submit(_compute, stage, job, base, protocol, inherited, preparation)
                futures[future] = job
            for future in as_completed(futures):
                job = futures[future]
                result = future.result()
                validate_result(stage, job, result, protocol)
                path = result_path(output, stage, job)
                write_json(path, result)
                seal(output, manifest, [path])
                done += 1
                print(f"completed {stage} {done}/{len(expected)} {path.name}", flush=True)
    if not all(completed(result_path(output, stage, job), manifest) for job in expected):
        raise ValueError("Stage cannot complete with missing required results")
    manifest["stage_elapsed_seconds"][stage] = manifest["stage_elapsed_seconds"].get(stage, 0.) + time.perf_counter() - start
    seal(output, manifest, [], stage)


def rollout_counts(record):
    """Count unique physical trials, without double-counting shared shams."""
    data = record.get("calibration", record)
    trials = {}
    pulse_rows = set()
    for row in data["rollouts"]:
        pulse_key = (row["variant"], row["noise_seed"], row["amplitude"])
        if pulse_key in pulse_rows or row["amplitude"] == 0:
            raise ValueError("Pulse records must be unique and nonzero")
        pulse_rows.add(pulse_key)
        for arm, amplitude in (("sham", 0.), ("pulse", row["amplitude"])):
            key = (row["variant"], row["noise_seed"], amplitude)
            stats = row[arm]
            if key in trials and trials[key] != stats:
                raise ValueError("A reused physical sham has inconsistent statistics")
            trials[key] = stats
    if len(trials) != record["physical_rollouts"]:
        raise ValueError("Declared physical trial count differs from unique recorded trials")
    for stats in trials.values():
        if stats["censored"]:
            if stats["position_rms"] is not None or not stats["task_failure"]:
                raise ValueError("Censored trials must retain null metrics and failure status")
        elif not _finite(stats["position_rms"], nonnegative=True):
            raise ValueError("Complete trials, including failures, must retain numeric metrics")
    return {"physical_rollouts": len(trials),
            "task_failures": sum(bool(stats["task_failure"]) for stats in trials.values()),
            "censored": sum(bool(stats["censored"]) for stats in trials.values()),
            "complete": sum(not stats["censored"] for stats in trials.values())}


def write_audit(base, protocol, output, manifest):
    audit = {"stages": {}, "all_required_stages_complete": set(manifest["stages"]) == set(STAGES)}
    for stage in manifest["stages"]:
        counts = {"records": 0, "physical_rollouts": 0, "task_failures": 0, "censored": 0, "complete": 0}
        statuses = {}
        for job in stage_jobs(stage, base, protocol):
            record, _ = checked_record(result_path(output, stage, job), manifest)
            validate_result(stage, job, record, protocol)
            counts["records"] += 1
            if stage == "mechanism":
                counts["physical_rollouts"] += record["physical_rollouts"]
                statuses[record.get("status", "unspecified")] = statuses.get(record.get("status", "unspecified"), 0) + 1
            else:
                for key, value in rollout_counts(record).items():
                    counts[key] += value
        if stage == "mechanism":
            # Simulator-parity runs are accounted separately from stochastic
            # task trials; their numerical diagnostics remain in each record.
            counts = {key: value for key, value in counts.items() if key not in ("task_failures", "censored", "complete")}
            counts["record_statuses"] = statuses
        audit["stages"][stage] = counts
    audit["stochastic_physical_rollouts"] = sum(value["physical_rollouts"] for stage, value in audit["stages"].items() if stage != "mechanism")
    audit["simulator_validation_rollouts"] = audit["stages"].get("mechanism", {}).get("physical_rollouts", 0)
    path = Path(output) / "execution_audit.json"
    write_json(path, audit)
    seal(output, manifest, [path])
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/loop_mechanism.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/loop_mechanism")
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    args = parser.parse_args()
    protocol = read_json(args.config)
    base, inherited = verify_parent(protocol)
    validate_protocol(base, protocol, inherited["parent_protocol"], inherited["delay_protocol"])
    output = args.output.resolve()
    manifest = initialize(output, args.config, base, protocol, inherited)
    for stage in STAGES:
        if args.stage not in ("all", stage):
            continue
        if scientific_sources(args.config) != manifest["scientific_sources_sha256"]:
            raise ValueError("Scientific inputs changed during the experiment")
        print(f"stage_start {stage}", flush=True)
        run_stage(stage, base, protocol, inherited, output, manifest)
        if scientific_sources(args.config) != manifest["scientific_sources_sha256"]:
            raise ValueError("Scientific inputs changed during the experiment")
        print(f"stage_complete {stage}", flush=True)
    audit = write_audit(base, protocol, output, manifest)
    print(json.dumps({"complete_stages": manifest["stages"], "output": str(output), "audit": audit}), flush=True)


if __name__ == "__main__":
    main()
