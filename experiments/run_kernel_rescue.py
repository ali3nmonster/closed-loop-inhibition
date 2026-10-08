"""Sealed 50 ms kernel corrections and independent training replication."""

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
from closed_loop_inhibition.collective_suppression import build_bank, bank_metadata
from closed_loop_inhibition.kernel_training import training_config, train_replica, discover_selected, prepare_fresh_settings

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/KERNEL_RESCUE.md"
INHERITED_RUNNERS = [ROOT / "experiments" / name for name in (
    "run_timescale_maps.py", "run_collective_suppression.py", "run_delay_sweep.py", "run_loop_mechanism.py")]
STAGES = ("discovery_bank", "fresh_train", "fresh_discovery", "prepare", "passive", "confirm", "mechanism")
VARIANTS = {"native", "joint_weak", "weak_equilibrium", "weak_scalar", "weak_kernel"}
_spec = importlib.util.spec_from_file_location("_kernel_inherited_runner", INHERITED_RUNNERS[-1])
_old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_old)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(data)
    temporary.replace(path)


def scientific_sources(config_path):
    paths = [*sorted((ROOT / "src").rglob("*.py")), Path(__file__).resolve(),
             *INHERITED_RUNNERS, PROTOCOL, Path(config_path).resolve()]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def models_to_run(base, protocol):
    full = [(population, tau, noise, seed)
            for population, seeds in (("existing", base["seeds"]), ("fresh", protocol["fresh_seeds"]))
            for tau, noise, seed in itertools.product(protocol["plant_taus"], base["noise_taus"], seeds)]
    subset = protocol.get("model_subset")
    if subset is None:
        return full
    if not protocol.get("development", False):
        raise ValueError("A subset requires an explicitly declared development run")
    chosen = [(row["population"], row["tau"], row["noise_tau"], row["seed"]) for row in subset]
    if not chosen or len(chosen) != len(set(chosen)) or not set(chosen) <= set(full):
        raise ValueError("Model subset must contain unique declared population cells")
    return chosen


def stage_jobs(stage, base, protocol):
    models = models_to_run(base, protocol)
    fresh = [model for model in models if model[0] == "fresh"]
    if stage == "discovery_bank":
        return [(tau,) for tau in sorted({model[1] for model in fresh})]
    if stage in ("fresh_train", "fresh_discovery"):
        return fresh
    if stage == "prepare":
        return models
    if stage == "confirm":
        return [(*model, delay, duration) for model in models
                for delay in protocol["delays"] for duration in protocol["confirmation_durations"]]
    if stage == "mechanism":
        return [(*model, delay) for model in models for delay in protocol["delays"]]
    if stage == "passive":
        return [(tau, noise, duration) for tau, noise in sorted({(m[1], m[2]) for m in models})
                for duration in protocol["confirmation_durations"]]
    raise ValueError("Unknown stage")


def verify_parent(protocol):
    inherited_protocol = deepcopy(protocol)
    inherited_protocol.pop("model_subset", None)
    base, inherited = _old.verify_parent(inherited_protocol)
    directory = ROOT / protocol["loop_results"]
    path = directory / "run_manifest.json"
    committed = subprocess.check_output(["git", "show", f"{protocol['loop_commit']}:{path.relative_to(ROOT)}"], cwd=ROOT)
    if path.read_bytes() != committed:
        raise ValueError("Loop parent manifest differs from its committed record")
    parent = read_json(path)
    if set(parent["stages"]) != set(_old.STAGES):
        raise ValueError("Loop parent experiment is incomplete")
    for category in ("scientific_sources_sha256", "completed_sha256"):
        for relative, expected in parent[category].items():
            target = ROOT / relative
            if not target.exists() or sha(target) != expected:
                raise ValueError(f"Inherited artifact or source changed: {relative}")
    if (parent["base_config"] != base or read_json(directory / "base_config.json") != base
            or read_json(directory / "config.json") != parent["protocol_config"]):
        raise ValueError("Loop parent frozen configuration differs")
    for key in ("parent_manifest_sha256", "ancestor_manifest_sha256", "delay_manifest_sha256", "model_records"):
        if parent["inherited"][key] != inherited[key]:
            raise ValueError("Loop and earlier lineage bindings differ")
    inherited.update(loop_commit=protocol["loop_commit"], loop_manifest=str(path.relative_to(ROOT)),
                     loop_manifest_sha256=sha(path), loop_protocol=parent["protocol_config"],
                     loop_source_hashes_checked=len(parent["scientific_sources_sha256"]),
                     loop_artifact_hashes_checked=len(parent["completed_sha256"]))
    return base, inherited


def _finite(value, positive=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            and (not positive or value > 0))


def validate_streams(base, protocol, inherited):
    names = ("discovery_seeds", "offset_calibration_seeds", "calibration_seeds", "confirmation_seeds")
    streams = []
    for name in names:
        values = protocol[name]
        if not values or len(values) != len(set(values)) or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            raise ValueError("Streams must contain distinct nonnegative integer seeds")
        _old._delay_runner.validate_streams(base, {"confirmation_seeds": values}, inherited["parent_protocol"])
        old_streams = list(inherited["delay_protocol"]["confirmation_seeds"])
        old_streams += inherited["loop_protocol"]["calibration_seeds"] + inherited["loop_protocol"]["confirmation_seeds"]
        if set(values) & set(old_streams):
            raise ValueError("New streams overlap inherited delay or mechanism streams")
        streams.append(set(values))
    settings = protocol["training_streams"]
    for key in ("train_base", "validation_base", "stride"):
        if isinstance(settings[key], bool) or not isinstance(settings[key], int) or settings[key] <= 0:
            raise ValueError("Training stream bases and stride must be positive integers")
    if settings["stride"] < max(base["training"]["episodes"], base["training"]["validation_episodes"]):
        raise ValueError("Training stride must separate complete episode families")
    for seed, noise in itertools.product(protocol["fresh_seeds"], base["noise_taus"]):
        config = training_config(base, protocol, noise, seed)
        for name, count in (("train", "episodes"), ("validation", "validation_episodes")):
            first = config["training"][f"{name}_seed"]
            values = set(range(first, first + config["training"][count]))
            _old._delay_runner.validate_streams(base, {"confirmation_seeds": sorted(values)}, inherited["parent_protocol"])
            if values & set(old_streams):
                raise ValueError("New training tapes overlap inherited streams")
            streams.append(values)
    if any(streams[a] & streams[b] for a in range(len(streams)) for b in range(a + 1, len(streams))):
        raise ValueError("All discovery, calibration, confirmation and training streams must be disjoint")
    if protocol.get("development", False) and any(6000000 <= value < 7000000 for values in streams for value in values):
        raise ValueError("Development must not consume reserved production streams")


def validate_protocol(base, protocol, inherited):
    if not isinstance(protocol.get("development", False), bool):
        raise ValueError("development must be Boolean")
    if protocol["plant_taus"] != [.2] or protocol["delays"] != [.05, .1]:
        raise ValueError("Kernel rescue requires the 200 ms plant and 50/100 ms endpoints")
    if (base["period"] != .05 or protocol["delay_cue"] != .05 or protocol["calibration_delay"] != .05
            or protocol["baseline_delay"] != .05 or protocol["primary_low_delay"] != .05
            or protocol["primary_high_delay"] != .1):
        raise ValueError("Calibration, cadence, fixed cue and primary endpoints must remain frozen")
    if protocol["confirmation_durations"] != [4., 12.]:
        raise ValueError("Require 4 s primary and 12 s secondary horizons")
    seeds = protocol["fresh_seeds"]
    if (not seeds or len(seeds) != len(set(seeds)) or set(seeds) & set(base["seeds"])
            or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)):
        raise ValueError("Fresh initialization seeds must be distinct nonnegative integers")
    if not protocol.get("development", False) and seeds != [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]:
        raise ValueError("Production requires all ten fresh initialization seeds")
    if not protocol.get("development", False) and "development_training" in protocol:
        raise ValueError("Production cannot override inherited training choices")
    for key in ("workers", "cpu_threads"):
        if isinstance(protocol[key], bool) or not isinstance(protocol[key], int) or protocol[key] < 1:
            raise ValueError("Worker and thread counts must be positive integers")
    probe = protocol["probe"]
    if (probe["duration"] != 4. or probe["pulse_onset"] != 1. or probe["pulse_width"] != .1
            or probe["pulse_amplitudes"] != [-.02, .02]):
        raise ValueError("Physical probes must retain the prospectively declared pulses")
    for key in ("baseline_epsilon", "metric_denominator_epsilon", "response_energy_floor", "match_tolerance"):
        if not _finite(protocol[key], positive=True):
            raise ValueError(f"{key} must be finite and positive")
    if (protocol["weak_scale"] != .9 or protocol["min_fractional_increase"] != .01
            or protocol["min_positive_fraction"] != .75 or protocol["baseline_epsilon"] != .0001):
        raise ValueError("Fresh discovery must retain the inherited selection rule")
    if protocol["discovery_probe"] != inherited["parent_protocol"]["probe"]:
        raise ValueError("Fresh discovery must retain the inherited common teacher probe")
    bounds = protocol["gain_bounds"]
    if len(bounds) != 2 or any(not _finite(value, positive=True) for value in bounds) or not bounds[0] < 1 < bounds[1]:
        raise ValueError("Gain bounds must strictly bracket one")
    mechanism = protocol["mechanism"]
    for key in ("duration", "pulse_width", "parity_duration", "equilibrium_tolerance", "verification_amplitude",
                "linearization_relative_tolerance", "kernel_jacobian_tolerance", "parity_state_tolerance",
                "parity_command_tolerance"):
        if not _finite(mechanism[key], positive=True):
            raise ValueError(f"mechanism.{key} must be finite and positive")
    if mechanism["pulse_width"] >= mechanism["duration"] or mechanism["parity_duration"] <= base["history_seconds"]:
        raise ValueError("Mechanism horizons must contain mature history and the complete pulse")
    for key in ("duration", "pulse_width", "parity_duration"):
        periods = mechanism[key] / base["period"]
        if not math.isclose(periods, round(periods), abs_tol=1e-9, rel_tol=0.):
            raise ValueError("Mechanism horizons must be integer decision periods")
    frequencies = mechanism["frequencies"]
    if (not frequencies or frequencies != sorted(set(frequencies)) or any(
            not _finite(value, positive=True) or value >= .5 / base["period"] for value in frequencies)):
        raise ValueError("Mechanism frequencies must be distinct ordered positive values below Nyquist")
    if mechanism["pulse_amplitudes"] != [-.02, .02]:
        raise ValueError("Mechanism finite disturbances must retain both declared pulse signs")
    fraction = mechanism["settling_fraction"]
    if not _finite(fraction, positive=True) or fraction >= 1:
        raise ValueError("Settling fraction must lie strictly between zero and one")
    statistics = protocol["statistics"]
    for key in ("bootstrap_seed", "bootstrap_resamples"):
        if isinstance(statistics[key], bool) or not isinstance(statistics[key], int) or statistics[key] <= 0:
            raise ValueError("Bootstrap seed and resample count must be positive integers")
    if not _finite(statistics["confidence_level"], positive=True) or statistics["confidence_level"] >= 1:
        raise ValueError("Bootstrap confidence level must lie strictly between zero and one")
    if (not protocol.get("development", False) and statistics !=
            dict(bootstrap_seed=6070001, bootstrap_resamples=10000, confidence_level=.95)):
        raise ValueError("Production bootstrap settings must retain the prospective specification")
    for key, value in protocol.get("development_training", {}).items():
        if key not in ("epochs", "intermediate_epoch") or isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("Development training allows positive epoch counts only")
    config = training_config(base, protocol, base["noise_taus"][0], seeds[0])
    if config["training"]["intermediate_epoch"] > config["training"]["epochs"]:
        raise ValueError("Intermediate checkpoint must fall within the training budget")
    validate_streams(base, protocol, inherited)
    models_to_run(base, protocol)


def label(model):
    population, tau, noise, seed = model
    return f"{population}_{cell_id(tau, noise)}_seed_{seed}"


def _number(value):
    return f"{value:.9f}".rstrip("0").rstrip(".")


def result_path(output, stage, job):
    if stage == "discovery_bank":
        name = f"tau_{job[0]:g}"
    elif stage == "passive":
        name = f"{cell_id(*job[:2])}_duration_{_number(job[2])}"
    else:
        name = label(job[:4])
        if stage in ("confirm", "mechanism"):
            name += f"_delay_{_number(job[4])}"
        if stage == "confirm":
            name += f"_duration_{_number(job[5])}"
    return Path(output) / stage / f"{name}.json"


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
    return read_json(path), {"path": str(Path(path).relative_to(ROOT)), "sha256": sha(path)}


def seal(output, manifest, paths, stage=None):
    for path in paths:
        manifest["completed_sha256"][str(Path(path).relative_to(ROOT))] = sha(path)
    if stage and stage not in manifest["stages"]:
        manifest["stages"].append(stage)
    write_json(Path(output) / "run_manifest.json", manifest)


def initialize(output, artifacts, config_path, base, protocol, inherited):
    output, artifacts = Path(output), Path(artifacts)
    output.relative_to(ROOT)
    artifacts.relative_to(ROOT)
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
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))
    if dirty and not protocol.get("development", False):
        raise ValueError("Production requires committed, clean scientific inputs before launch")
    manifest = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "scientific_sources_sha256": sources, "protocol_config": protocol, "base_config": base,
                "inherited": inherited, "completed_sha256": {}, "stages": [],
                "artifacts_directory": str(artifacts),
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "source_dirty": dirty, "python": sys.version, "platform": platform.platform(),
                "numpy": np.__version__, "torch": torch.__version__, "workers": protocol["workers"],
                "threads_per_worker": protocol["cpu_threads"], "stage_elapsed_seconds": {}}
    write_json(output / "config.json", protocol)
    write_json(output / "base_config.json", base)
    seal(output, manifest, [output / "config.json", output / "base_config.json"])
    return manifest


def require_stage(stage, base, protocol, output, manifest):
    if stage not in manifest["stages"] or not all(completed(result_path(output, stage, job), manifest)
                                                 for job in stage_jobs(stage, base, protocol)):
        raise ValueError(f"All {stage} records must be sealed before dependent work")


def dependencies(stage, job, base, protocol, inherited, output, artifacts, manifest):
    if stage in ("discovery_bank", "passive"):
        return {}
    model = job[:4]
    population, tau, noise, seed = model
    if stage == "fresh_train":
        directory = artifacts / label(model)
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("Unsealed training artifacts require inspection before retry")
        return {"artifact_directory": str(directory)}
    if population == "existing":
        metadata = deepcopy(inherited["model_records"][f"{cell_id(tau, noise)}_seed_{seed}"])
        metadata["load_config"] = base
    else:
        record, binding = checked_record(result_path(output, "fresh_train", model), manifest)
        metadata = {"training": binding, "load_config": record["training_config"], "training_status": record["status"]}
        if record["status"] != "trained":
            return metadata
        metadata["checkpoint"] = {"path": str(Path(record["summary"]["checkpoint"]).relative_to(ROOT)),
                                  "sha256": record["summary"]["checkpoint_sha256"]}
        if stage == "fresh_discovery":
            _, metadata["discovery_bank"] = checked_record(result_path(output, "discovery_bank", (tau,)), manifest)
        else:
            _, metadata["fresh_discovery"] = checked_record(result_path(output, "fresh_discovery", model), manifest)
    if stage in ("confirm", "mechanism"):
        _, metadata["calibration"] = checked_record(result_path(output, "prepare", model), manifest)
    return metadata


def _read_binding(binding):
    path = ROOT / binding["path"]
    if not path.exists() or sha(path) != binding["sha256"]:
        raise ValueError("Frozen dependency changed during execution")
    return path


def _worker_init(threads):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)


def _discovery_protocol(inherited, protocol):
    result = deepcopy(inherited["parent_protocol"])
    result["discovery_seeds"] = protocol["discovery_seeds"]
    result["probe"] = deepcopy(protocol["discovery_probe"])
    return result


def _unavailable(stage, job, reason):
    population, tau, noise, seed = job[:4]
    result = dict(population=population, tau=tau, noise_tau=noise, seed=seed,
                  status="upstream_unavailable", failure=reason, physical_rollouts=0,
                  unavailable_controls={name: reason for name in VARIANTS})
    if stage == "prepare":
        result["variants"] = {}
    if stage in ("confirm", "mechanism"):
        result["delay"] = job[4]
        result["frozen_settings"] = {}
    if stage == "confirm":
        result.update(duration=job[5], summary={name: None for name in VARIANTS}, rollouts=[], timing=[])
    return result


def _compute(stage, job, base, protocol, inherited, metadata, artifacts):
    start = time.perf_counter()
    extra_paths = []
    if stage == "discovery_bank":
        bank = build_bank(base, _discovery_protocol(inherited, protocol), job[0], protocol["discovery_seeds"])
        path = Path(artifacts) / "discovery_bank" / f"tau_{job[0]:g}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise ValueError("An unsealed discovery bank requires inspection")
        arrays = {key: value for key, value in bank.items() if key not in ("pulse", "sham")}
        for arm in ("pulse", "sham"):
            arrays[f"{arm}_tokens"], arrays[f"{arm}_valid"] = bank[arm]
        np.savez_compressed(path, **arrays)
        result = dict(tau=job[0], bank={"path": str(path.relative_to(ROOT)), "sha256": sha(path)},
                      metadata=bank_metadata(bank), physical_rollouts=len(protocol["discovery_seeds"]) *
                      (1 + len(inherited["parent_protocol"]["probe"]["amplitudes"])))
        extra_paths.append(str(path))
    elif stage == "passive":
        from closed_loop_inhibition.delay_sweep import passive_cell
        config = deepcopy(protocol)
        config["probe"]["duration"] = job[2]
        result = passive_cell(base, config, *job[:2])
        result["duration"] = job[2]
    else:
        population, tau, noise, seed = job[:4]
        if stage == "fresh_train":
            directory = Path(metadata["artifact_directory"])
            result = train_replica(base, protocol, tau, noise, seed, directory)
            extra_paths += [str(path) for path in sorted(directory.rglob("*")) if path.is_file()]
        elif metadata.get("training_status") == "training_failed":
            result = _unavailable(stage, job, "training_failed")
        else:
            checkpoint = _read_binding(metadata["checkpoint"])
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if (payload["tau"], payload["noise_tau"], payload["seed"]) != (tau, noise, seed):
                raise ValueError("Selected checkpoint belongs to another model cell")
            model = load_checkpoint(checkpoint, metadata["load_config"])
            if stage == "fresh_discovery":
                bank_record = read_json(_read_binding(metadata["discovery_bank"]))
                with np.load(_read_binding(bank_record["bank"]), allow_pickle=False) as saved:
                    bank = {key: saved[key].copy() for key in saved.files if not key.startswith(("pulse_", "sham_"))}
                    for arm in ("pulse", "sham"):
                        bank[arm] = saved[f"{arm}_tokens"].copy(), saved[f"{arm}_valid"].copy()
                discovery = discover_selected(base, _discovery_protocol(inherited, protocol), tau, noise, seed, model, bank)
                settings = prepare_fresh_settings(base, protocol, tau, noise, seed, model, discovery)
                result = {**discovery, "settings": settings, "physical_rollouts": settings["physical_rollouts"],
                          "calibration": settings["calibration"]}
            elif stage == "prepare":
                from closed_loop_inhibition.kernel_rescue import prepare_kernel
                if population == "existing":
                    settings = read_json(_read_binding({"path": metadata["prepared_path"], "sha256": metadata["prepared_sha256"]}))
                    _read_binding({"path": metadata["discovery_path"], "sha256": metadata["discovery_sha256"]})
                else:
                    settings = read_json(_read_binding(metadata["fresh_discovery"]))["settings"]
                if settings.get("status") == "offset_calibration_failed":
                    result = _unavailable(stage, job, "offset_calibration_failed")
                else:
                    result = prepare_kernel(base, protocol, tau, noise, seed, model, settings)
            else:
                from closed_loop_inhibition.kernel_rescue import confirm_kernel, analyze_kernel
                prepared = read_json(_read_binding(metadata["calibration"]))
                if prepared["status"] == "upstream_unavailable":
                    result = _unavailable(stage, job, prepared["failure"])
                elif stage == "confirm":
                    result = confirm_kernel(base, protocol, tau, noise, seed, model, prepared, job[4], duration=job[5])
                    result["duration"] = job[5]
                elif stage == "mechanism":
                    result = analyze_kernel(base, protocol, tau, noise, seed, model, prepared, job[4])
                else:
                    raise ValueError("Unknown computation stage")
        result["population"] = population
        result["dependencies"] = deepcopy(metadata)
    result["elapsed_seconds"] = time.perf_counter() - start
    return result, extra_paths


def validate_result(stage, job, record, protocol):
    if stage == "discovery_bank":
        actual = (record["tau"],)
    elif stage == "passive":
        actual = record["tau"], record["noise_tau"], record["duration"]
    else:
        actual = record["population"], record["tau"], record["noise_tau"], record["seed"]
        if stage in ("confirm", "mechanism"):
            actual += (record["delay"],)
        if stage == "confirm":
            actual += (record["duration"],)
    if actual != tuple(job):
        raise ValueError("Worker result differs from its submitted job identity")
    json.dumps(record, allow_nan=False)
    count = record["physical_rollouts"]
    if stage == "fresh_train" and record.get("status") == "training_failed":
        if count is not None:
            raise ValueError("A failed training attempt requires an explicitly unknown physical trial count")
        return
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("Physical rollout count must be a nonnegative integer")
    if record.get("status") == "upstream_unavailable":
        if count != 0 or set(record["unavailable_controls"]) != VARIANTS:
            raise ValueError("Unavailable models must retain all five explicit unavailable variants")
        if stage == "confirm" and (set(record["summary"]) != VARIANTS or any(value is not None for value in record["summary"].values())):
            raise ValueError("Unavailable confirmation requires null summaries for all variants")
        return
    if stage in ("discovery_bank", "fresh_train", "mechanism"):
        if stage == "fresh_train":
            if record.get("status") != "trained" or len(record["history"]) != record["training_config"]["training"]["epochs"]:
                raise ValueError("Successful training requires every epoch and its frozen configuration")
            if count != (record["training_config"]["training"]["episodes"] +
                         record["training_config"]["training"]["validation_episodes"]):
                raise ValueError("Training count must include every teacher demonstration")
        return
    _old.rollout_counts(record)
    data = record.get("calibration", record)
    if stage in ("prepare", "confirm"):
        available = record["variants" if stage == "prepare" else "frozen_settings"]
        missing = record["unavailable_controls"]
        if set(available) & set(missing) or set(available) | set(missing) != VARIANTS:
            raise ValueError("Every variant must be available or explicitly unavailable")
        if stage == "confirm" and (set(record["summary"]) != VARIANTS or any(
                (record["summary"][name] is None) != (name in missing) for name in VARIANTS)):
            raise ValueError("Confirmation must retain numeric or null summaries for every variant")
    labels = {"native"} if stage in ("fresh_discovery", "prepare") else ({"zero_policy"} if stage == "passive" else set(available))
    stream_key = "offset_calibration_seeds" if stage == "fresh_discovery" else "calibration_seeds" if stage == "prepare" else "confirmation_seeds"
    expected = set(itertools.product(labels, protocol[stream_key], protocol["probe"]["pulse_amplitudes"]))
    observed = {(row["variant"], row["noise_seed"], row["amplitude"]) for row in data["rollouts"]}
    if observed != expected:
        raise ValueError("Physical records omit declared variants, streams or pulse signs")
    timing = [(row["variant"], row["noise_seed"], row["amplitude"]) for row in data["timing"]]
    expected_timing = expected | set(itertools.product(labels, protocol[stream_key], [0.]))
    if len(timing) != len(set(timing)) or set(timing) != expected_timing or count != len(timing):
        raise ValueError("Timing audits must cover every unique physical trial exactly once")


def run_stage(stage, base, protocol, inherited, output, artifacts, manifest):
    required = {"fresh_discovery": ("fresh_train", "discovery_bank"), "prepare": ("fresh_discovery",),
                "passive": ("prepare",), "confirm": ("prepare",), "mechanism": ("prepare",)}
    for dependency in required.get(stage, ()):
        require_stage(dependency, base, protocol, output, manifest)
    expected = stage_jobs(stage, base, protocol)
    pending = [job for job in expected if not completed(result_path(output, stage, job), manifest)]
    done = len(expected) - len(pending)
    start = time.perf_counter()
    if pending:
        with ProcessPoolExecutor(max_workers=protocol["workers"], mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_worker_init, initargs=(protocol["cpu_threads"],)) as executor:
            futures = {}
            for job in pending:
                metadata = dependencies(stage, job, base, protocol, inherited, output, artifacts, manifest)
                future = executor.submit(_compute, stage, job, base, protocol, inherited, metadata, str(artifacts))
                futures[future] = job
            for future in as_completed(futures):
                job = futures[future]
                result, extra_paths = future.result()
                validate_result(stage, job, result, protocol)
                path = result_path(output, stage, job)
                write_json(path, result)
                seal(output, manifest, [path, *map(Path, extra_paths)])
                done += 1
                print(f"completed {stage} {done}/{len(expected)} {path.name}", flush=True)
    if not all(completed(result_path(output, stage, job), manifest) for job in expected):
        raise ValueError("Stage cannot finish with missing results")
    manifest["stage_elapsed_seconds"][stage] = manifest["stage_elapsed_seconds"].get(stage, 0.) + time.perf_counter() - start
    seal(output, manifest, [], stage)


def write_audit(base, protocol, output, manifest):
    audit = {"stages": {}, "all_required_stages_complete": set(manifest["stages"]) == set(STAGES)}
    for stage in manifest["stages"]:
        counts = dict(records=0, physical_rollouts=0, known_physical_rollouts=0, unavailable_models=0)
        for job in stage_jobs(stage, base, protocol):
            record, _ = checked_record(result_path(output, stage, job), manifest)
            validate_result(stage, job, record, protocol)
            counts["records"] += 1
            if record["physical_rollouts"] is None:
                counts["physical_rollouts"] = None
            else:
                counts["known_physical_rollouts"] += record["physical_rollouts"]
                if counts["physical_rollouts"] is not None:
                    counts["physical_rollouts"] += record["physical_rollouts"]
            counts["unavailable_models"] += record.get("status") in ("upstream_unavailable", "training_failed")
            if stage in ("fresh_discovery", "prepare", "passive", "confirm") and record.get("status") != "upstream_unavailable":
                for key, value in _old.rollout_counts(record).items():
                    if key != "physical_rollouts":
                        counts[key] = counts.get(key, 0) + value
        audit["stages"][stage] = counts
    audit["total_physical_rollouts"] = (sum(row["physical_rollouts"] for row in audit["stages"].values())
        if all(row["physical_rollouts"] is not None for row in audit["stages"].values()) else None)
    audit["known_physical_rollouts"] = sum(row["known_physical_rollouts"] for row in audit["stages"].values())
    audit["population_sizes"] = {population: sum(model[0] == population for model in models_to_run(base, protocol))
                                 for population in ("existing", "fresh")}
    path = Path(output) / "execution_audit.json"
    write_json(path, audit)
    seal(output, manifest, [path])
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/kernel_rescue.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/kernel_rescue")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "runs/kernel_rescue")
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    args = parser.parse_args()
    protocol = read_json(args.config)
    base, inherited = verify_parent(protocol)
    validate_protocol(base, protocol, inherited)
    output, artifacts = args.output.resolve(), args.artifacts.resolve()
    manifest = initialize(output, artifacts, args.config, base, protocol, inherited)
    for stage in STAGES:
        if args.stage not in ("all", stage):
            continue
        if scientific_sources(args.config) != manifest["scientific_sources_sha256"]:
            raise ValueError("Scientific inputs changed during execution")
        print(f"stage_start {stage}", flush=True)
        run_stage(stage, base, protocol, inherited, output, artifacts, manifest)
        if scientific_sources(args.config) != manifest["scientific_sources_sha256"]:
            raise ValueError("Scientific inputs changed during execution")
        print(f"stage_complete {stage}", flush=True)
    audit = write_audit(base, protocol, output, manifest)
    print(json.dumps({"complete_stages": manifest["stages"], "output": str(output), "audit": audit}), flush=True)


if __name__ == "__main__":
    main()
