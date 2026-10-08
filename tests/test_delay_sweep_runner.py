"""Causal sweep coverage, immutable lineage and root-only result publication."""

from concurrent.futures import Future
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("delay_runner", ROOT / "experiments/run_delay_sweep.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def settings():
    return tuple(json.loads((ROOT / path).read_text()) for path in (
        "configs/timescale_maps.json", "configs/delay_sweep.json",
        "configs/collective_suppression.json"))


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    paths = {name: tmp_path / name for name in (
        "src/model.py", "experiments/run_delay_sweep.py",
        "experiments/run_timescale_maps.py", "experiments/run_collective_suppression.py",
        "docs/DELAY_SWEEP.md")}
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen scientific input\n")
    base, protocol, parent = settings()
    config = tmp_path / "config.json"
    runner.write_json(config, protocol)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "__file__", str(paths["experiments/run_delay_sweep.py"]))
    monkeypatch.setattr(runner, "PROTOCOL", paths["docs/DELAY_SWEEP.md"])
    monkeypatch.setattr(runner, "INHERITED_RUNNERS", [paths["experiments/run_timescale_maps.py"],
                                                      paths["experiments/run_collective_suppression.py"]])
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: "abc123\n")
    output, artifacts = tmp_path / "results", tmp_path / "artifacts"
    inherited = {"parent_manifest_sha256": "immutable-parent", "parent_protocol": parent}
    manifest = runner.initialize(output, artifacts, config, base, protocol, inherited)
    return base, protocol, config, output, artifacts, inherited, manifest


@pytest.mark.parametrize("changed", ["src/model.py", "config.json", "docs/DELAY_SWEEP.md",
                                      "experiments/run_delay_sweep.py",
                                      "experiments/run_timescale_maps.py",
                                      "experiments/run_collective_suppression.py"])
def test_resume_rejects_changed_scientific_inputs(workspace, changed):
    base, protocol, config, output, artifacts, inherited, _ = workspace
    path = runner.ROOT / changed
    path.write_text(path.read_text() + "# changed\n")
    with pytest.raises(ValueError, match="Frozen scientific inputs changed"):
        runner.initialize(output, artifacts, config, base, protocol, inherited)


def test_resume_preserves_config_seals_and_parent_binding(workspace):
    base, protocol, config, output, artifacts, inherited, manifest = workspace
    assert runner.initialize(output, artifacts, config, base, protocol, inherited) == manifest
    for name in ("config.json", "base_config.json"):
        assert runner.completed(output / name, manifest)
    with pytest.raises(ValueError, match="Frozen scientific inputs changed"):
        runner.initialize(output, artifacts, config, base, protocol,
                          {**inherited, "parent_manifest_sha256": "another-parent"})
    (output / "base_config.json").write_text("{}\n")
    with pytest.raises(ValueError, match="Completed artifact changed"):
        runner.initialize(output, artifacts, config, base, protocol, inherited)


def test_dependency_must_be_present_sealed_and_unchanged(workspace):
    _, _, _, output, _, _, manifest = workspace
    path = output / "confirmation/model.json"
    with pytest.raises(ValueError, match="absent, unsealed or changed"):
        runner.checked_record(path, manifest)
    runner.write_json(path, {"group": ["L0H0"]})
    with pytest.raises(ValueError, match="absent, unsealed or changed"):
        runner.checked_record(path, manifest)
    with pytest.raises(ValueError, match="unsealed or changed"):
        runner.completed(path, manifest)
    runner.seal(output, manifest, [path])
    record, digest = runner.checked_record(path, manifest)
    assert record == {"group": ["L0H0"]} and digest == runner.sha(path)
    runner.write_json(path, {"group": ["L1H0"]})
    with pytest.raises(ValueError, match="absent, unsealed or changed"):
        runner.checked_record(path, manifest)


@pytest.fixture
def parent_lineage(tmp_path, monkeypatch):
    base, protocol, parent_protocol = settings()
    protocol["model_subset"] = [{"tau": .1, "noise_tau": .2, "seed": 11}]
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    source = tmp_path / "src/original.py"
    source.parent.mkdir()
    source.write_text("original = True\n")
    checkpoint = tmp_path / "checkpoints/selected.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"frozen selected weights")
    ancestor_path = tmp_path / "results/timescale_maps/run_manifest.json"
    ancestor = {"config": base, "scientific_sources_sha256": {"src/original.py": runner.sha(source)},
                "completed_sha256": {"checkpoints/selected.pt": runner.sha(checkpoint)}}
    runner.write_json(ancestor_path, ancestor)
    directory = tmp_path / protocol["parent_results"]
    metadata = {"path": "checkpoints/selected.pt", "sha256": runner.sha(checkpoint), "epoch": 37}
    discovery = {"tau": .1, "noise_tau": .2, "seed": 11,
                 "checkpoints": {"selected": {"selected": ["L0H0"]}},
                 "checkpoint_metadata": {"selected": metadata}}
    discovery_path = directory / "discovery/tau_0.1_noise_0.2_seed_11.json"
    prepared_path = directory / "dynamics_prepared/tau_0.1_noise_0.2_seed_11.json"
    runner.write_json(discovery_path, discovery)
    prepared = {"tau": .1, "noise_tau": .2, "seed": 11, "group": ["L0H0"],
                "dependencies": {"discovery_sha256": runner.sha(discovery_path)},
                "checkpoint_metadata": {"selected": metadata}}
    runner.write_json(prepared_path, prepared)
    runner.write_json(directory / "base_config.json", base)
    runner.write_json(directory / "config.json", parent_protocol)
    parent = {"base_config": base, "protocol_config": parent_protocol,
              "stages": sorted(runner.PARENT_STAGES),
              "inherited": {"parent_manifest": str(ancestor_path.relative_to(tmp_path)),
                            "parent_manifest_sha256": runner.sha(ancestor_path)},
              "scientific_sources_sha256": {"src/original.py": runner.sha(source)},
              "completed_sha256": {str(path.relative_to(tmp_path)): runner.sha(path)
                                   for path in (discovery_path, prepared_path,
                                                directory / "base_config.json", directory / "config.json")}}
    committed = [None]

    def commit_parent():
        runner.write_json(directory / "run_manifest.json", parent)
        committed[0] = (directory / "run_manifest.json").read_bytes()

    commit_parent()
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: committed[0])
    return {"base": base, "protocol": protocol, "parent": parent, "directory": directory,
            "prepared": prepared, "prepared_path": prepared_path, "source": source,
            "checkpoint": checkpoint, "ancestor_path": ancestor_path, "commit": commit_parent}


def test_parent_verification_binds_selected_checkpoint_and_accepts_new_sources(parent_lineage):
    item = parent_lineage
    (runner.ROOT / "src/new_delay.py").write_text("new = True\n")
    base, inherited = runner.verify_parent(item["protocol"])
    assert base == item["base"]
    metadata = inherited["model_records"]["tau_0.1_noise_0.2_seed_11"]
    assert metadata["prepared_sha256"] == runner.sha(item["prepared_path"])
    assert metadata["checkpoint"]["sha256"] == runner.sha(item["checkpoint"])
    assert inherited["ancestor_artifact_hashes_checked"] == 1


@pytest.mark.parametrize("which", ["source", "checkpoint", "prepared_path", "ancestor_path"])
def test_parent_verification_rejects_lineage_mutation(parent_lineage, which):
    item = parent_lineage
    path = item[which]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed"):
        runner.verify_parent(item["protocol"])


def test_parent_manifest_itself_must_equal_committed_bytes(parent_lineage):
    item = parent_lineage
    path = item["directory"] / "run_manifest.json"
    runner.write_json(path, {**item["parent"], "edited": True})
    with pytest.raises(ValueError, match="differs from its committed record"):
        runner.verify_parent(item["protocol"])


def test_complete_stage_labels_do_not_substitute_for_required_model_records(parent_lineage):
    item = parent_lineage
    relative = str(item["prepared_path"].relative_to(runner.ROOT))
    del item["parent"]["completed_sha256"][relative]
    item["commit"]()
    with pytest.raises(ValueError, match="absent, unsealed or changed"):
        runner.verify_parent(item["protocol"])


@pytest.mark.parametrize("change,match", [
    ("stages", "experiment is incomplete"),
    ("group", "group or discovery dependency"),
    ("discovery_hash", "group or discovery dependency"),
    ("checkpoint_binding", "checkpoint binding"),
])
def test_parent_records_require_semantically_consistent_bindings(parent_lineage, change, match):
    item = parent_lineage
    if change == "stages":
        item["parent"]["stages"].remove("dynamics_prepared")
    else:
        prepared = deepcopy(item["prepared"])
        if change == "group":
            prepared["group"] = ["L1H0"]
        elif change == "discovery_hash":
            prepared["dependencies"]["discovery_sha256"] = "another-discovery"
        else:
            prepared["checkpoint_metadata"]["selected"]["epoch"] = 2
        runner.write_json(item["prepared_path"], prepared)
        item["parent"]["completed_sha256"][str(item["prepared_path"].relative_to(runner.ROOT))] = runner.sha(item["prepared_path"])
    item["commit"]()
    with pytest.raises(ValueError, match=match):
        runner.verify_parent(item["protocol"])


def test_grid_contains_every_model_delay_once_and_subset_preserves_pairing():
    base, protocol, _ = settings()
    expected = runner.jobs(base, protocol)
    assert len(expected) == len(set(expected)) == 48 * 7
    assert len({item[:2] for item in expected}) == 16
    protocol["model_subset"] = [{"tau": .1, "noise_tau": .2, "seed": 11}]
    assert runner.jobs(base, protocol) == [(.1, .2, 11, delay) for delay in protocol["delays"]]
    protocol["model_subset"].append(protocol["model_subset"][0])
    with pytest.raises(ValueError, match="unique inherited"):
        runner.jobs(base, protocol)


def test_fresh_streams_are_disjoint_from_every_inherited_seed_family():
    base, protocol, parent = settings()
    runner.validate_streams(base, protocol, parent)
    inherited = [base["evaluation"]["seeds"][0], base["training"]["train_seed"] + 1,
                 base["training"]["validation_seed"] + 1]
    inherited += [base["diagnostics"][name][0] for name in
                  ("discovery_seeds", "calibration_seeds", "confirmation_seeds")]
    inherited += [parent[name][0] for name in ("discovery_seeds", "confirmation_seeds")]
    inherited += [parent["dynamics"][name][0] for name in ("calibration_seeds", "confirmation_seeds")]
    for seed in inherited:
        changed = deepcopy(protocol)
        changed["confirmation_seeds"][0] = seed
        with pytest.raises(ValueError, match="overlap inherited streams"):
            runner.validate_streams(base, changed, parent)


@pytest.mark.parametrize("seeds", [[], [1, 1], [True], [-1], [1.5]])
def test_confirmation_seeds_require_unique_nonnegative_integers(seeds):
    base, protocol, parent = settings()
    protocol["confirmation_seeds"] = seeds
    with pytest.raises(ValueError, match="unique nonnegative integer"):
        runner.validate_streams(base, protocol, parent)


@pytest.mark.parametrize("key,value", [
    ("delays", []), ("delays", [0., .05, .05, .1]), ("delays", [.1, .05, 0.]),
    ("delays", [-.01, 0., .05, .1]), ("delays", [0., .05, float("nan")]),
    ("delays", [0., .05, float("inf")]), ("delays", [False, .05, .1]),
    ("delays", [0., .1]), ("delay_cue", .1), ("baseline_delay", .1),
    ("primary_high_delay", .3), ("primary_low_delay", .15),
    ("workers", 0), ("workers", True), ("cpu_threads", 1.5),
])
def test_invalid_delay_grids_and_execution_settings_are_rejected(key, value):
    base, protocol, parent = settings()
    protocol[key] = value
    with pytest.raises(ValueError):
        runner.validate_protocol(base, protocol, parent)


def test_distinct_delay_jobs_cannot_share_a_result_path():
    base, protocol, parent = settings()
    protocol["delays"] = [0., .05, .1, .1000001, .1000002]
    try:
        runner.validate_protocol(base, protocol, parent)
    except ValueError:
        return  # Explicit rejection is also a safe handling of this grid.
    paths = [runner.result_path(ROOT / "results/test", "confirm", job)
             for job in runner.jobs(base, protocol)]
    assert len(paths) == len(set(paths))


def test_probe_windows_amplitudes_and_metric_floors_are_finite_and_defined():
    base, original, parent = settings()
    runner.validate_protocol(base, original, parent)
    changes = [("pulse_onset", -1.), ("pulse_width", 0.), ("duration", float("inf")),
               ("pulse_onset", float("nan")), ("pulse_width", 4.),
               ("pulse_amplitudes", []), ("pulse_amplitudes", [0.]),
               ("pulse_amplitudes", [.02, .02]), ("pulse_amplitudes", [float("inf")])]
    for key, value in changes:
        changed = deepcopy(original)
        changed["probe"][key] = value
        with pytest.raises(ValueError):
            runner.validate_protocol(base, changed, parent)
    for value in (0., -1., float("nan"), float("inf")):
        changed = deepcopy(original)
        changed["metric_denominator_epsilon"] = value
        with pytest.raises(ValueError):
            runner.validate_protocol(base, changed, parent)


class ImmediateExecutor:
    """Exercise root publication without spawning processes or physical trials."""

    def __init__(self, **kwargs):
        self.settings = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def submit(self, callback, *args):
        future = Future()
        try:
            future.set_result(callback(*args))
        except Exception as error:
            future.set_exception(error)
        return future


def test_root_seals_every_job_then_resumes_without_recomputation(workspace, monkeypatch):
    base, protocol, _, output, _, inherited, manifest = workspace
    protocol = {**protocol, "model_subset": [{"tau": .1, "noise_tau": .2, "seed": 11}]}
    calls = []

    def compute(stage, job, *args):
        calls.append(job)
        assert not runner.result_path(output, stage, job).exists()
        return dict(zip(("tau", "noise_tau", "seed", "delay"), job))

    monkeypatch.setattr(runner, "ProcessPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(runner, "_compute", compute)
    runner.run_stage("confirm", base, protocol, inherited, output, manifest)
    assert len(calls) == 7 and manifest["stages"] == ["confirm"]
    for job in runner.jobs(base, protocol):
        path = runner.result_path(output, "confirm", job)
        assert runner.completed(path, manifest)
        runner.validate_result("confirm", job, runner.read_json(path))
    runner.run_stage("confirm", base, protocol, inherited, output, manifest)
    assert len(calls) == 7


def test_root_refuses_mislabeled_worker_output_before_publication(workspace, monkeypatch):
    base, protocol, _, output, _, inherited, manifest = workspace
    protocol = {**protocol, "model_subset": [{"tau": .1, "noise_tau": .2, "seed": 11}]}
    monkeypatch.setattr(runner, "ProcessPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(runner, "_compute", lambda *args: {"tau": .1, "noise_tau": .2, "seed": 99, "delay": 0.})
    with pytest.raises(ValueError, match="submitted job identity"):
        runner.run_stage("confirm", base, protocol, inherited, output, manifest)
    assert not (output / "confirmation").exists()
    assert "confirm" not in manifest["stages"]
