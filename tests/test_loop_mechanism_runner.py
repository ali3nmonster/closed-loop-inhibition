"""Calibration isolation, inherited lineage and physical-trial publication."""

from concurrent.futures import Future
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("mechanism_runner", ROOT / "experiments/run_loop_mechanism.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def settings():
    return tuple(json.loads((ROOT / path).read_text()) for path in (
        "configs/timescale_maps.json", "configs/loop_mechanism.json",
        "configs/collective_suppression.json", "configs/delay_sweep.json"))


def development(protocol):
    protocol = deepcopy(protocol)
    protocol.update(development=True, calibration_seeds=[5990001, 5990002],
                    confirmation_seeds=[5991001, 5991002],
                    model_subset=[dict(tau=.2, noise_tau=.01, seed=11)])
    return protocol


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    paths = {name: tmp_path / name for name in (
        "src/model.py", "experiments/run_loop_mechanism.py", "experiments/run_delay_sweep.py",
        "experiments/run_timescale_maps.py", "experiments/run_collective_suppression.py",
        "docs/LOOP_MECHANISM.md")}
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen scientific input\n")
    base, protocol, parent, delay = settings()
    protocol = development(protocol)
    config = tmp_path / "config.json"
    runner.write_json(config, protocol)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "__file__", str(paths["experiments/run_loop_mechanism.py"]))
    monkeypatch.setattr(runner, "PROTOCOL", paths["docs/LOOP_MECHANISM.md"])
    monkeypatch.setattr(runner, "INHERITED_RUNNERS", [paths["experiments" + suffix] for suffix in (
        "/run_timescale_maps.py", "/run_collective_suppression.py", "/run_delay_sweep.py")])
    monkeypatch.setattr(runner.subprocess, "check_output", lambda command, **kwargs:
                        "" if command[1] == "status" else "abc123\n")
    output = tmp_path / "results"
    inherited = {"parent_manifest_sha256": "parent", "parent_protocol": parent, "delay_protocol": delay}
    manifest = runner.initialize(output, config, base, protocol, inherited)
    return base, protocol, config, output, inherited, manifest


@pytest.mark.parametrize("changed", ["src/model.py", "config.json", "docs/LOOP_MECHANISM.md",
                                      "experiments/run_loop_mechanism.py", "experiments/run_delay_sweep.py",
                                      "experiments/run_timescale_maps.py", "experiments/run_collective_suppression.py"])
def test_resume_rejects_any_changed_scientific_input(workspace, changed):
    base, protocol, config, output, inherited, _ = workspace
    path = runner.ROOT / changed
    path.write_text(path.read_text() + "# changed\n")
    with pytest.raises(ValueError, match="Frozen scientific inputs changed"):
        runner.initialize(output, config, base, protocol, inherited)


def test_resume_binds_both_inputs_and_outputs(workspace):
    base, protocol, config, output, inherited, manifest = workspace
    assert runner.initialize(output, config, base, protocol, inherited) == manifest
    with pytest.raises(ValueError, match="Frozen scientific inputs changed"):
        runner.initialize(output, config, base, protocol, {**inherited, "delay_manifest_sha256": "changed"})
    (output / "base_config.json").write_text("{}\n")
    with pytest.raises(ValueError, match="Completed artifact changed"):
        runner.initialize(output, config, base, protocol, inherited)


def test_production_refuses_dirty_launch_but_development_records_it(workspace, monkeypatch):
    base, protocol, config, _, inherited, _ = workspace
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *args, **kwargs: "dirty\n")
    production = {**protocol, "development": False}
    with pytest.raises(ValueError, match="committed, clean"):
        runner.initialize(runner.ROOT / "production", config, base, production, inherited)
    manifest = runner.initialize(runner.ROOT / "development", config, base, protocol, inherited)
    assert manifest["source_dirty"]


def test_new_run_does_not_adopt_unsealed_existing_files(workspace):
    base, protocol, config, _, inherited, _ = workspace
    output = runner.ROOT / "unsealed"
    runner.write_json(output / "some-result.json", {})
    with pytest.raises(ValueError, match="empty result"):
        runner.initialize(output, config, base, protocol, inherited)


def test_confirmation_requires_every_preparation_and_global_seal(workspace):
    base, protocol, _, output, _, manifest = workspace
    expected = runner.jobs(base, protocol)
    for job in expected:
        runner.write_json(runner.result_path(output, "prepare", job), {})
    runner.seal(output, manifest, [runner.result_path(output, "prepare", job) for job in expected])
    with pytest.raises(ValueError, match="All calibration"):
        runner.require_all_prepared(base, protocol, output, manifest)
    runner.seal(output, manifest, [], "prepare")
    runner.require_all_prepared(base, protocol, output, manifest)
    runner.result_path(output, "prepare", expected[-1]).unlink()
    with pytest.raises(ValueError, match="All calibration"):
        runner.require_all_prepared(base, protocol, output, manifest)


@pytest.mark.parametrize("stage", ["passive", "confirm", "mechanism"])
def test_no_postcalibration_stage_can_start_early(workspace, monkeypatch, stage):
    base, protocol, _, output, inherited, manifest = workspace
    monkeypatch.setattr(runner, "ProcessPoolExecutor", lambda **kwargs: pytest.fail("workers must not start"))
    with pytest.raises(ValueError, match="All calibration"):
        runner.run_stage(stage, base, protocol, inherited, output, manifest)


def test_grid_and_development_subset():
    base, protocol, parent, delay = settings()
    runner.validate_protocol(base, protocol, parent, delay)
    assert len(runner.jobs(base, protocol)) == 60
    assert len(runner.stage_jobs("passive", base, protocol)) == 4
    protocol = development(protocol)
    protocol["delays"] = [0., .1]
    runner.validate_protocol(base, protocol, parent, delay)
    assert runner.jobs(base, protocol) == [(.2, .01, 11, 0.), (.2, .01, 11, .1)]
    protocol["development"] = False
    with pytest.raises(ValueError):
        runner.validate_protocol(base, protocol, parent, delay)


@pytest.mark.parametrize("stream", ["calibration_seeds", "confirmation_seeds"])
def test_both_new_stream_families_reject_all_historical_families(stream):
    base, protocol, parent, delay = settings()
    old = [base["training"]["train_seed"] + 1, base["training"]["validation_seed"] + 1,
           base["evaluation"]["seeds"][0], delay["confirmation_seeds"][0]]
    old += [base["diagnostics"][key][0] for key in ("calibration_seeds", "discovery_seeds", "confirmation_seeds")]
    old += [parent[key][0] for key in ("discovery_seeds", "confirmation_seeds")]
    old += [parent["dynamics"][key][0] for key in ("calibration_seeds", "confirmation_seeds")]
    for seed in old:
        changed = deepcopy(protocol)
        changed[stream][0] = seed
        with pytest.raises(ValueError, match="overlap inherited"):
            runner.validate_streams(base, changed, parent, delay)


def test_calibration_confirmation_and_development_streams_are_separate():
    base, protocol, parent, delay = settings()
    protocol["calibration_seeds"][0] = protocol["confirmation_seeds"][0]
    with pytest.raises(ValueError, match="disjoint"):
        runner.validate_streams(base, protocol, parent, delay)
    protocol = settings()[1]
    protocol["development"] = True
    with pytest.raises(ValueError, match="reserved production"):
        runner.validate_streams(base, protocol, parent, delay)


@pytest.mark.parametrize("key,value", [
    ("delays", []), ("delays", [0., .05, .05, .1]), ("delays", [0., .1]),
    ("delays", [0., .05, float("nan")]), ("delay_cue", .1), ("baseline_delay", .1),
    ("primary_low_delay", .1), ("primary_high_delay", .2), ("workers", 0), ("cpu_threads", True),
    ("gain_bounds", [-1., 4.]), ("gain_bounds", [2., 4.]), ("response_energy_floor", 0.),
    ("min_positive_fraction", 1.1), ("plant_taus", [.1]), ("development", "yes"),
])
def test_invalid_protocol_inputs_rejected(key, value):
    base, protocol, parent, delay = settings()
    protocol[key] = value
    with pytest.raises(ValueError):
        runner.validate_protocol(base, protocol, parent, delay)


@pytest.mark.parametrize("key,value", [
    ("duration", float("inf")), ("duration", 12.01), ("pulse_width", 13.),
    ("parity_duration", .1), ("frequencies", [1., 10.]), ("frequencies", [1., .5]),
    ("pulse_amplitudes", [.02]), ("pulse_amplitudes", [0., .0001]),
    ("equilibrium_tolerance", 0.), ("settling_fraction", 1.), ("parity_command_tolerance", -1.),
])
def test_invalid_mechanism_inputs_rejected(key, value):
    base, protocol, parent, delay = settings()
    protocol["mechanism"][key] = value
    with pytest.raises(ValueError):
        runner.validate_protocol(base, protocol, parent, delay)


@pytest.fixture
def delay_lineage(tmp_path, monkeypatch):
    base, protocol, parent, delay_protocol = settings()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    runner.write_json(tmp_path / protocol["parent_results"] / "base_config.json", base)
    source = tmp_path / "old_source.py"
    source.write_text("frozen=True\n")
    inherited = dict(parent_commit="d9f7e9f", parent_manifest="parent.json", parent_manifest_sha256="psha",
                     ancestor_manifest_sha256="asha", model_records={"model": {"checkpoint": "frozen"}})
    monkeypatch.setattr(runner._delay_runner, "verify_parent", lambda value: (base, deepcopy(inherited)))
    directory = tmp_path / protocol["delay_results"]
    runner.write_json(directory / "base_config.json", base)
    runner.write_json(directory / "config.json", delay_protocol)
    manifest = dict(stages=["passive", "confirm"], base_config=base, protocol_config=delay_protocol,
                    inherited=deepcopy(inherited), scientific_sources_sha256={"old_source.py": runner.sha(source)},
                    completed_sha256={str(path.relative_to(tmp_path)): runner.sha(path) for path in
                                      (directory / "base_config.json", directory / "config.json")})
    committed = [None]

    def commit():
        runner.write_json(directory / "run_manifest.json", manifest)
        committed[0] = (directory / "run_manifest.json").read_bytes()

    commit()
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *args, **kwargs: committed[0])
    return base, protocol, directory, manifest, source, commit


def test_delay_parent_checks_committed_bytes_static_hash_inventory_and_bindings(delay_lineage):
    base, protocol, directory, _, source, _ = delay_lineage
    (runner.ROOT / "new_source.py").write_text("new=True\n")
    actual, inherited = runner.verify_parent(protocol)
    assert actual == base
    assert inherited["delay_source_hashes_checked"] == 1
    assert inherited["delay_artifact_hashes_checked"] == 2
    source.write_text("changed=True\n")
    with pytest.raises(ValueError, match="Inherited artifact or source changed"):
        runner.verify_parent(protocol)


@pytest.mark.parametrize("change,match", [
    ("uncommitted", "committed record"), ("stage", "experiment is incomplete"),
    ("lineage", "lineage differ"), ("model", "controller binding differs"),
    ("artifact", "artifact or source changed"),
])
def test_delay_parent_rejects_incomplete_mutated_or_mismatched_records(delay_lineage, change, match):
    _, protocol, directory, manifest, _, commit = delay_lineage
    if change == "uncommitted":
        runner.write_json(directory / "run_manifest.json", {**manifest, "extra": True})
    elif change == "artifact":
        (directory / "config.json").write_text("{}\n")
    else:
        if change == "stage":
            manifest["stages"] = ["passive"]
        elif change == "lineage":
            manifest["inherited"]["parent_manifest_sha256"] = "another"
        else:
            manifest["inherited"]["model_records"]["model"] = {}
        commit()
    with pytest.raises(ValueError, match=match):
        runner.verify_parent(protocol)


def physical_record(protocol, stage="prepare", missing=()):
    labels = ["native"] if stage == "prepare" else [name for name in (
        "native", "joint_weak", "joint_strong", "gain_matched", "weak_rescue") if name not in missing]
    streams = protocol["calibration_seeds" if stage == "prepare" else "confirmation_seeds"]
    good = dict(censored=False, task_failure=False, position_rms=.01)
    rows, timing = [], []
    for label in labels:
        for seed in streams:
            timing.append(dict(variant=label, noise_seed=seed, amplitude=0.))
            for amplitude in protocol["probe"]["pulse_amplitudes"]:
                identity = dict(variant=label, noise_seed=seed, amplitude=amplitude)
                rows.append({**identity, "sham": deepcopy(good), "pulse": deepcopy(good)})
                timing.append(identity)
    data = dict(rollouts=rows, timing=timing, physical_rollouts=len(timing))
    variants = {name: {} for name in ("native", "joint_weak", "joint_strong", "gain_matched", "weak_rescue") if name not in missing}
    record = dict(tau=.2, noise_tau=.01, seed=11, delay=0., physical_rollouts=len(timing),
                  unavailable_controls={name: "degenerate" for name in missing})
    if stage == "prepare":
        record.update(calibration=data, variants=variants)
    else:
        record.update(**data, frozen_settings=variants,
                      summary={name: None if name in missing else {} for name in
                               ("native", "joint_weak", "joint_strong", "gain_matched", "weak_rescue")})
    return record


def test_failure_counts_do_not_double_count_reused_shams_or_discard_failure_metrics():
    protocol = settings()[1]
    record = physical_record(protocol)
    rows = record["calibration"]["rollouts"]
    rows[0]["sham"].update(task_failure=True, position_rms=.2)
    rows[1]["sham"] = deepcopy(rows[0]["sham"])
    rows[0]["pulse"].update(censored=True, task_failure=True, position_rms=None)
    assert runner.rollout_counts(record) == dict(physical_rollouts=6, task_failures=2, censored=1, complete=5)
    runner.validate_result("prepare", (.2, .01, 11, 0.), record, protocol)


@pytest.mark.parametrize("change", ["duplicate", "conflicting_sham", "false_count", "null_complete", "numeric_censored"])
def test_trial_count_audit_rejects_corrupt_records(change):
    record = physical_record(settings()[1])
    rows = record["calibration"]["rollouts"]
    if change == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif change == "conflicting_sham":
        rows[0]["sham"]["position_rms"] = .2
    elif change == "false_count":
        record["physical_rollouts"] = 7
    elif change == "null_complete":
        rows[0]["pulse"]["position_rms"] = None
    else:
        rows[0]["pulse"].update(censored=True, task_failure=True)
    with pytest.raises(ValueError):
        runner.rollout_counts(record)


def test_unavailable_controls_remain_explicit_without_fabricated_rollouts():
    protocol = settings()[1]
    record = physical_record(protocol, "confirm", missing=("weak_rescue",))
    assert record["physical_rollouts"] == 48
    runner.validate_result("confirm", (.2, .01, 11, 0.), record, protocol)
    record["summary"]["weak_rescue"] = {}
    with pytest.raises(ValueError, match="null summaries"):
        runner.validate_result("confirm", (.2, .01, 11, 0.), record, protocol)


def test_result_identity_and_physical_coverage_checked_before_publication():
    protocol = settings()[1]
    record = physical_record(protocol)
    with pytest.raises(ValueError, match="job identity"):
        runner.validate_result("prepare", (.2, .01, 22, 0.), record, protocol)
    record["calibration"]["timing"].pop()
    with pytest.raises(ValueError, match="Timing audits"):
        runner.validate_result("prepare", (.2, .01, 11, 0.), record, protocol)


def test_nonfinite_serialization_leaves_no_partial_file(tmp_path):
    path = tmp_path / "result.json"
    with pytest.raises(ValueError):
        runner.write_json(path, {"measurement": float("nan")})
    assert not path.exists() and not path.with_suffix(".json.tmp").exists()


def test_worker_results_are_validated_before_root_writes_or_seals(workspace, monkeypatch):
    base, protocol, _, output, inherited, manifest = workspace
    protocol["delays"] = [0.]

    class FakeExecutor:
        def __init__(self, **kwargs):
            assert kwargs["mp_context"].get_start_method() == "spawn"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, stage, job, *args):
            future = Future()
            record = physical_record(protocol)
            record["seed"] = 22
            future.set_result(record)
            return future

    monkeypatch.setattr(runner, "ProcessPoolExecutor", FakeExecutor)
    with pytest.raises(ValueError, match="job identity"):
        runner.run_stage("prepare", base, protocol, inherited, output, manifest)
    assert "prepare" not in manifest["stages"]
    assert not runner.result_path(output, "prepare", (.2, .01, 11, 0.)).exists()
