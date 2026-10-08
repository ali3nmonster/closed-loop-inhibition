"""Prospective populations, complete stage coverage and immutable provenance."""

from copy import deepcopy
import importlib.util
import itertools
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("kernel_runner", ROOT / "experiments/run_kernel_rescue.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def settings():
    def read(name):
        return json.loads((ROOT / "configs" / f"{name}.json").read_text())
    return read("timescale_maps"), read("kernel_rescue"), {
        "parent_protocol": read("collective_suppression"),
        "delay_protocol": read("delay_sweep"), "loop_protocol": read("loop_mechanism")}


def development(protocol):
    result = deepcopy(protocol)
    result.update(development=True, model_subset=[dict(population="existing", tau=.2, noise_tau=.01, seed=11),
                                                   dict(population="fresh", tau=.2, noise_tau=.01, seed=101)],
                  discovery_seeds=[7010001, 7010002], offset_calibration_seeds=[7020001, 7020002],
                  calibration_seeds=[7030001, 7030002], confirmation_seeds=[7050001, 7050002],
                  training_streams=dict(train_base=7110000, validation_base=7210000, stride=100),
                  development_training=dict(epochs=2, intermediate_epoch=1))
    return result


def test_full_grid_keeps_existing_and_fresh_populations_separate():
    base, protocol, inherited = settings()
    runner.validate_protocol(base, protocol, inherited)
    models = runner.models_to_run(base, protocol)
    assert len(models) == 52
    assert sum(row[0] == "fresh" for row in models) == 40
    assert sum(row[0] == "existing" for row in models) == 12
    expected = dict(discovery_bank=1, fresh_train=40, fresh_discovery=40, prepare=52, passive=8,
                    confirm=208, mechanism=104)
    assert {stage: len(runner.stage_jobs(stage, base, protocol)) for stage in runner.STAGES} == expected


def test_development_subset_does_not_consume_production_streams():
    base, protocol, inherited = settings()
    protocol = development(protocol)
    runner.validate_protocol(base, protocol, inherited)
    assert len(runner.stage_jobs("fresh_train", base, protocol)) == 1
    assert len(runner.stage_jobs("confirm", base, protocol)) == 8
    protocol["development"] = False
    with pytest.raises(ValueError):
        runner.validate_protocol(base, protocol, inherited)


def test_stream_validation_does_not_mutate_any_inherited_record():
    base, protocol, inherited = settings()
    before = deepcopy(inherited)
    runner.validate_streams(base, protocol, inherited)
    runner.validate_streams(base, protocol, inherited)
    assert inherited == before


@pytest.mark.parametrize("name", ["discovery_seeds", "offset_calibration_seeds", "calibration_seeds", "confirmation_seeds"])
def test_each_new_stream_family_rejects_historical_or_new_overlap(name):
    base, protocol, inherited = settings()
    old = [base["training"]["train_seed"] + 1, base["training"]["validation_seed"] + 1,
           inherited["parent_protocol"]["discovery_seeds"][0],
           inherited["delay_protocol"]["confirmation_seeds"][0],
           inherited["loop_protocol"]["calibration_seeds"][0],
           inherited["loop_protocol"]["confirmation_seeds"][0], 6110000, 6210000]
    for seed in old:
        changed = deepcopy(protocol)
        changed[name][0] = seed
        with pytest.raises(ValueError):
            runner.validate_streams(base, changed, inherited)


def test_new_streams_all_disjoint_and_demo_stride_validated():
    base, protocol, inherited = settings()
    protocol["calibration_seeds"][0] = protocol["offset_calibration_seeds"][0]
    with pytest.raises(ValueError, match="disjoint"):
        runner.validate_streams(base, protocol, inherited)
    protocol = settings()[1]
    protocol["training_streams"]["stride"] = 15
    with pytest.raises(ValueError, match="stride"):
        runner.validate_streams(base, protocol, inherited)


@pytest.mark.parametrize("key,value", [
    ("plant_taus", [.1]), ("delays", [0., .1]), ("calibration_delay", .1),
    ("baseline_delay", .1), ("delay_cue", .1), ("primary_low_delay", 0.),
    ("confirmation_durations", [4.]), ("fresh_seeds", [11]), ("fresh_seeds", [101, 101]),
    ("workers", 0), ("cpu_threads", True), ("development", "yes"),
    ("baseline_epsilon", 0.), ("weak_scale", .8), ("min_fractional_increase", .02),
    ("min_positive_fraction", .5), ("gain_bounds", [-1., 4.]), ("gain_bounds", [2., 4.]),
    ("development_training", {"epochs": 2}), ("discovery_probe", {}),
])
def test_invalid_protocol_cannot_launch(key, value):
    base, protocol, inherited = settings()
    protocol[key] = value
    with pytest.raises(ValueError):
        runner.validate_protocol(base, protocol, inherited)


def test_development_rejects_production_streams_and_invalid_training_override():
    base, protocol, inherited = settings()
    protocol["development"] = True
    with pytest.raises(ValueError, match="reserved production"):
        runner.validate_protocol(base, protocol, inherited)
    protocol = development(settings()[1])
    protocol["development_training"]["intermediate_epoch"] = 3
    with pytest.raises(ValueError, match="Intermediate"):
        runner.validate_protocol(base, protocol, inherited)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    paths = ["src/model.py", "experiments/run_kernel_rescue.py", "docs/KERNEL_RESCUE.md"] + [
        "experiments/" + name for name in ("run_timescale_maps.py", "run_collective_suppression.py",
                                         "run_delay_sweep.py", "run_loop_mechanism.py")]
    for relative in paths:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# frozen scientific source\n")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "__file__", str(tmp_path / paths[1]))
    monkeypatch.setattr(runner, "PROTOCOL", tmp_path / paths[2])
    monkeypatch.setattr(runner, "INHERITED_RUNNERS", [tmp_path / relative for relative in paths[3:]])
    monkeypatch.setattr(runner.subprocess, "check_output", lambda command, **kwargs:
                        "" if command[1] == "status" else "abc123\n")
    base, protocol, inherited = settings()
    protocol = development(protocol)
    config = tmp_path / "config.json"
    runner.write_json(config, protocol)
    output, artifacts = tmp_path / "results", tmp_path / "artifacts"
    manifest = runner.initialize(output, artifacts, config, base, protocol, inherited)
    return base, protocol, inherited, config, output, artifacts, manifest


@pytest.mark.parametrize("relative", ["src/model.py", "config.json", "docs/KERNEL_RESCUE.md",
                                     "experiments/run_kernel_rescue.py", "experiments/run_loop_mechanism.py"])
def test_resume_rejects_changed_scientific_inputs(workspace, relative):
    base, protocol, inherited, config, output, artifacts, _ = workspace
    path = runner.ROOT / relative
    path.write_text(path.read_text() + "# changed\n")
    with pytest.raises(ValueError, match="Frozen scientific"):
        runner.initialize(output, artifacts, config, base, protocol, inherited)


def test_resume_binds_outputs_artifact_directory_and_lineage(workspace):
    base, protocol, inherited, config, output, artifacts, manifest = workspace
    assert runner.initialize(output, artifacts, config, base, protocol, inherited) == manifest
    with pytest.raises(ValueError, match="Frozen scientific"):
        runner.initialize(output, runner.ROOT / "other", config, base, protocol, inherited)
    with pytest.raises(ValueError, match="Frozen scientific"):
        runner.initialize(output, artifacts, config, base, protocol, {**inherited, "changed": True})
    (output / "config.json").write_text("{}\n")
    with pytest.raises(ValueError, match="Completed artifact changed"):
        runner.initialize(output, artifacts, config, base, protocol, inherited)


def test_production_requires_clean_commit(workspace, monkeypatch):
    base, protocol, inherited, config, _, _, _ = workspace
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *args, **kwargs: "dirty\n")
    with pytest.raises(ValueError, match="committed, clean"):
        runner.initialize(runner.ROOT / "production", runner.ROOT / "prod_artifacts", config,
                          base, {**protocol, "development": False}, inherited)


def test_result_names_encode_population_delay_and_horizon():
    path = runner.result_path(Path("results"), "confirm", ("fresh", .2, .01, 101, .05, 12.))
    assert path.name == "fresh_tau_0.2_noise_0.01_seed_101_delay_0.05_duration_12.json"
    assert runner.result_path(Path("results"), "passive", (.2, .01, 4.)).name.endswith("_duration_4.json")


@pytest.mark.parametrize("stage", ["passive", "confirm", "mechanism"])
def test_no_confirmation_can_begin_before_all_fifty_two_preparations(workspace, monkeypatch, stage):
    base, protocol, inherited, _, output, artifacts, manifest = workspace
    monkeypatch.setattr(runner, "ProcessPoolExecutor", lambda **kwargs: pytest.fail("must not create workers"))
    with pytest.raises(ValueError, match="All prepare"):
        runner.run_stage(stage, base, protocol, inherited, output, artifacts, manifest)


def test_require_stage_needs_global_seal_and_every_expected_result(workspace):
    base, protocol, _, _, output, _, manifest = workspace
    paths = [runner.result_path(output, "prepare", job) for job in runner.stage_jobs("prepare", base, protocol)]
    for path in paths:
        runner.write_json(path, {})
    runner.seal(output, manifest, paths)
    with pytest.raises(ValueError, match="All prepare"):
        runner.require_stage("prepare", base, protocol, output, manifest)
    runner.seal(output, manifest, [], "prepare")
    runner.require_stage("prepare", base, protocol, output, manifest)
    paths[-1].unlink()
    with pytest.raises(ValueError, match="All prepare"):
        runner.require_stage("prepare", base, protocol, output, manifest)


def physical_record(protocol, stage="confirm"):
    labels = runner.VARIANTS if stage == "confirm" else {"native"}
    stream = "confirmation_seeds" if stage == "confirm" else "calibration_seeds"
    good = dict(censored=False, task_failure=False, position_rms=.01)
    rows, timing = [], []
    for variant, seed in itertools.product(labels, protocol[stream]):
        timing.append(dict(variant=variant, noise_seed=seed, amplitude=0.))
        for amplitude in protocol["probe"]["pulse_amplitudes"]:
            identity = dict(variant=variant, noise_seed=seed, amplitude=amplitude)
            rows.append({**identity, "sham": deepcopy(good), "pulse": deepcopy(good)})
            timing.append(identity)
    result = dict(population="fresh", tau=.2, noise_tau=.01, seed=101, delay=.05, duration=4.,
                  physical_rollouts=len(timing), unavailable_controls={},
                  frozen_settings={name: {} for name in runner.VARIANTS},
                  summary={name: {} for name in runner.VARIANTS}, rollouts=rows, timing=timing)
    if stage == "prepare":
        result["variants"] = result.pop("frozen_settings")
        result["calibration"] = {key: result.pop(key) for key in ("rollouts", "timing")}
        result["calibration"]["physical_rollouts"] = result["physical_rollouts"]
    return result


@pytest.mark.parametrize("change,match", [
    ("identity", "job identity"), ("stream", "physical trial count|omit declared"), ("timing", "Timing audits"),
    ("variant", "Every variant"), ("summary", "every variant"),
    ("count", "Declared physical"), ("failure", "numeric metrics"), ("censored", "null metrics"),
])
def test_confirmation_validation_rejects_missing_or_misrepresented_trials(change, match):
    protocol = settings()[1]
    record = physical_record(protocol)
    if change == "identity":
        record["population"] = "existing"
    elif change == "stream":
        record["rollouts"][0]["noise_seed"] = 999
    elif change == "timing":
        record["timing"].pop()
    elif change == "variant":
        record["frozen_settings"].pop("weak_kernel")
    elif change == "summary":
        record["summary"]["weak_kernel"] = None
    elif change == "count":
        record["physical_rollouts"] += 1
    elif change == "failure":
        record["rollouts"][0]["pulse"].update(task_failure=True, position_rms=None)
    else:
        record["rollouts"][0]["pulse"].update(censored=True, task_failure=True)
    with pytest.raises(ValueError, match=match):
        runner.validate_result("confirm", ("fresh", .2, .01, 101, .05, 4.), record, protocol)


def test_complete_failed_trials_stay_numeric_and_are_not_dropped():
    protocol = settings()[1]
    record = physical_record(protocol)
    record["rollouts"][0]["pulse"].update(task_failure=True, position_rms=.5)
    runner.validate_result("confirm", ("fresh", .2, .01, 101, .05, 4.), record, protocol)
    assert runner._old.rollout_counts(record)["task_failures"] == 1


def test_unavailable_models_retain_all_null_summaries():
    protocol = settings()[1]
    job = ("fresh", .2, .01, 101, .1, 12.)
    record = runner._unavailable("confirm", job, "training_failed")
    runner.validate_result("confirm", job, record, protocol)
    record["summary"]["native"] = {}
    with pytest.raises(ValueError, match="null summaries"):
        runner.validate_result("confirm", job, record, protocol)


def test_failed_training_count_is_unknown_not_a_false_zero():
    protocol = settings()[1]
    record = dict(population="fresh", tau=.2, noise_tau=.01, seed=101, status="training_failed", physical_rollouts=None)
    runner.validate_result("fresh_train", ("fresh", .2, .01, 101), record, protocol)
    record["physical_rollouts"] = 0
    with pytest.raises(ValueError, match="unknown"):
        runner.validate_result("fresh_train", ("fresh", .2, .01, 101), record, protocol)


def test_strict_json_serializes_before_touching_target(tmp_path):
    path = tmp_path / "record.json"
    runner.write_json(path, {"safe": 1})
    before = path.read_bytes()
    with pytest.raises(ValueError):
        runner.write_json(path, {"unsafe": float("nan")})
    assert path.read_bytes() == before


@pytest.mark.parametrize("key,value", [("duration", 4.01), ("pulse_width", 5.), ("parity_duration", .2),
    ("verification_amplitude", 0.), ("kernel_jacobian_tolerance", float("nan")),
    ("frequencies", [1., 10.]), ("frequencies", [1., .5]), ("pulse_amplitudes", [.02]),
    ("settling_fraction", 1.)])
def test_invalid_mechanism_cannot_launch(key, value):
    base, protocol, inherited = settings()
    protocol["mechanism"][key] = value
    with pytest.raises(ValueError):
        runner.validate_protocol(base, protocol, inherited)


@pytest.mark.parametrize("key,value", [("bootstrap_seed", True), ("bootstrap_resamples", 0),
                                       ("confidence_level", 1.), ("bootstrap_seed", 7070001)])
def test_production_statistical_specification_cannot_change(key, value):
    base, protocol, inherited = settings()
    protocol["statistics"][key] = value
    with pytest.raises(ValueError):
        runner.validate_protocol(base, protocol, inherited)


@pytest.fixture
def loop_lineage(tmp_path, monkeypatch):
    base, protocol, _ = settings()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    inherited = dict(parent_manifest_sha256="collective", ancestor_manifest_sha256="original",
                     delay_manifest_sha256="delay", model_records={"model": {"checkpoint": "selected"}})
    monkeypatch.setattr(runner._old, "verify_parent", lambda protocol: (base, deepcopy(inherited)))
    directory = tmp_path / protocol["loop_results"]
    runner.write_json(directory / "config.json", {"frozen": True})
    runner.write_json(directory / "base_config.json", base)
    source = tmp_path / "frozen.py"
    source.write_text("# inherited scientific source\n")
    manifest = dict(stages=list(runner._old.STAGES), base_config=base, protocol_config={"frozen": True},
                    inherited=deepcopy(inherited), scientific_sources_sha256={"frozen.py": runner.sha(source)},
                    completed_sha256={str(path.relative_to(tmp_path)): runner.sha(path) for path in
                                      (directory / "config.json", directory / "base_config.json")})
    committed = []

    def commit():
        runner.write_json(directory / "run_manifest.json", manifest)
        committed[:] = [(directory / "run_manifest.json").read_bytes()]

    commit()
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *args, **kwargs: committed[0])
    return base, protocol, manifest, directory, source, commit


def test_parent_verification_checks_declared_inventory_not_new_source_files(loop_lineage):
    base, protocol, _, _, _, _ = loop_lineage
    (runner.ROOT / "new.py").write_text("# new source is not an inherited mutation\n")
    actual, inherited = runner.verify_parent(protocol)
    assert actual == base
    assert inherited["loop_source_hashes_checked"] == 1
    assert inherited["loop_artifact_hashes_checked"] == 2


@pytest.mark.parametrize("change,match", [("source", "artifact or source"), ("artifact", "artifact or source"),
    ("uncommitted", "committed record"), ("stage", "incomplete"), ("lineage", "lineage bindings")])
def test_parent_verification_rejects_mutation_incompleteness_or_rebinding(loop_lineage, change, match):
    _, protocol, manifest, directory, source, commit = loop_lineage
    if change == "source":
        source.write_text("changed")
    elif change == "artifact":
        (directory / "config.json").write_text("{}\n")
    elif change == "uncommitted":
        runner.write_json(directory / "run_manifest.json", {**manifest, "changed": True})
    elif change == "stage":
        manifest["stages"] = ["prepare"]
        commit()
    else:
        manifest["inherited"]["delay_manifest_sha256"] = "another"
        commit()
    with pytest.raises(ValueError, match=match):
        runner.verify_parent(protocol)
