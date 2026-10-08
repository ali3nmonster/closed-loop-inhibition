"""Synthetic integration checks; no held-out scientific scenarios are run."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from closed_loop_inhibition.centered_calibration import centered_commands, weighted_mean
from closed_loop_inhibition.plants import Oscillator


@pytest.fixture(scope="module")
def runner():
    path = Path(__file__).resolve().parents[1] / "experiments/run_centered_suppression.py"
    spec = importlib.util.spec_from_file_location("centered_runner_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_policy_uses_physical_command_units_and_one_final_actuator_clip(runner, monkeypatch):
    class ReadFeature(torch.nn.Module):
        def forward(self, tokens, valid):
            assert not torch.is_grad_enabled()
            return tokens[:, 0, 0]

    def encode(value, **kwargs):
        return np.array([[value]], dtype=np.float32), np.array([True])

    monkeypatch.setattr(runner.prior, "encode_snapshot", encode)
    policy = runner.CenteredPolicy(ReadFeature(), Oscillator(), {"action_limit": 5},
                                   center=1.0, gain=2.0, offset=-17.0)
    # Raw normalized 2 becomes physical 10, then 1+2*(10-1)-17 = 2.
    # Clipping the physical command before transformation would give -8.
    assert policy(2.0) == 2.0
    assert policy(3.0) == 12.0  # The simulator owns final clipping.
    expected = centered_commands([10, 15], action_limit=5, center=1, gain=2, offset=-17)
    np.testing.assert_array_equal(np.clip([policy(2), policy(3)], -5, 5), expected)
    assert policy.offset == -17.0
    assert policy.command_gain == 2.0


def test_fit_uses_sham_only_and_keeps_offset_fixed_on_pulse_histories(runner):
    native = (np.array([0.2, -0.4]), np.array([-0.4, 0.4]))
    changed = (np.array([10.0, -10.0]), np.array([1.6, -0.2]))
    fitted, outputs = runner.fit_variant(native, changed, ["a", "a"], 1, atol=1e-10)
    assert fitted["offset"] == pytest.approx(-0.7, abs=1e-10)
    np.testing.assert_allclose(outputs[1], [0.9, -0.9], atol=1e-10)
    np.testing.assert_array_equal(outputs[0], [1, -1])
    other, _ = runner.fit_variant(native, (np.array([100.0, 300.0]), changed[1]),
                                  ["a", "a"], 1, atol=1e-10)
    assert other["offset"] == fitted["offset"]
    assert other["achieved_rms"] != fitted["achieved_rms"]


def test_fitted_gain_and_runtime_transform_agree_with_saturation(runner):
    native = (np.array([4.0, -3.0, 0.5]), np.array([-0.8, 0.4, 0.9]))
    groups = ["a", "a", "a"]
    center = weighted_mean(native[1], groups)
    fitted, outputs = runner.fit_variant(native, native, groups, 1, atol=1e-10,
                                         center=center, gain=2)
    assert abs(fitted["offset"]) > 0.01
    assert weighted_mean(outputs[1], groups) == pytest.approx(center, abs=1e-10)
    for raw, output in zip(native, outputs):
        expected = np.clip(center + 2 * (raw - center) + fitted["offset"], -1, 1)
        np.testing.assert_array_equal(output, expected)


def test_match_ties_use_ascending_parameter_even_when_identity_is_nearer(runner):
    grid = [{"parameter": 0.9, "achieved_rms": 0.75},
            {"parameter": 0.5, "achieved_rms": 1.25}]
    result = runner.choose_match(grid, 1.0, {"matching_relative_tolerance": 0.05})
    assert result["parameter"] == 0.5
    assert result["relative_error"] == 0.25
    assert result["available"] and not result["matched"]
    assert runner.choose_match(list(reversed(grid)), 1, {"matching_relative_tolerance": .05}) == result


@pytest.mark.parametrize("target", [0.0, 0.5e-8])
def test_near_zero_target_retains_identity_without_a_ratio(runner, target):
    grid = [{"parameter": 0.5, "achieved_rms": target, "offset": 0.1},
            {"parameter": 1.0, "achieved_rms": 0.0, "offset": 0.0}]
    result = runner.choose_match(grid, target, {"matching_relative_tolerance": .05, "target_epsilon": 1e-8})
    assert result["parameter"] == 1
    assert result["offset"] == 0
    assert result["relative_error"] is None
    assert result["available"] is False
    assert result["matched"] is False
    json.dumps(result, allow_nan=False)


def test_control_matching_is_local_to_each_calibration_cell(runner):
    config = {"candidate_scales": {"weak": .9, "strong": 1.1}, "calibration": {
        "mean_atol": 1e-10, "gain_min": 1.0, "gain_max": 1.2, "gain_step": .1,
        "matching_relative_tolerance": .05, "target_epsilon": 1e-8,
    }}
    pair = lambda pulse, sham: (np.full(2, pulse), np.full(2, sham))
    native = pair(1.0, 0.0)
    alternative = {.5: pair(1.4, .6), .75: pair(1.1, .2), 1.0: native}
    first, first_grids, first_outputs = runner.calibrate_cell(
        native, {.9: pair(1.1, .3), 1.1: pair(.9, -.3)}, alternative, ["a", "b"], 5, config)
    second, _, _ = runner.calibrate_cell(
        native, {.9: pair(1.2, .3), 1.1: pair(.9, -.3)}, alternative, ["a", "b"], 5, config)
    assert first["alternative_matched"]["head_scale"] == .5
    assert second["alternative_matched"]["head_scale"] == .75
    assert first["gain_matched"]["gain"] == 1.2
    assert second["gain_matched"]["gain"] == 1.1
    assert first["centered_weak"]["offset"] == pytest.approx(-.3, abs=1e-10)
    assert first["raw_weak"]["offset"] == 0
    assert first["centered_weak"]["achieved_rms"] == pytest.approx(.2 / np.sqrt(2), abs=1e-10)
    np.testing.assert_allclose(first_outputs["centered_weak"][1], 0, atol=1e-10)
    assert set(first) == set(runner.VARIANTS)
    for cell in (first, second):
        assert cell["alternative_matched"]["matched"]
        assert cell["gain_matched"]["matched"]
    for grid in first_grids.values():
        assert all(abs(row["mean_error"]) <= 1e-10 for row in grid)


def test_evaluation_reuses_each_condition_policy_for_both_arms_and_scenarios(runner, monkeypatch):
    config = {"model_seeds": [11], "schedules": ["fixed_cadence"], "compute_durations": [0, .1],
              "timing": {"duration": 1.5, "observation_interval": .1, "decision_interval": .1,
                         "sample_interval": .05, "action_limit": 5},
              "recovery_seconds": .4,
              "competence": {"maximum_peak_error": 1, "maximum_last_second_rmse": 1}}
    scenarios = [{"id": f"synthetic-{i}", "initial_state": [0, 0], "reference": 0,
                  "duration": 1.5, "pulse_start": .2 + .1 * i, "pulse_duration": .1,
                  "amplitude": (-1) ** i * .4} for i in range(2)]
    made, calls = {}, []

    def factory(schedule, delay):
        assert (schedule, delay) not in made
        command = .2 if delay == 0 else -.3
        policy = lambda snapshot: command
        made[schedule, delay] = policy
        return policy

    real_pair = runner.paired_rollouts

    def record(plant, policy, scenario, timing):
        pair = real_pair(plant, policy, scenario, timing)
        calls.append((scenario["id"], timing.compute_duration, policy, pair))
        return pair

    monkeypatch.setattr(runner, "paired_rollouts", record)
    rows = runner.evaluate_policy(Oscillator(), factory, scenarios, config, 11, "centered_weak")
    assert len(rows) == len(calls) == 4
    assert len(made) == 2
    for (scenario_id, delay, policy, (pulse, sham)), row in zip(calls, rows):
        assert policy is made["fixed_cadence", delay]
        assert row["scenario_id"] == scenario_id
        expected = .2 if delay == 0 else -.3
        for run in (pulse, sham):
            assert all(action.value == expected for action in run.applied_actions if action.job_id is not None)
        assert row["sham_mean_applied_action"] == pytest.approx(expected * (1.5 - delay) / 1.5)
        assert row["sham_tracking_rmse"] > 0
        assert row["pulse_clipped_actions"] == row["sham_clipped_actions"] == 0


def test_drift_threshold_uses_frozen_competence_setting_and_scenario_pairs(runner):
    config = {"model_seeds": [11], "schedules": ["serial"], "compute_durations": [0],
              "competence": {"maximum_sham_rmse_gap": .002}}
    rows = [{"seed": 11, "schedule": "serial", "compute_duration": 0, "scenario_id": scenario,
             "variant": variant, "sham_tracking_rmse": error, "censored": False,
             "absolute_competence": True}
            for scenario, native, changed in [("a", .02, .025), ("b", .03, .031)]
            for variant, error in [("native", native), ("centered_weak", changed)]]
    result, = runner.drift_checks(list(reversed(rows)), config)
    assert result["mean_sham_rmse_gap"] == pytest.approx(.003)
    assert result["maximum_gap"] == .002
    assert not result["passed"]
    assert result["absolute_competence"]


@pytest.fixture
def frozen_stage(runner, tmp_path, monkeypatch):
    output, artifacts = tmp_path / "results", tmp_path / "artifacts"
    output.mkdir()
    artifacts.mkdir()
    checkpoint, protocol = tmp_path / "checkpoint.pt", tmp_path / "protocol.md"
    checkpoint.write_bytes(b"immutable checkpoint")
    protocol.write_text("prospective protocol\n")
    for name in ("config.json", "scenarios.json", "selection.json", "calibration.json"):
        (output / name).write_text("{}\n")
    (artifacts / "calibration_11.npz").write_bytes(b"immutable calibration probes")
    sources = {"experiments/run_centered_suppression.py": "source-hash"}
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "PROTOCOL", protocol)
    monkeypatch.setattr(runner.prior, "ROOT", tmp_path)
    monkeypatch.setattr(runner.prior, "source_hashes", lambda: dict(sources))
    manifest = {"source_sha256": sources, "protocol_sha256": runner.prior.sha(protocol),
                "checkpoint_sha256": {checkpoint.name: runner.prior.sha(checkpoint)},
                "output_sha256": {path.name: runner.prior.sha(path) for path in output.iterdir()},
                "artifact_sha256": {path.name: runner.prior.sha(path) for path in artifacts.iterdir()},
                "confirmation_evaluated": False}
    runner.prior.write_json(output / "calibration_manifest.json", manifest)
    return output, artifacts, checkpoint, protocol, manifest


def test_unchanged_frozen_stage_is_read_only(runner, frozen_stage):
    output, artifacts, _, _, manifest = frozen_stage
    before = {path: path.read_bytes() for path in output.iterdir()}
    assert runner.verify_stage(output, artifacts, "calibration") == manifest
    assert {path: path.read_bytes() for path in output.iterdir()} == before


@pytest.mark.parametrize("target", ["source", "protocol", "checkpoint", "config.json", "scenarios.json",
                                    "selection.json", "calibration.json", "calibration_11.npz"])
def test_frozen_stage_rejects_mutation(runner, frozen_stage, monkeypatch, target):
    output, artifacts, checkpoint, protocol, _ = frozen_stage
    if target == "source":
        monkeypatch.setattr(runner.prior, "source_hashes", lambda: {"changed": "source"})
    else:
        path = {"protocol": protocol, "checkpoint": checkpoint,
                "calibration_11.npz": artifacts / target}.get(target, output / target)
        path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed"):
        runner.verify_stage(output, artifacts, "calibration")


def test_scenario_guard_rejects_relabelled_reuse_and_reused_random_streams(runner, tmp_path, monkeypatch):
    prior_results = tmp_path / "old"
    prior_results.mkdir()
    base = {"id": "old-000", "initial_state": [0, 0], "reference": 0, "duration": 1.5,
            "pulse_start": .2, "pulse_duration": .1, "amplitude": .4}
    runner.prior.write_json(prior_results / "config.json", {"splits": {"old": {"seed": 10}}})
    runner.prior.write_json(prior_results / "scenarios.json", {"old": [base]})
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    config = {"prior_suppression_results": "old", "splits": {
        "calibration": {"count": 1, "seed": 11}, "confirmation": {"count": 1, "seed": 12}}}
    monkeypatch.setattr(runner, "make_scenarios", lambda count, seed, split: [{**base, "id": split}])
    with pytest.raises(ValueError, match="duplicate"):
        runner.generate_scenarios(config)
    for reused in (10, 12):
        changed = deepcopy(config)
        changed["splits"]["calibration"]["seed"] = reused
        with pytest.raises(ValueError, match="independent random streams"):
            runner.generate_scenarios(changed)


def test_inherited_selection_must_match_committed_bytes(runner, tmp_path, monkeypatch):
    old = tmp_path / "old"
    old.mkdir()
    config = {"prior_suppression_results": "old", "prior_suppression_commit": "frozen", "model_seeds": [11]}
    runner.prior.write_json(old / "selection.json", [{"seed": 11, "selected": True, "candidate": {"head": 1}}])
    runner.prior.write_json(old / "discovery_manifest.json", {"output_sha256": {"selection.json": runner.prior.sha(old / "selection.json")}})
    for name in ("config.json", "scenarios.json"):
        (old / name).write_text("{}\n")
    committed = {f"frozen:old/{path.name}": path.read_bytes() for path in old.iterdir()}
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda arguments, cwd: committed[arguments[-1]])
    selection, provenance = runner.verify_inherited_selection(config)
    assert selection[0]["candidate"]["head"] == 1
    assert provenance["commit"] == "frozen"
    (old / "selection.json").write_text("[]\n")
    with pytest.raises(ValueError, match="Inherited selection provenance changed"):
        runner.verify_inherited_selection(config)
