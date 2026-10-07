"""Synthetic integration checks; no new scientific scenarios are evaluated."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from closed_loop_inhibition.interventions import head_residuals
from closed_loop_inhibition.neural import CausalTransformer


@pytest.fixture(scope="module")
def runner():
    path = Path(__file__).resolve().parents[1] / "experiments" / "run_suppression.py"
    spec = importlib.util.spec_from_file_location("suppression_runner_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prediction_applies_actuator_clip_after_action_gain(runner):
    class ReadFirstFeature(torch.nn.Module):
        def forward(self, tokens, valid):
            assert not torch.is_grad_enabled()
            return tokens[:, 0, 0]

    data = {
        "pulse_tokens": np.array([[[2.0]], [[-2.0]], [[0.5]]], dtype=np.float32),
        "sham_tokens": np.array([[[1.5]], [[-1.5]], [[0.0]]], dtype=np.float32),
        "valid": np.ones((3, 1), dtype=bool),
    }
    model = ReadFirstFeature().eval()
    raw = runner.predict(model, data, 5.0, clip=False)
    gained = runner.predict(model, data, 5.0, gain=0.5)
    prematurely_clipped = tuple(np.clip(values, -5, 5) * 0.5 for values in raw)
    np.testing.assert_array_equal(raw[0], [10, -10, 2.5])
    np.testing.assert_array_equal(raw[1], [7.5, -7.5, 0])
    np.testing.assert_array_equal(gained[0], [5, -5, 1.25])
    np.testing.assert_array_equal(gained[1], [3.75, -3.75, 0])
    assert not np.array_equal(gained[0], prematurely_clipped[0])
    for actual, original in zip(runner.predict(model, data, 5.0, gain=2, clip=False), raw):
        np.testing.assert_array_equal(actual, 2 * original)


def test_response_adapter_passes_discovery_calibration_and_baseline_thresholds(runner):
    config = {"discovery": {
        "minimum_fractional_increase": 0.5,
        "minimum_positive_fraction": 0.75,
        "baseline_epsilon": 1e-4,
    }}
    data = {"groups": np.array(["a", "b"]), "amplitudes": np.ones(2)}
    base = (np.ones(2), np.zeros(2))
    changed = (np.full(2, 1.25), np.zeros(2))
    discovery = runner.response(base, changed, data, config)
    calibration = runner.response(base, changed, data, config, calibration=True)
    assert discovery["median_fractional_increase"] == pytest.approx(0.25)
    assert not discovery["eligible"]
    assert calibration["eligible"]
    mixed = (np.array([2.0, 0.75]), np.zeros(2))
    assert not runner.response(base, mixed, data, config, calibration=True)["eligible"]
    near_zero = (np.full(2, 1e-5), np.zeros(2))
    assert not runner.response(near_zero, changed, data, config, calibration=True)["eligible"]


@pytest.fixture
def synthetic_outcomes():
    config = {
        "model_seeds": [11, 22], "schedules": ["fixed_cadence", "serial"],
        "compute_durations": [0.0, 0.2], "primary": {"low_delay": 0.0, "high_delay": 0.2},
    }
    # Entries are native low/high and candidate low/high, independently for
    # each seed and scenario. Neither seed nor scenario is a pooled replicate.
    values = {
        (11, "a"): (2, 5, 3, 9),
        (11, "b"): (10, 7, 9, 8),
        (22, "a"): (4, 8, 8, 9),
        (22, "b"): (9, 15, 8, 19),
    }
    rows = []
    for (seed, scenario), (native_low, native_high, changed_low, changed_high) in values.items():
        for schedule in config["schedules"]:
            multiplier = 1 if schedule == "fixed_cadence" else 2
            for delay, native, changed in ((0.0, native_low, changed_low), (0.2, native_high, changed_high)):
                for variant, score in (("native", native), ("candidate_half", native + multiplier * (changed - native))):
                    rows.append({
                        "seed": seed, "scenario_id": scenario, "variant": variant,
                        "schedule": schedule, "compute_duration": delay,
                        "censored": False, "absolute_competence": True,
                        "J_response": float(score), "peak_response": 1.0,
                        "sham_tracking_rmse": 0.1, "pulse_tracking_rmse": 0.2,
                        "pulse_action_effort": 2.0, "sham_action_effort": 1.0,
                        "pulse_action_variation": 0.4, "sham_action_variation": 0.3,
                    })
    return config, rows


def test_timing_interactions_pair_scenarios_within_seed_and_schedule(runner, synthetic_outcomes):
    config, rows = synthetic_outcomes
    aggregates, effects = runner.summarize_results(list(reversed(rows)), config)
    expected = {
        (11, "fixed_cadence"): ([3.0, 2.0], 2.5),
        (22, "fixed_cadence"): ([-3.0, 5.0], 1.0),
        (11, "serial"): ([6.0, 4.0], 5.0),
        (22, "serial"): ([-6.0, 10.0], 2.0),
    }
    assert len(effects) == 4
    for effect in effects:
        interactions, mean = expected[(effect["seed"], effect["schedule"])]
        assert effect["variant"] == "candidate_half"
        assert effect["complete_paired_scenarios"] == 2
        assert effect["scenario_interactions"] == interactions
        assert effect["mean_interaction"] == mean
    assert all(row["pairs"] == 2 and row["censored_pairs"] == 0 for row in aggregates)
    assert runner.summarize_results(rows, config) == (aggregates, effects)


@pytest.mark.parametrize("variant,delay", [
    ("native", 0.0), ("native", 0.2), ("candidate_half", 0.0), ("candidate_half", 0.2),
])
def test_censoring_any_quartet_member_excludes_the_entire_scenario(
    runner, synthetic_outcomes, variant, delay
):
    config, rows = synthetic_outcomes
    for row in rows:
        if (row["seed"], row["scenario_id"], row["schedule"], row["variant"], row["compute_duration"]) == (
            11, "b", "fixed_cadence", variant, delay
        ):
            row.update(censored=True, absolute_competence=False, J_response=None)
    aggregates, effects = runner.summarize_results(rows, config)
    target = next(effect for effect in effects if effect["seed"] == 11 and effect["schedule"] == "fixed_cadence")
    assert target["complete_paired_scenarios"] == 1
    assert target["scenario_interactions"] == [3.0]
    assert target["mean_interaction"] == 3.0
    assert sum(row["censored_pairs"] for row in aggregates) == 1
    assert all(effect["complete_paired_scenarios"] == 2 for effect in effects if effect is not target)


def test_no_complete_quartets_produce_null_interaction_not_partial_average(runner, synthetic_outcomes):
    config, rows = synthetic_outcomes
    for row in rows:
        if row["seed"] == 11 and row["schedule"] == "fixed_cadence" and row["variant"] == "candidate_half":
            row.update(censored=True, absolute_competence=False, J_response=None)
    _, effects = runner.summarize_results(rows, config)
    target = next(effect for effect in effects if effect["seed"] == 11 and effect["schedule"] == "fixed_cadence")
    assert target["complete_paired_scenarios"] == 0
    assert target["scenario_interactions"] == []
    assert target["mean_interaction"] is None
    json.dumps(effects, allow_nan=False)


@pytest.fixture
def frozen_stage(runner, tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    output = repository / "results"
    artifacts = repository / "artifacts"
    checkpoint = repository / "pretrained/checkpoints/transformer_11.pt"
    output.mkdir(parents=True)
    artifacts.mkdir()
    checkpoint.parent.mkdir(parents=True)
    torch.save({"state_dict": {}}, checkpoint)
    runner.write_json(output / "config.json", {"discovery": {"attenuation_scale": 0.5}})
    runner.write_json(output / "selection.json", [{"seed": 11, "selected": False}])
    runner.write_json(output / "scenarios.json", {"discovery": [{"id": "synthetic"}]})
    np.savez_compressed(artifacts / "discovery_11.npz", values=np.array([1, 2, 3]))
    sources = {"experiments/run_suppression.py": "frozen-source-hash"}
    monkeypatch.setattr(runner, "ROOT", repository)
    monkeypatch.setattr(runner, "source_hashes", lambda: dict(sources))
    manifest = {
        "source_sha256": sources,
        "checkpoint_sha256": {str(checkpoint.relative_to(repository)): runner.sha(checkpoint)},
        "output_sha256": {path.name: runner.sha(path) for path in sorted(output.iterdir())},
        "artifact_sha256": {path.name: runner.sha(path) for path in sorted(artifacts.iterdir())},
        "confirmation_evaluated": False,
    }
    runner.write_json(output / "discovery_manifest.json", manifest)
    return output, artifacts, checkpoint, manifest


def test_unchanged_stage_verifies_without_writing_any_output(runner, frozen_stage):
    output, artifacts, _, manifest = frozen_stage
    before = {path: path.read_bytes() for path in output.iterdir()}
    assert runner.verify_stage(output, artifacts, "discovery") == manifest
    assert {path: path.read_bytes() for path in output.iterdir()} == before


def test_stage_verification_rejects_source_changes(runner, frozen_stage, monkeypatch):
    output, artifacts, _, _ = frozen_stage
    monkeypatch.setattr(runner, "source_hashes", lambda: {"experiments/run_suppression.py": "changed"})
    with pytest.raises(ValueError, match="Sources changed"):
        runner.verify_stage(output, artifacts, "discovery")


@pytest.mark.parametrize("location,name", [
    ("output", "config.json"), ("output", "selection.json"), ("output", "scenarios.json"),
    ("artifacts", "discovery_11.npz"), ("checkpoint", None),
])
def test_stage_verification_rejects_config_selection_data_or_checkpoint_mutation(
    runner, frozen_stage, location, name
):
    output, artifacts, checkpoint, _ = frozen_stage
    path = checkpoint if location == "checkpoint" else (output if location == "output" else artifacts) / name
    path.write_bytes(path.read_bytes() + b"altered")
    message = "Frozen checkpoint changed" if location == "checkpoint" else "Stage artifact changed"
    with pytest.raises(ValueError, match=message):
        runner.verify_stage(output, artifacts, "discovery")


def test_residual_norm_diagnostic_matches_valid_token_norms_without_mutation(runner):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(57)
            model = CausalTransformer(max_tokens=3, width=8).eval()
            data = {
                "pulse_tokens": torch.randn(2, 3, 10).numpy(),
                "sham_tokens": torch.randn(2, 3, 10).numpy(),
                "valid": np.array([[True, True, False], [True, False, False]]),
            }
            original_state = deepcopy(model.state_dict())
            original_flags = [parameter.requires_grad for parameter in model.parameters()]
            before = runner.predict(model, data, 5, clip=False)
            actual = runner.residual_rms(model, data)
            sums = np.zeros((2, 4))
            for arm in ("pulse", "sham"):
                contributions = head_residuals(
                    model, torch.from_numpy(data[f"{arm}_tokens"]), torch.from_numpy(data["valid"])
                ).numpy()
                valid_contributions = contributions[data["valid"]]
                sums += np.sum(valid_contributions.astype(float) ** 2, axis=(0, 3))
            expected = np.sqrt(sums / (2 * data["valid"].sum()))
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)
            for previous, current in zip(before, runner.predict(model, data, 5, clip=False)):
                np.testing.assert_array_equal(previous, current)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, original_state[name], atol=0, rtol=0)
            assert [parameter.requires_grad for parameter in model.parameters()] == original_flags
            assert not model.training
    finally:
        torch.set_num_threads(previous_threads)
