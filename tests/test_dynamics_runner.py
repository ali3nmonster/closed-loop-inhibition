"""Scoring, censoring, phase convention, and source freeze for dynamics checks."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from closed_loop_inhibition.records import AppliedAction
from closed_loop_inhibition.timing import SimulationResult, TimingConfig

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("dynamics_runner", ROOT / "experiments/run_dynamics_validation.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def config():
    return json.loads((ROOT / "configs/dynamics_validation.json").read_text())


def synthetic_run(*, failure=None):
    return SimulationResult(
        samples=[{"time": t, "position": 2.0, "velocity": 0.0, "action": 99.0}
                 for t in [0.0, 0.25, 0.5, 0.75, 1.0]],
        events=[], jobs=[{"raw_action": -4.0}],
        applied_actions=[AppliedAction(0, 0, None), AppliedAction(0.25, 3.0, 0),
                         AppliedAction(0.75, -1.0, 1), AppliedAction(1.0, 500.0, 2)],
        config=TimingConfig(duration=1, sample_interval=0.25), failure_reason=failure,
    )


def test_finite_window_uses_true_positions_and_exact_held_actions():
    scored = runner.finite_window_metrics(synthetic_run(), 0.25, 0.25)
    assert scored["completed"]
    assert scored["mean_position"] == 2.0
    assert scored["rms_position"] == 2.0
    assert scored["std_position"] == 0.0
    assert scored["mean_action"] == pytest.approx((0.5 * 3 - 0.25) / 0.75)
    assert scored["rms_action"] == pytest.approx(np.sqrt((0.5 * 9 + 0.25) / 0.75))
    assert scored["position_lag_correlation"] is None
    # The endpoint command is preserved as a peak, but has zero integration time.
    assert scored["maximum_abs_applied_action"] == 500
    assert scored["maximum_abs_command"] == 4


@pytest.mark.parametrize("explicit_failure", [False, True])
def test_censored_window_never_receives_full_horizon_response_scores(explicit_failure):
    run = synthetic_run(failure="guard" if explicit_failure else None)
    run.samples.pop()
    scored = runner.finite_window_metrics(run, 0.25, 0.25)
    assert not scored["completed"]
    assert scored["observed_duration"] == 0.75
    for key in ("mean_position", "rms_position", "std_position", "mean_action", "rms_action"):
        assert scored[key] is None


def test_state_error_scales_velocity_and_uses_only_provided_reference_window():
    expected = np.array([[1.0, 0.0], [0.0, 10.0]])
    actual = expected + [0, 0.1]
    assert runner.normalized_state_error(actual, expected, 0.1) == pytest.approx(0.01)
    with pytest.raises(ValueError):
        runner.normalized_state_error(actual, expected[:1], 0.1)


@pytest.mark.parametrize("rate", [-1.2, 0.8])
def test_growth_estimate_recovers_exponential_norm(rate):
    times = np.linspace(0, 8, 101)
    states = np.column_stack((np.exp(rate * times), 2 * np.exp(rate * times)))
    assert runner.empirical_growth(times, states, 0.2) == pytest.approx(rate)


def test_sine_input_phasor_convention_includes_complex_phase_and_offset():
    times = np.linspace(0.1, 4.3, 501)
    expected = 1.5 - 0.7j
    omega, amplitude = 3.1, 0.02
    positions = amplitude * (expected.real * np.sin(omega * times)
                              + expected.imag * np.cos(omega * times)) + 0.3
    measured, residual = runner.fit_phasor(times, positions, omega, amplitude)
    assert measured == pytest.approx(expected)
    assert residual < 1e-14


def test_config_generates_the_declared_fixed_cadence_unsaturated_loop():
    settings = config()
    timing = runner.timing(settings, duration=10, delay_steps=4)
    assert timing.compute_duration == 0.2
    assert timing.observation_interval == timing.decision_interval == 0.05
    assert timing.sensor_delay == timing.actuator_delay == 0
    assert timing.action_limit is None
    assert timing.schedule == "fixed_cadence"
    assert timing.max_abs_state == 1e4


def test_manifests_detect_protocol_config_and_source_changes(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    (tmp_path / "experiments").mkdir()
    source = tmp_path / "src/model.py"
    source.write_text("x = 1\n")
    configuration, protocol = tmp_path / "config.json", tmp_path / "protocol.md"
    configuration.write_text("{}\n")
    protocol.write_text("Prospective protocol\n")
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *args, **kwargs: "abc123\n")
    manifest = runner.freeze_manifest(configuration, protocol=protocol, root=tmp_path)
    runner.verify_manifest(manifest, root=tmp_path)
    for path in (source, configuration, protocol):
        original = path.read_text()
        path.write_text(original + "changed\n")
        with pytest.raises(ValueError, match="changed"):
            runner.verify_manifest(manifest, root=tmp_path)
        path.write_text(original)
    (tmp_path / "experiments/new.py").write_text("pass\n")
    with pytest.raises(ValueError, match="source"):
        runner.verify_manifest(manifest, root=tmp_path)


def test_json_is_strict_and_preserves_explicit_failed_checks(tmp_path):
    checks = []
    runner.add_check(checks, "unstable_guarded", False, value=None)
    runner.save_checks(tmp_path, checks, complete=True)
    result = json.loads((tmp_path / "checks.json").read_text())
    assert result["complete"]
    assert not result["all_passed"]
    assert result["checks"][0]["value"] is None
    with pytest.raises(ValueError, match="Nonfinite"):
        runner.write_json(tmp_path / "bad.json", {"metric": np.inf})


def test_transfer_stage_retains_censoring_without_fitting_incomplete_data(tmp_path, monkeypatch):
    settings = deepcopy(config())
    settings["plant"]["taus"] = [0.5]
    settings["anchors"] = []
    settings["transfer"]["scaled_frequencies"] = [1]
    output, artifacts = tmp_path / "results", tmp_path / "artifacts"
    output.mkdir()
    artifacts.mkdir()
    monkeypatch.setattr(runner, "simulate", lambda *args, **kwargs: synthetic_run(failure="guard"))
    monkeypatch.setattr(runner, "fit_phasor", lambda *args, **kwargs: pytest.fail("Censored data must not be fit"))
    checks = []
    rows = runner.transfer_stage(settings, output, artifacts, checks)
    assert len(rows) == len(checks) == 1
    assert not rows[0]["completed"]
    assert rows[0]["complex_relative_error"] is None
    assert rows[0]["failure_reason"] == "guard"
    assert not checks[0]["passed"]
    assert len(list(artifacts.glob("*.npz"))) == 1


def test_transfer_stage_retains_analytically_unstable_anchor(tmp_path, monkeypatch):
    settings = deepcopy(config())
    settings["plant"]["taus"] = []
    settings["anchors"] = [{"tau": 0.1, "delay_steps": 4}]
    output, artifacts = tmp_path / "results", tmp_path / "artifacts"
    output.mkdir()
    artifacts.mkdir()
    monkeypatch.setattr(runner, "simulate", lambda *args, **kwargs: pytest.fail("No attracting response at unstable anchor"))
    checks = []
    rows = runner.transfer_stage(settings, output, artifacts, checks)
    assert len(rows) == 1
    assert not rows[0]["steady_response_available"]
    assert rows[0]["classification"] == "unstable"
    assert not checks[0]["passed"]
