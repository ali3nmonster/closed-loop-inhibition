"""Prospective provenance and discovery/confirmation boundaries for grid runs."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "timescale_runner", ROOT / "experiments/run_timescale_maps.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    (tmp_path / "experiments").mkdir()
    source = tmp_path / "src/model.py"
    source.write_text("x = 1\n")
    script = tmp_path / "experiments/run_timescale_maps.py"
    script.write_text("# scientific runner\n")
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Prospective fixed protocol\n")
    config_path = tmp_path / "config.json"
    config = {"plant_taus": [.05], "noise_taus": [.01], "seeds": [11]}
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "PROTOCOL", protocol)
    monkeypatch.setattr(runner, "__file__", str(script))
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: "abc123\n")
    output, artifacts = tmp_path / "results", tmp_path / "artifacts"
    manifest = runner.initialize(output, artifacts, config_path, config)
    return config, config_path, output, artifacts, manifest


@pytest.mark.parametrize("changed", ["src/model.py", "protocol.md", "config.json"])
def test_resuming_refuses_changed_scientific_inputs(workspace, changed):
    config, config_path, output, artifacts, _ = workspace
    target = runner.ROOT / changed
    target.write_text(target.read_text() + "changed\n")
    with pytest.raises(ValueError, match="Scientific sources"):
        runner.initialize(output, artifacts, config_path, config)


def test_new_scientific_source_is_detected_but_plot_script_is_independent(workspace):
    _, config_path, _, _, manifest = workspace
    (runner.ROOT / "experiments/plot_timescale_maps.py").write_text("# plot only\n")
    runner.check_unchanged(manifest, config_path)
    (runner.ROOT / "src/new_dynamics.py").write_text("# new scientific source\n")
    with pytest.raises(ValueError, match="changed during execution"):
        runner.check_unchanged(manifest, config_path)


@pytest.mark.parametrize("damage", ["mutate", "remove"])
def test_resuming_refuses_damaged_completed_artifacts(workspace, damage):
    config, config_path, output, artifacts, manifest = workspace
    target = artifacts / "weights.pt"
    target.write_bytes(b"original checkpoint")
    runner.seal(output, manifest, [target], "training")
    if damage == "mutate":
        target.write_bytes(b"different checkpoint")
    else:
        target.unlink()
    with pytest.raises(ValueError, match="Completed artifact changed or disappeared"):
        runner.initialize(output, artifacts, config_path, config)


def test_confirmation_refuses_changed_discovery_and_calibration(workspace, monkeypatch):
    config, _, output, artifacts, manifest = workspace
    target = output / "prepared/tau_0.05_noise_0.01_seed_11.json"
    runner.write_json(target, {"selected": "discovery choice"})
    runner.seal(output, manifest, [target])
    runner.write_json(target, {"selected": "changed after calibration"})
    monkeypatch.setattr(
        runner, "confirm_model", lambda *a, **k: pytest.fail("Must reject before confirmation")
    )
    with pytest.raises(ValueError, match="changed before confirmation"):
        runner.confirmation_stage(config, output, artifacts, manifest)


def test_confirmation_records_exact_sealed_preparation(workspace, monkeypatch):
    config, _, output, artifacts, manifest = workspace
    target = output / "prepared/tau_0.05_noise_0.01_seed_11.json"
    prepared = {"selected": "discovery choice", "gain": 1.035}
    runner.write_json(target, prepared)
    runner.seal(output, manifest, [target])
    expected = runner.sha(target)
    monkeypatch.setattr(runner, "selected_model", lambda *a: (object(), artifacts))

    def confirm(*args):
        assert args[-1] == prepared
        assert runner.sha(target) == expected
        return {"effect": None}

    monkeypatch.setattr(runner, "confirm_model", confirm)
    runner.confirmation_stage(config, output, artifacts, manifest)
    result = output / "confirmation" / target.name
    assert runner.read_json(result)["prepared_sha256"] == expected
    assert str(result.relative_to(runner.ROOT)) in manifest["completed_sha256"]
    assert "confirm" in manifest["stages"]


def test_only_sealed_unchanged_stage_outputs_may_be_reused(workspace):
    _, _, output, _, manifest = workspace
    target = output / "partial.json"
    assert runner.completed(target, manifest) is False
    runner.write_json(target, {"done": True})
    with pytest.raises(ValueError, match="unsealed or changed"):
        runner.completed(target, manifest)
    runner.seal(output, manifest, [target])
    assert runner.completed(target, manifest) is True
    runner.write_json(target, {"done": False})
    with pytest.raises(ValueError, match="unsealed or changed"):
        runner.completed(target, manifest)


def test_json_preserves_null_failures_and_rejects_nonfinite_scores(tmp_path):
    target = tmp_path / "results.json"
    runner.write_json(target, {"censored": True, "position_rms": None})
    assert runner.read_json(target) == {"censored": True, "position_rms": None}
    with pytest.raises(ValueError):
        runner.write_json(tmp_path / "bad.json", {"score": np.inf})


def test_csv_uses_literal_newline_records(tmp_path):
    target = tmp_path / "results.csv"
    runner.write_csv(target, [{"censored": True, "position_rms": None}])
    assert b"\r\n" not in target.read_bytes()
    assert target.read_bytes().endswith(b"True,\n")
