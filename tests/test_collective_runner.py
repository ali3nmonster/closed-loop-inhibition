"""Frozen parent lineage, common-bank persistence, and assay stream boundaries."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("collective_runner", ROOT / "experiments/run_collective_suppression.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
PLOT_SPEC = importlib.util.spec_from_file_location("collective_plot", ROOT / "experiments/plot_collective_suppression.py")
plot = importlib.util.module_from_spec(PLOT_SPEC)
PLOT_SPEC.loader.exec_module(plot)


def settings():
    return (json.loads((ROOT / "configs/timescale_maps.json").read_text()),
            json.loads((ROOT / "configs/collective_suppression.json").read_text()))


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    (tmp_path / "experiments").mkdir()
    (tmp_path / "src/model.py").write_text("original = True\n")
    script = tmp_path / "experiments/run_collective_suppression.py"
    prior = tmp_path / "experiments/run_timescale_maps.py"
    script.write_text("# collective science\n")
    prior.write_text("# inherited runner\n")
    protocol_path = tmp_path / "protocol.md"
    protocol_path.write_text("Frozen protocol\n")
    base, protocol = settings()
    config = tmp_path / "config.json"
    config.write_text(json.dumps(protocol))
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.prior, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "__file__", str(script))
    monkeypatch.setattr(runner, "PRIOR_RUNNER", prior)
    monkeypatch.setattr(runner, "PROTOCOL", protocol_path)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: "abc123\n")
    output, artifacts = tmp_path / "results", tmp_path / "artifacts"
    inherited = {"parent_manifest_sha256": "recorded-parent"}
    manifest = runner.initialize(output, artifacts, config, base, protocol, inherited)
    return base, protocol, config, output, artifacts, inherited, manifest


@pytest.mark.parametrize("changed", ["src/model.py", "protocol.md", "config.json", "experiments/run_timescale_maps.py"])
def test_resumption_refuses_scientific_input_changes(workspace, changed):
    base, protocol, config, output, artifacts, inherited, _ = workspace
    path = runner.ROOT / changed
    path.write_text(path.read_text() + "changed\n")
    with pytest.raises(ValueError, match="Frozen scientific inputs changed"):
        runner.initialize(output, artifacts, config, base, protocol, inherited)


def test_config_copies_are_sealed_and_parent_binding_cannot_change(workspace):
    base, protocol, config, output, artifacts, inherited, manifest = workspace
    for filename in ("config.json", "base_config.json"):
        assert str((output / filename).relative_to(runner.ROOT)) in manifest["completed_sha256"]
    changed = {**inherited, "parent_manifest_sha256": "another-parent"}
    with pytest.raises(ValueError, match="Frozen scientific inputs changed"):
        runner.initialize(output, artifacts, config, base, protocol, changed)


def test_downstream_stage_rejects_unsealed_or_modified_dependencies(workspace):
    _, _, _, output, _, _, manifest = workspace
    target = output / "discovery/model.json"
    runner.write_json(target, {"group": ["L0H0"]})
    with pytest.raises(ValueError, match="unsealed or changed"):
        runner.checked_record(output, "discovery", target.name, manifest)
    runner.seal(output, manifest, [target])
    record, digest = runner.checked_record(output, "discovery", target.name, manifest)
    assert record == {"group": ["L0H0"]} and digest == runner.sha(target)
    runner.write_json(target, {"group": ["L0H1"]})
    with pytest.raises(ValueError, match="unsealed or changed"):
        runner.checked_record(output, "discovery", target.name, manifest)


def test_probe_bank_roundtrip_preserves_arrays_and_does_not_require_pickle(tmp_path):
    tokens = np.arange(60, dtype=np.float32).reshape(2, 3, 10)
    valid = np.asarray([[True, True, False], [True, True, True]])
    bank = {"pulse": (tokens, valid), "sham": (tokens + 1, valid.copy()),
            "groups": np.asarray(["noise_1_amplitude_-.02", "noise_2_amplitude_.02"]),
            "amplitudes": np.asarray([-.02, .02]), "noise_seeds": np.asarray([1, 2]),
            "decision_times": np.asarray([1., 1.05])}
    target = tmp_path / "banks/probe.npz"
    runner.save_bank(target, bank)
    loaded = runner.load_bank(target)
    assert set(loaded) == set(bank)
    for key, value in bank.items():
        if key in {"pulse", "sham"}:
            for expected, actual in zip(value, loaded[key]):
                np.testing.assert_array_equal(actual, expected)
                assert actual.dtype == expected.dtype
        else:
            np.testing.assert_array_equal(loaded[key], value)
            assert loaded[key].dtype == value.dtype


def test_new_and_inherited_assay_streams_must_be_disjoint():
    base, protocol = settings()
    runner.validate_streams(base, protocol)
    changed = deepcopy(protocol)
    changed["confirmation_seeds"][0] = changed["discovery_seeds"][0]
    with pytest.raises(ValueError, match="streams overlap"):
        runner.validate_streams(base, changed)
    changed = deepcopy(protocol)
    changed["dynamics"]["confirmation_seeds"][0] = base["training"]["train_seed"] + 1
    with pytest.raises(ValueError, match="overlaps inherited"):
        runner.validate_streams(base, changed)


def test_parent_verification_accepts_new_modules_but_rejects_parent_mutation(tmp_path, monkeypatch):
    base, protocol = settings()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    directory = tmp_path / protocol["parent_results"]
    directory.mkdir(parents=True)
    source = tmp_path / "old_source.py"
    checkpoint = tmp_path / "checkpoint.pt"
    source.write_text("old = True\n")
    checkpoint.write_bytes(b"frozen weights")
    manifest = {"config": base, "artifacts_directory": str(tmp_path / "runs"),
                "scientific_sources_sha256": {"old_source.py": runner.sha(source)},
                "completed_sha256": {"checkpoint.pt": runner.sha(checkpoint)}}
    runner.write_json(directory / "config.json", base)
    runner.write_json(directory / "run_manifest.json", manifest)
    committed = (directory / "run_manifest.json").read_bytes()
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: committed)
    (tmp_path / "new_source.py").write_text("new = True\n")
    recovered, inherited = runner.verify_parent(protocol)
    assert recovered == base and inherited["artifact_hashes_checked"] == 1
    checkpoint.write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="Parent scientific artifact changed"):
        runner.verify_parent(protocol)
    checkpoint.write_bytes(b"frozen weights")
    runner.write_json(directory / "run_manifest.json", {**manifest, "edited": True})
    with pytest.raises(ValueError, match="differs from its committed record"):
        runner.verify_parent(protocol)


def test_plot_missing_required_outcomes_are_not_averaged_away():
    result = plot.stats([1., None, 3.])
    assert result["mean"] is None and result["n"] == 3 and result["valid_n"] == 2


def test_frequency_phase_aggregation_is_circular_and_preserves_missingness():
    measured = plot.circular_stats([179., -179.])
    assert abs(measured["mean"]) == pytest.approx(180.)
    assert measured["range"] == pytest.approx(2.)
    assert plot.circular_stats([179., None])["mean"] is None


def test_control_contrasts_use_only_identical_matched_direction_cohorts():
    base = {"plant_taus": [.1], "noise_taus": [.2]}
    rows = [{"tau": .1, "noise_tau": .2, "seed": seed,
             "joint_weak_position_rms_percent": weak,
             "gain_matched_minus_weak_position_rms_percent": contrast,
             "gain_matched_usable": usable, "alternative_matched_usable": False}
            for seed, weak, contrast, usable in [(11, 2., 1., True), (22, 100., 50., False)]]
    result = plot.aggregate(base, rows)[0]
    assert result["joint_weak_position_rms_percent_mean"] == pytest.approx(51.)
    assert result["gain_matched_minus_weak_position_rms_percent_mean"] == pytest.approx(1.)
    assert result["gain_matched_minus_weak_position_rms_percent_n"] == 1
