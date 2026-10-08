"""Fresh demonstrations, unchanged training choices and discovery boundaries."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from closed_loop_inhibition import kernel_training as training
from closed_loop_inhibition.collective_suppression import branches
from closed_loop_inhibition.timescale_maps import make_model


ROOT = Path(__file__).resolve().parents[1]


def configs():
    base = json.loads((ROOT / "configs/timescale_maps.json").read_text())
    protocol = {"fresh_seeds": [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010],
                "training_streams": {"train_base": 6110000, "validation_base": 6210000, "stride": 100}}
    return base, protocol


def test_forty_replicas_have_disjoint_demonstration_tapes_and_unchanged_budget():
    base, protocol = configs()
    families = []
    for seed in protocol["fresh_seeds"]:
        for noise in base["noise_taus"]:
            config = training.training_config(base, protocol, noise, seed)
            original = deepcopy(config)
            for name, count in (("train", "episodes"), ("validation", "validation_episodes")):
                first = config["training"][f"{name}_seed"]
                families.append(set(range(first, first + config["training"][count])))
                original["training"][f"{name}_seed"] = base["training"][f"{name}_seed"]
            assert original == base
    assert len(families) == 80
    assert all(not a & b for index, a in enumerate(families) for b in families[index + 1:])
    assert base["training"]["train_seed"] == 710001


def test_development_can_shorten_epochs_but_not_change_optimizer():
    base, protocol = configs()
    protocol.update(development=True, development_training={"epochs": 2, "intermediate_epoch": 1})
    assert training.training_config(base, protocol, .01, 101)["training"]["epochs"] == 2
    protocol["development_training"]["lr"] = .5
    with pytest.raises(ValueError, match="epochs only"):
        training.training_config(base, protocol, .01, 101)


def test_training_summary_preserves_actual_config_and_all_history(monkeypatch, tmp_path):
    base, protocol = configs()
    calls = []

    def train(config, tau, noise, seed, directory):
        calls.append((config, tau, noise, seed, directory))
        return None, {"best_epoch": 17}, [{"epoch": 1}, {"epoch": 50}]

    monkeypatch.setattr(training, "train_cell", train)
    result = training.train_replica(base, protocol, .2, .01, 101, tmp_path)
    assert result["status"] == "trained"
    assert result["summary"]["best_epoch"] == 17
    assert len(result["history"]) == 2
    assert result["physical_rollouts"] == 22
    assert calls[0][0] == result["training_config"]
    assert calls[0][3] == 101


def test_failed_training_is_an_explicit_population_record(monkeypatch, tmp_path):
    base, protocol = configs()

    def fail(*args):
        raise RuntimeError("Nonfinite training loss")

    monkeypatch.setattr(training, "train_cell", fail)
    result = training.train_replica(base, protocol, .2, .01, 101, tmp_path)
    assert result["status"] == "training_failed"
    assert "Nonfinite" in result["failure"]
    assert result["training_rollouts_expected"] == 22


def _bank():
    return dict(noise_seeds=np.asarray([6010001] * 4), groups=np.asarray(["a"] * 4),
                amplitudes=np.asarray([.02] * 4), decision_times=np.asarray([1., 1.05, 1.1, 1.15]))


@pytest.mark.parametrize("native_value,weak_value,status,selected_count", [
    (.02, .022, "selected", 10), (.02, .019, "empty_group", 0),
    (0., .001, "unclassifiable", 0)])
def test_selected_discovery_preserves_empty_and_unclassifiable_groups(monkeypatch, native_value, weak_value, status, selected_count):
    base, _ = configs()
    model = make_model("transformer", **base["model"])
    protocol = dict(discovery_seeds=[6010001], weak_scale=.9, baseline_epsilon=.0001,
                    min_fractional_increase=.01, min_positive_fraction=.75)
    monkeypatch.setattr(training, "_commands", lambda model, bank, base, scales=None:
                        (np.full(4, weak_value if scales else native_value), np.zeros(4)))
    result = training.discover_selected(base, protocol, .2, .01, 101, model, _bank())
    assert result["status"] == status
    assert len(result["group"]) == selected_count
    assert len(result["branches"]) == len(branches(model)) == 10
    assert result["physical_rollouts"] == 0
    assert (result["selected"] is None) == (status == "unclassifiable")


def test_discovery_cannot_substitute_confirmation_tapes():
    base, _ = configs()
    with pytest.raises(ValueError, match="declared streams"):
        training.discover_selected(base, {"discovery_seeds": [999]}, .2, .01, 101, None, _bank())


def test_offset_calibration_uses_its_own_streams_and_identity_native(monkeypatch):
    base, _ = configs()
    protocol = dict(calibration_delay=.05, offset_calibration_seeds=[6020001, 6020002],
                    calibration_seeds=[6030001, 6030002], weak_scale=.9)
    discovery = dict(tau=.2, noise_tau=.01, seed=101, group=[], selected=[], status="empty_group")
    calls = []
    bank = {"groups": np.asarray(["a", "a"])}

    def run(config, local, tau, noise, model, variants, seeds):
        calls.append((config, local, variants, seeds))
        return {"physical_rollouts": 6}, bank, None

    monkeypatch.setattr(training, "_run", run)
    monkeypatch.setattr(training, "_raw", lambda *args: (np.ones(2), np.zeros(2)))
    monkeypatch.setattr(training, "scale_branches", lambda model, scales: model)
    monkeypatch.setattr(training, "_fit", lambda *args: dict(center=0., gain=1., offset=1e-17))
    result = training.prepare_fresh_settings(base, protocol, .2, .01, 101, None, discovery)
    assert calls[0][0]["delay"] == .05
    assert calls[0][1]["calibration_seeds"] == calls[0][3] == [6020001, 6020002]
    assert calls[0][2]["native"]["offset"] == 0.
    assert result["variants"]["joint_weak"]["identity_group"]
    assert result["variants"]["joint_weak"]["offset"] == 0.
    assert result["variants"]["joint_weak"]["offset_fit_diagnostic"]["offset"] == 1e-17
    assert protocol["calibration_seeds"] == [6030001, 6030002]


def test_offset_censoring_is_retained_without_inventing_a_weak_fit(monkeypatch):
    base, _ = configs()
    protocol = dict(calibration_delay=.05, offset_calibration_seeds=[6020001], weak_scale=.9)
    discovery = dict(tau=.2, noise_tau=.01, seed=101, group=[], selected=None, status="unclassifiable")
    monkeypatch.setattr(training, "_run", lambda *args: ({"physical_rollouts": 3}, None, {"reason": "censored"}))
    result = training.prepare_fresh_settings(base, protocol, .2, .01, 101, None, discovery)
    assert result["status"] == "offset_calibration_failed"
    assert set(result["variants"]) == {"native"}
    assert result["physical_rollouts"] == 3
