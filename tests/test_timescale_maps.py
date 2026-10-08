"""Contracts that protect timing, causal information and grid comparability."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from closed_loop_inhibition.records import AppliedAction, Observation, PolicyInput
from closed_loop_inhibition.timing import SimulationResult
from closed_loop_inhibition.timescale_maps import (
    NeuralPolicy, collect_dataset, encode, load_checkpoint, make_plant, make_timing, metrics,
    prepare_data, run_episode, teacher_policy, train_cell,
)


@pytest.fixture
def config():
    return {
        "zeta": .15, "period": .05, "delay": .05, "noise_dt": .0025,
        "noise_std": .02, "action_limit": .5, "state_scale": .1,
        "output_scale": .1, "max_tokens": 11, "history_seconds": .5,
        "model": {"input_dim": 10, "max_tokens": 11, "width": 8},
        "training": {"episodes": 2, "validation_episodes": 1, "duration": .2,
                     "initial_std": .05, "epochs": 2, "batch_size": 16,
                     "lr": .001, "weight_decay": .0001, "gradient_clip": 1.,
                     "intermediate_epoch": 1, "train_seed": 710001,
                     "validation_seed": 720001},
        "evaluation": {"duration": .3, "burn_in": .1, "failure_rms": .1,
                       "max_abs_state": 100.},
    }


def test_encoding_scales_physical_values_and_preserves_causality(config):
    obs = Observation(0, 0., 0., .2, .4, .1, .2, .3)
    snapshot = PolicyInput(.05, .1, (obs,), (AppliedAction(0., .3, None),))
    tokens, valid = encode(snapshot, make_plant(config, .5), config)
    np.testing.assert_allclose(tokens[0, :5], [2., 2., 1., 1., 3.])
    assert tokens[0, 8] == pytest.approx(3.)
    assert tokens[0, 7] == pytest.approx(.1)
    assert valid.sum() == 1
    assert np.all(tokens[1:] == 0)
    future = PolicyInput(.05, .1, (obs,), (AppliedAction(.06, .3, None),))
    with pytest.raises(ValueError, match="already applied"):
        encode(future, make_plant(config, .5), config)


def test_teacher_information_matches_latest_applied_action_budget(config):
    plant = make_plant(config, .1)
    obs = Observation(0, .1, .1, .02, -.1, 0., 0., .03)
    all_actions = (AppliedAction(0., -100., None), AppliedAction(.1, .03, 1))
    first = PolicyInput(.1, .15, (obs,), all_actions)
    second = PolicyInput(.1, .15, (obs,), (all_actions[-1],))
    teacher = teacher_policy(config, plant)
    assert teacher(first) == teacher(second)


def test_noise_tapes_do_not_depend_on_plant_or_controller(config):
    zero = run_episode(config, .05, .2, lambda snapshot: 0., 91)
    changed = run_episode(config, .5, .2, lambda snapshot: .1, 91)
    np.testing.assert_array_equal([s["disturbance"] for s in zero.samples],
                                  [s["disturbance"] for s in changed.samples])
    pulse = {"onset": .1, "width": .07, "amplitude": .02}
    pulsed = run_episode(config, .05, .2, lambda snapshot: 0., 91, pulse=pulse)
    delta = np.array([s["disturbance"] for s in pulsed.samples]) - np.array(
        [s["disturbance"] for s in zero.samples])
    expected = [.02 if .1 <= s["time"] < .17 else 0. for s in zero.samples]
    np.testing.assert_allclose(delta, expected, atol=1e-16)


def test_recorded_policy_histories_have_fixed_cadence_and_delay(config):
    run, probes = run_episode(config, .2, .05, lambda snapshot: .1, 42, record=True)
    np.testing.assert_allclose(np.diff(probes["decision_times"]), .05)
    np.testing.assert_allclose(probes["commands"], .1)
    np.testing.assert_allclose(probes["tokens"][:, 0, 7], .25)
    assert run.applied_actions[1].time == .05
    assert probes["valid"].dtype == np.bool_


def test_neural_policy_transforms_raw_physical_command_before_clipping(config):
    class Constant(torch.nn.Module):
        def forward(self, tokens, valid):
            return torch.full((len(tokens),), 8.)

    obs = Observation(0, 0., 0., 0., 0., 0., 0., 0.)
    snapshot = PolicyInput(0., .05, (obs,), ())
    policy = NeuralPolicy(Constant(), make_plant(config, .2), config,
                          center=.1, gain=.5, offset=-.05)
    assert policy(snapshot) == pytest.approx(.4)


def test_metrics_use_exact_held_actions_and_preserve_failed_cells(config):
    timing = make_timing(config, 1.)
    run = SimulationResult(
        samples=[{"time": t, "position": .2} for t in (0., .5, 1.)],
        events=[], jobs=[], applied_actions=[AppliedAction(0., 0., None),
                                              AppliedAction(.25, .5, 0),
                                              AppliedAction(.75, -.5, 1)],
        config=timing,
    )
    measured = metrics(config, run, burn_in=.5)
    assert measured["position_rms"] == pytest.approx(.2)
    assert measured["position_mean"] == pytest.approx(.2)
    assert measured["action_mean"] == pytest.approx(0.)
    assert measured["action_rms"] == pytest.approx(.5)
    assert measured["saturation_fraction"] == pytest.approx(1.)
    assert measured["task_failure"] and not measured["censored"]
    run.failure_reason = "state guard"
    censored = metrics(config, run, burn_in=.5)
    assert censored["task_failure"] and censored["censored"]
    assert censored["position_rms"] is None
    assert censored["action_rms"] is None


def test_initial_states_are_paired_across_cells_but_splits_are_independent(config):
    first = collect_dataset(config, .05, .01, "train")
    second = collect_dataset(config, .5, 1., "train")
    starts = np.r_[True, np.diff(first["episode_ids"]) != 0]
    np.testing.assert_allclose(first["tokens"][starts, 0, :2],
                               second["tokens"][starts, 0, :2])
    validation = collect_dataset(config, .05, .01, "validation")
    assert not np.allclose(first["tokens"][0, 0, :2], validation["tokens"][0, 0, :2])
    assert first["targets"].dtype == np.float32
    assert np.max(np.abs(first["targets"])) <= config["action_limit"] / config["output_scale"]


@pytest.mark.parametrize("times", [(.1, .2, .3), (0., .1, .2)])
def test_metrics_censor_missing_start_or_end_even_without_guard_reason(config, times):
    run = SimulationResult(samples=[{"time": t, "position": .001} for t in times],
                           events=[], jobs=[], applied_actions=[AppliedAction(0., 0., None)],
                           config=make_timing(config, .3))
    measured = metrics(config, run)
    assert measured["censored"] and measured["task_failure"]
    assert measured["position_rms"] is None


@pytest.mark.parametrize("times,positions", [([0., .2, .1, .3], [0.] * 4),
                                              ([0., .3], [0., float("nan")]),
                                              ([0., .4], [0., 0.])])
def test_metrics_reject_malformed_sample_records(config, times, positions):
    run = SimulationResult(samples=[{"time": t, "position": q} for t, q in zip(times, positions)],
                           events=[], jobs=[], applied_actions=[AppliedAction(0., 0., None)],
                           config=make_timing(config, .3))
    with pytest.raises(ValueError, match="finite, ordered"):
        metrics(config, run)


def test_dataset_cache_refuses_config_drift(config, tmp_path):
    first, _ = prepare_data(config, .2, .05, tmp_path)
    second, _ = prepare_data(config, .2, .05, tmp_path)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    changed = deepcopy(config)
    changed["noise_std"] *= 2
    with pytest.raises(ValueError, match="config mismatch"):
        prepare_data(changed, .2, .05, tmp_path)


def test_training_retains_initial_mid_final_selected_and_equal_budgets(config, tmp_path):
    torch.set_num_threads(2)
    _, summary, rows = train_cell(config, .2, .05, 11, tmp_path)
    assert [row["epoch"] for row in rows] == [1, 2]
    assert summary["epochs_run"] == 2
    for name in ("checkpoint", "initial_checkpoint", "intermediate_checkpoint", "final_checkpoint"):
        load_checkpoint(summary[name], config)
    initial = torch.load(summary["initial_checkpoint"], weights_only=True)
    final = torch.load(summary["final_checkpoint"], weights_only=True)
    assert initial["epoch"] == 0 and final["epoch"] == 2
    assert any(not torch.equal(initial["state_dict"][key], final["state_dict"][key])
               for key in initial["state_dict"])
    changed = deepcopy(config)
    changed["output_scale"] *= 2
    with pytest.raises(ValueError, match="config mismatch"):
        load_checkpoint(summary["checkpoint"], changed)
