"""Branch gating and inference invariants for collective suppression assays."""

from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from closed_loop_inhibition.collective_suppression import (
    aggregate_sensitivity, bank_metadata, branches, build_bank, central_sensitivity,
    confirm_model, discover_model, nonadditivity, predict_bank, response_effect, scale_branches,
)
from closed_loop_inhibition.neural import make_model


@pytest.fixture
def config():
    return {"period": .05, "delay": .05, "zeta": .15, "noise_dt": .0025,
            "noise_std": .02, "state_scale": .1, "output_scale": .1,
            "action_limit": .5, "max_tokens": 3, "history_seconds": .1,
            "evaluation": {"duration": .3, "burn_in": .1, "max_abs_state": 100., "failure_rms": .1}}


@pytest.fixture
def protocol():
    return {"probe": {"duration": .3, "onset": .1, "width": .05,
                      "amplitudes": [-.02, .02], "background_tau": .2, "background_std": .01},
            "discovery_seeds": [1010001], "calibration_seeds": [1020001],
            "confirmation_seeds": [1030001], "weak_scale": .9, "strong_scale": 1.1,
            "joint_weak_scales": [.95, .9, .8], "baseline_epsilon": 1e-4,
            "min_fractional_increase": .01, "min_positive_fraction": .75}


@pytest.fixture
def model():
    torch.manual_seed(824)
    return make_model("transformer", input_dim=10, max_tokens=3, width=8)


def test_partition_and_identity_gate_preserve_exact_parameters_and_predictions(model):
    assert branches(model) == ["L0H0", "L0H1", "L0H2", "L0H3", "L0MLP",
                               "L1H0", "L1H1", "L1H2", "L1H3", "L1MLP"]
    model.eval()
    copy = scale_branches(model, dict.fromkeys(branches(model), 1.))
    for key, values in model.state_dict().items():
        assert torch.equal(values, copy.state_dict()[key])
    tokens = torch.randn(3, 3, 10)
    valid = torch.ones(3, 3, dtype=torch.bool)
    with torch.inference_mode():
        assert torch.equal(model(tokens, valid), copy(tokens, valid))
    assert not copy.training and all(not parameter.requires_grad for parameter in copy.parameters())
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert copy.layers[0].linear2.weight.data_ptr() != model.layers[0].linear2.weight.data_ptr()


def test_head_gate_excludes_shared_bias_and_other_heads(model):
    copy = scale_branches(model, {"L0H1": .3})
    for name, value in model.state_dict().items():
        expected = value.clone()
        if name == "layers.0.self_attn.out_proj.weight":
            expected[:, 2:4] *= .3
        assert torch.equal(copy.state_dict()[name], expected)
    assert model.training  # cloning does not change the original mode


def test_mlp_gate_scales_entire_output_including_bias(model):
    copy = scale_branches(model, {"L1MLP": .4})
    for name, value in model.state_dict().items():
        expected = value * .4 if name in {"layers.1.linear2.weight", "layers.1.linear2.bias"} else value
        assert torch.equal(copy.state_dict()[name], expected)
    inputs = torch.randn(5, model.layers[1].linear2.in_features)
    torch.testing.assert_close(copy.layers[1].linear2(inputs), .4 * model.layers[1].linear2(inputs))


def test_joint_gates_are_disjoint_and_composition_is_order_independent(model):
    joint = scale_branches(model, {"L0H0": .7, "L0MLP": .6, "L1H3": 1.1})
    sequential = scale_branches(scale_branches(scale_branches(model, {"L1H3": 1.1}),
                                               {"L0MLP": .6}), {"L0H0": .7})
    assert all(torch.equal(value, sequential.state_dict()[name]) for name, value in joint.state_dict().items())


@pytest.mark.parametrize("scales", [{"unknown": .5}, {"L0H0": -1}, {"L0MLP": float("nan")},
                                    {"L0H0": True}, {"L0H0": "0.5"}, [1.]])
def test_invalid_gates_rejected(model, scales):
    with pytest.raises(ValueError):
        scale_branches(model, scales)


def test_gated_model_preserves_causal_prefix(model):
    gated = scale_branches(model, {"L0H0": .5, "L0MLP": 1.2, "L1MLP": 0.})
    first = torch.randn(2, 3, 10)
    second = first.clone()
    second[:, 2] += 20
    valid = torch.ones(2, 3, dtype=torch.bool)
    with torch.inference_mode():
        before = gated.forward_tokens(first, valid)
        after = gated.forward_tokens(second, valid)
    torch.testing.assert_close(before[:, :2], after[:, :2], atol=0., rtol=0.)


def test_response_floor_does_not_turn_unclassifiable_into_zero(protocol):
    bank = {"groups": np.array(["a", "b"]), "amplitudes": np.array([.01, .02])}
    result = response_effect(([1e-5, .01], [0, 0]), ([2e-5, .012], [0, 0]), bank, protocol)
    assert not result["classifiable"] and not result["eligible"]
    assert result["median_fractional_change"] is None
    assert result["positive_fraction"] is None
    assert result["per_group"][0]["absolute_change"] == pytest.approx(1e-5)
    assert result["per_group"][1]["fractional_change"] == pytest.approx(.2)


def test_central_sensitivity_is_a_two_sided_per_group_derivative(protocol):
    bank = {"groups": np.array(["a", "b"]), "amplitudes": np.array([.01, .02])}
    native = ([1., 2.], [0, 0])
    weak = response_effect(native, ([1.2, 2.2], [0, 0]), bank, protocol)
    strong = response_effect(native, ([.9, 1.8], [0, 0]), bank, protocol)
    result = central_sensitivity(weak, strong, protocol)
    assert result["median_fractional"] == pytest.approx(1.25)
    assert result["median_physical"] == pytest.approx(1.75)
    negative = deepcopy(result)
    negative["median_fractional"] *= -2
    negative["median_physical"] *= -2
    negative["median_amplitude_normalized"] *= -2
    aggregate = aggregate_sensitivity([{"central_sensitivity": result}, {"central_sensitivity": negative}])
    assert aggregate["fractional"]["positive_sum"] == pytest.approx(1.25)
    assert aggregate["fractional"]["negative_magnitude_sum"] == pytest.approx(2.5)
    assert aggregate["fractional"]["signed_sum"] == pytest.approx(-1.25)


def test_nonadditivity_is_computed_before_aggregating_scenarios(protocol):
    bank = {"groups": np.array(["a", "b"]), "amplitudes": np.array([.01, .02])}
    native = ([1., 2.], [0, 0])
    one = response_effect(native, ([1.1, 2.2], [0, 0]), bank, protocol)
    two = response_effect(native, ([1.2, 2.4], [0, 0]), bank, protocol)
    joint = response_effect(native, ([1.4, 2.5], [0, 0]), bank, protocol)
    result = nonadditivity(joint, [{"branch": "L0H0", "weak": one},
                                  {"branch": "L1MLP", "weak": two}], ["L0H0", "L1MLP"])
    assert result["per_group"][0]["absolute_nonadditivity"] == pytest.approx(.1)
    assert result["per_group"][1]["absolute_nonadditivity"] == pytest.approx(-.1)
    assert result["median_absolute_nonadditivity"] == pytest.approx(0, abs=1e-14)
    assert result["median_fractional_nonadditivity"] == pytest.approx(.025)


def test_common_bank_is_independent_of_training_noise_configuration(config, protocol):
    first = build_bank(config, protocol, .1, protocol["discovery_seeds"])
    changed_config = deepcopy(config)
    changed_config["noise_std"] = .4
    second = build_bank(changed_config, protocol, .1, protocol["discovery_seeds"])
    for arm in ("pulse", "sham"):
        for index in (0, 1):
            np.testing.assert_array_equal(first[arm][index], second[arm][index])
    assert not np.array_equal(first["pulse"][0], first["sham"][0])
    assert bank_metadata(first)["scenario_count"] == 2
    assert bank_metadata(first)["first_decision_time"] == .1
    assert bank_metadata(first)["last_decision_time"] == .25


def test_raw_predictions_are_separate_from_actuator_clipping(config, protocol, model):
    with torch.no_grad():
        model.readout.weight.zero_()
        model.readout.bias.fill_(10.)
    bank = build_bank(config, protocol, .1, protocol["discovery_seeds"])
    predictions = predict_bank(model, bank, config)
    assert np.all(predictions[0] == 1.)
    assert np.all(predictions[0] > config["action_limit"])


def test_discovery_and_confirmation_stages_keep_empty_valid_groups_distinct(config, protocol, model):
    protocol["baseline_epsilon"] = 1e-12
    protocol["min_fractional_increase"] = 1e6
    models = {key: deepcopy(model) for key in ("initial", "epoch_025", "selected")}
    discovery_bank = build_bank(config, protocol, .1, protocol["discovery_seeds"])
    discovery = discover_model(config, protocol, .1, .05, 11, models, bank=discovery_bank)
    assert discovery["checkpoints"]["initial"]["selected"] == []
    assert discovery["checkpoints"]["initial"]["eligible_fraction"] == 0.
    assert discovery["checkpoints"]["initial"] == discovery["checkpoints"]["selected"]
    before = json.dumps(discovery, allow_nan=False, sort_keys=True)
    with pytest.raises(ValueError, match="confirmation seeds"):
        confirm_model(config, protocol, .1, .05, 11, models, discovery, bank=discovery_bank)
    confirmed = confirm_model(config, protocol, .1, .05, 11, models, discovery)
    assert json.dumps(discovery, allow_nan=False, sort_keys=True) == before
    for checkpoint in confirmed["checkpoints"].values():
        for item in checkpoint["joint"]:
            assert item["empty_group"] and item["group_size"] == 0
            assert item["response"]["median_fractional_change"] == 0
        assert checkpoint["nonadditivity_weak"]["median_fractional_nonadditivity"] == 0
    json.dumps(confirmed, allow_nan=False)


def test_unresponsive_checkpoint_has_no_classified_group(config, protocol, model):
    with torch.no_grad():
        model.readout.weight.zero_()
        model.readout.bias.zero_()
    models = {key: deepcopy(model) for key in ("initial", "epoch_025", "selected")}
    discovery = discover_model(config, protocol, .1, .05, 11, models)
    assert discovery["checkpoints"]["initial"]["selected"] is None
    assert discovery["checkpoints"]["initial"]["eligible_fraction"] is None
    confirmed = confirm_model(config, protocol, .1, .05, 11, models, discovery)
    assert confirmed["checkpoints"]["initial"]["joint"] is None
    assert confirmed["checkpoints"]["initial"]["aggregate_sensitivity"]["fractional"]["positive_sum"] is None
    assert confirmed["trained_group_across_checkpoints"]["selected"] is None


def test_checkpoint_groups_and_secondary_trained_group_are_kept_separate(config, protocol, model):
    protocol["baseline_epsilon"] = 1e-12
    models = {key: deepcopy(model) for key in ("initial", "epoch_025", "selected")}
    discovery = discover_model(config, protocol, .1, .05, 11, models)
    # Preset identities exercise the two tracking schemes without selecting on
    # confirmation outcomes. This fixture is not a scientific selection claim.
    for name, group in (("initial", ["L0H0"]), ("epoch_025", []), ("selected", ["L1MLP"])):
        discovery["checkpoints"][name].update(selected=group, eligible_count=len(group),
                                               eligible_fraction=len(group) / 10)
    confirmed = confirm_model(config, protocol, .1, .05, 11, models, discovery)
    assert confirmed["checkpoints"]["initial"]["selected"] == ["L0H0"]
    assert confirmed["checkpoints"]["epoch_025"]["selected"] == []
    assert confirmed["checkpoints"]["selected"]["selected"] == ["L1MLP"]
    for values in confirmed["trained_group_across_checkpoints"]["checkpoints"].values():
        assert all(item["selected"] == ["L1MLP"] for item in values["joint"])
    # One-gate groups have no interaction by construction.
    assert confirmed["checkpoints"]["initial"]["nonadditivity_weak"]["median_absolute_nonadditivity"] == 0


def test_overlapping_streams_are_rejected(config, protocol):
    protocol["confirmation_seeds"] = protocol["discovery_seeds"]
    with pytest.raises(ValueError, match="disjoint"):
        build_bank(config, protocol, .1, protocol["discovery_seeds"])
