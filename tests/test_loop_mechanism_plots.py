"""Independent pairing, missingness and numerical publication checks."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("mechanism_plots", ROOT / "experiments/plot_loop_mechanism.py")
plots = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plots)


def bundle(seed=11, delay=0., native=2., weak=1., missing=()):
    variants = {name: dict(branch_scales={}, gain=1., center=0., offset=0.) for name in plots.VARIANTS if name not in missing}
    record = dict(tau=.2, noise_tau=.01, seed=seed, delay=delay, group=["L0H0"],
                  frozen_settings=variants, unavailable_controls={name: "degenerate" for name in missing},
                  summary={}, rollouts=[], fixed_history={"variants": {}})
    for name in plots.VARIANTS:
        if name in missing:
            record["summary"][name] = None
            continue
        sham = dict(position_rms=.01, position_mean=0., action_rms=.02, action_mean=0.,
                    saturation_fraction=0., task_failure=False, censored=False)
        recovery = dict(complete=True, normalized_position_energy=weak if name == "joint_weak" else native)
        for tape in (101, 102):
            for amplitude in (-.02, .02):
                record["rollouts"].append(dict(variant=name, noise_seed=tape, amplitude=amplitude,
                                               sham=deepcopy(sham), pulse=deepcopy(sham), paired_recovery=deepcopy(recovery)))
        record["summary"][name] = dict(sham=deepcopy(sham), pulse=deepcopy(sham),
                                       paired_recovery=deepcopy(recovery), independent_noise_seeds=2)
    key = plots.cell(record)
    prepared = {key: {"variants": deepcopy(variants), "group": ["L0H0"], "calibration": {"fixed_history": {"variants": {}}}}}
    mechanism = {key: {"status": "complete", "variants": {name: {
        "status": "complete", "settings": deepcopy(value), "simulator_parity": {"passed": True},
        "equilibrium": {"position": 0., "action": 0.},
        "linear": {"spectral_radius": .8 + delay, "dominant_decay_rate_per_s": 2. - delay}}
        for name, value in variants.items()}}}
    return record, prepared, mechanism


def flattened(bundles):
    records, preparations, mechanisms = [], {}, {}
    for record, prepared, mechanism in bundles:
        records.append(record)
        preparations.update(prepared)
        mechanisms.update(mechanism)
    passive = {(.2, .01): {"summary": {"zero_policy": records[0]["summary"]["native"]}}}
    protocol = dict(confirmation_seeds=[101, 102], probe={"pulse_amplitudes": [-.02, .02]}, metric_denominator_epsilon=1e-12)
    return plots.flatten(records, preparations, mechanisms, passive, protocol)


def test_primary_interactions_pair_models_before_population_average():
    rows = flattened([bundle(11, 0., 2., 1.), bundle(11, .1, 10., 12.),
                      bundle(22, 0., 4., 5.), bundle(22, .1, 1., .5)])
    endpoints = plots.interactions(rows, [("primary", 0., .1)])
    assert endpoints[0]["joint_weak_recovery_absolute_interaction"] == 3.
    assert endpoints[0]["joint_weak_recovery_percent_interaction"] == pytest.approx(70.)
    assert endpoints[0]["joint_weak_helpful_to_harmful"]
    assert endpoints[1]["joint_weak_recovery_absolute_interaction"] == -1.5
    assert endpoints[1]["joint_weak_harmful_to_helpful"]
    aggregate = plots.aggregate(endpoints, ("comparison",))[0]
    assert aggregate["joint_weak_recovery_absolute_interaction_mean"] == .75
    assert aggregate["joint_weak_recovery_percent_interaction_mean"] == pytest.approx(-2.5)
    # The mode interaction is a difference of intervention effects, not the
    # delay change shared by native and altered controllers.
    assert aggregate["joint_weak_spectral_radius_interaction_mean"] == 0.


def test_unavailable_control_invalidates_full_population_mean_without_dropping_model():
    rows = flattened([bundle(11, 0., missing=("weak_rescue",)), bundle(22, 0.)])
    assert rows[0]["weak_rescue_recovery_absolute"] is None
    result = plots.aggregate(rows, ("delay",))[0]
    assert result["weak_rescue_recovery_absolute_mean"] is None
    assert result["weak_rescue_recovery_absolute_n"] == 2
    assert result["weak_rescue_recovery_absolute_valid_n"] == 1


def test_seed_means_preserve_initialization_replication_unit():
    rows = [dict(seed=11, delay=0., metric=1.), dict(seed=11, delay=0., metric=3.),
            dict(seed=22, delay=0., metric=8.)]
    grouped = plots.aggregate(rows, ("seed", "delay"))
    assert [(row["seed"], row["metric_mean"], row["metric_n"]) for row in grouped] == [(11, 2., 2), (22, 8., 1)]


def test_calibrated_settings_can_change_with_delay_but_inherited_policies_cannot():
    low, high = bundle(delay=0.), bundle(delay=.1)
    high[0]["frozen_settings"]["weak_rescue"]["gain"] = .9
    high[1][(.2, .01, 11, .1)]["variants"]["weak_rescue"]["gain"] = .9
    high[2][(.2, .01, 11, .1)]["variants"]["weak_rescue"]["settings"]["gain"] = .9
    flattened([low, high])
    high[0]["frozen_settings"]["native"]["offset"] = .1
    high[1][(.2, .01, 11, .1)]["variants"]["native"]["offset"] = .1
    high[2][(.2, .01, 11, .1)]["variants"]["native"]["settings"]["offset"] = .1
    with pytest.raises(ValueError, match="changed across delays"):
        flattened([low, high])


@pytest.mark.parametrize("which", ["calibration", "mechanism", "numerical", "parity"])
def test_publication_rejects_changed_settings_or_failed_numerical_checks(which):
    item = bundle()
    key = (.2, .01, 11, 0.)
    if which == "calibration":
        item[1][key]["variants"]["native"]["gain"] = 2.
    elif which == "mechanism":
        item[2][key]["variants"]["native"]["settings"]["gain"] = 2.
    elif which == "numerical":
        item[2][key]["status"] = "numerical_failure"
    else:
        item[2][key]["variants"]["native"]["simulator_parity"]["passed"] = False
    with pytest.raises(ValueError):
        flattened([item])


def test_null_summary_requires_explicit_unavailable_reason():
    record = bundle(missing=("weak_rescue",))[0]
    record["unavailable_controls"] = {}
    with pytest.raises(ValueError, match="unavailable controls disagree"):
        plots.validate_trials(record, [101, 102], [-.02, .02])


def test_raw_confirmation_means_are_recomputed_and_failure_metrics_retained():
    item = bundle()
    for row in item[0]["rollouts"]:
        row["pulse"]["task_failure"] = True
    for summary in item[0]["summary"].values():
        summary["pulse"]["task_failure"] = 1.
    result = flattened([item])[0]
    assert result["task_failures"] == 20
    assert result["joint_weak_recovery_absolute"] == -1.
    item[0]["summary"]["joint_weak"]["paired_recovery"]["normalized_position_energy"] = 100.
    with pytest.raises(ValueError, match="disagrees"):
        flattened([item])


def test_same_phase_comparisons_are_separate_secondary_pairs():
    rows = flattened([bundle(delay=value, native=2., weak=1. + value) for value in (0., .025, .05, .075, .1)])
    comparisons = [("primary", 0., .1), ("secondary_25_75", .025, .075), ("secondary_50_100", .05, .1)]
    result = plots.interactions(rows, comparisons)
    assert [row["comparison"] for row in result] == [row[0] for row in comparisons]
    assert [row["joint_weak_recovery_absolute_interaction"] for row in result] == pytest.approx([.1, .05, .05])
    with pytest.raises(ValueError, match="endpoints are absent"):
        plots.interactions(rows, [("absent", 0., .2)])


def test_ratio_floor_and_missingness_do_not_generate_nan():
    assert plots.percent(1., 0.) is None
    assert plots.stats([1., None])["mean"] is None
    assert plots.stats([1., float("nan")])["valid_n"] == 1


def test_frequency_population_retains_unstable_resolvents_and_declares_the_count():
    pool = [dict(tau=.2, noise_tau=.01, seed=seed, frequency_hz=1., magnitude=value,
                 model_locally_stable=stable) for seed, value, stable in ((11, 1., True), (22, 9., False))]
    frequency, magnitude, unstable = plots.frequency_curve(pool, 2)
    assert frequency == [1.]
    assert magnitude == pytest.approx([3.])  # Includes the unstable model.
    assert unstable == 1
    pool[1]["magnitude"] = None
    assert plots.frequency_curve(pool, 2)[1] == [None]  # No survivor-only curve.
