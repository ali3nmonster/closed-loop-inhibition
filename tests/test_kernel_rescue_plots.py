"""Independent checks of training-block inference and frozen corrections."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("kernel_plots", ROOT / "experiments/plot_kernel_rescue.py")
plots = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plots)


def protocol():
    return {"statistics": {"bootstrap_resamples": 1000, "bootstrap_seed": 6070001, "confidence_level": .95},
            "fresh_seeds": [101, 102], "existing_seeds": [11], "noise_taus": [.01, .2], "primary_duration": 4.,
            "confirmation_seeds": [501, 502], "probe": {"pulse_amplitudes": [-.02, .02]},
            "metric_denominator_epsilon": 1e-12}


def metric_row(seed=101, noise=.01, delay=.05, duration=4., native=2., weak=1., population="fresh"):
    row = dict(population=population, tau=.2, noise_tau=noise, seed=seed, delay=delay, duration=duration)
    for name in plots.VARIANTS:
        energy = weak if name == "joint_weak" else native
        row[f"{name}_recovery_absolute"] = energy - native
        row[f"{name}_recovery_percent"] = 100 * (energy / native - 1)
        row[f"{name}_position_rms_absolute"] = 0.
        row[f"{name}_action_rms_absolute"] = 0.
    return row


def bundle(seed=101, noise=.01, delay=.05, duration=4., weak=1.):
    settings = {name: dict(branch_scales={}, gain=1., offset=0., center=0.) for name in plots.VARIANTS}
    sham = dict(position_rms=.01, action_rms=.02, position_mean=0., action_mean=0., saturation_fraction=0.,
                task_failure=False, censored=False)
    record = dict(population="fresh", tau=.2, noise_tau=noise, seed=seed, delay=delay, duration=duration,
                  group=["L0H0"], frozen_settings=deepcopy(settings), rollouts=[], summary={}, fixed_history={"variants": {}})
    for name in plots.VARIANTS:
        recovery = dict(complete=True, normalized_position_energy=weak if name == "joint_weak" else 2.)
        record["summary"][name] = dict(sham=deepcopy(sham), pulse=deepcopy(sham), paired_recovery=deepcopy(recovery),
                                       independent_noise_seeds=2)
        for seed_value in (501, 502):
            for amplitude in (-.02, .02):
                record["rollouts"].append(dict(variant=name, noise_seed=seed_value, amplitude=amplitude,
                                               sham=deepcopy(sham), pulse=deepcopy(sham), paired_recovery=deepcopy(recovery)))
    key = plots.identity(record)
    prepared = {key: dict(variants=deepcopy(settings), group=["L0H0"], calibration={"fixed_history": {"variants": {}}})}
    mechanism = {(*key, delay): dict(status="complete", variants={name: dict(status="complete", settings=deepcopy(value),
                        simulator_parity={"passed": True}, equilibrium={"position": 0., "action": 0.},
                        linear={"spectral_radius": .8, "dominant_decay_rate_per_s": 2.}) for name, value in settings.items()})}
    passive = {(.2, noise, duration): {"summary": {"zero_policy": record["summary"]["native"]}}}
    return record, prepared, mechanism, passive


def flatten(bundles):
    preparations, mechanisms, passive = {}, {}, {}
    for record, p, m, z in bundles:
        preparations.update(p)
        mechanisms.update(m)
        passive.update(z)
    return plots.flatten([x[0] for x in bundles], preparations, mechanisms, passive, protocol())


def test_interaction_pairs_within_model_and_horizon_before_any_aggregation():
    rows = [metric_row(delay=.05, native=2., weak=1.), metric_row(delay=.1, native=10., weak=12.),
            metric_row(seed=102, delay=.05, native=4., weak=5.), metric_row(seed=102, delay=.1, native=1., weak=.5),
            metric_row(duration=12., delay=.05, native=2., weak=3.), metric_row(duration=12., delay=.1, native=2., weak=8.)]
    result = plots.interactions(rows, .05, .1)
    first = next(row for row in result if row["seed"] == 101 and row["duration"] == 4.)
    assert first["joint_weak_recovery_absolute_interaction"] == 3.
    assert first["joint_weak_recovery_percent_interaction"] == pytest.approx(70.)
    assert first["joint_weak_helpful_to_harmful"] is True
    assert next(row for row in result if row["duration"] == 12.)["joint_weak_recovery_absolute_interaction"] == 5.
    assert next(row for row in result if row["seed"] == 102)["joint_weak_harmful_to_helpful"] is True


def test_bootstrap_uses_seed_means_and_reuses_paired_draws():
    settings = dict(resamples=2000, seed=19, confidence=.95, expected_n=3)
    values = [1., 5., 12.]
    result = plots.seed_bootstrap(values, **settings)
    indices = np.random.default_rng(19).integers(0, 3, size=(2000, 3))
    expected = np.quantile(np.asarray(values)[indices].mean(axis=1), [.025, .975])
    assert [result["ci_low"], result["ci_high"]] == pytest.approx(expected)
    shifted = plots.seed_bootstrap([v + 100 for v in values], **settings)
    assert shifted["ci_low"] - result["ci_low"] == pytest.approx(100.)


@pytest.mark.parametrize("values,expected_n", [([1., None, 3.], 3), ([1., 2.], 3), ([1., float("nan"), 3.], 3)])
def test_missing_training_blocks_invalidate_inference_not_survivor_average(values, expected_n):
    result = plots.seed_bootstrap(values, resamples=10, seed=1, confidence=.95, expected_n=expected_n)
    assert result["mean"] is result["ci_low"] is result["ci_high"] is None


def test_seed_block_inference_equal_weights_noise_cells_and_keeps_population_separate():
    rows = []
    for population, seeds in (("fresh", (101, 102)), ("existing", (11,))):
        for seed in seeds:
            for noise in (.01, .2):
                for delay in (.05, .1):
                    effect = (1. if seed == 101 else 9.) + (2. if noise == .2 else 0.)
                    rows.append(metric_row(population=population, seed=seed, noise=noise, delay=delay,
                                           weak=2. if delay == .05 else 2. + effect))
    seed_rows, summaries = plots.block_inference(plots.interactions(rows, .05, .1), protocol())
    fresh = next(row for row in summaries if row["population"] == "fresh")
    item = fresh["metrics"]["joint_weak_recovery_absolute_interaction"]
    assert [row["value"] for row in item["by_seed"]] == [2., 10.]
    assert item["mean"] == 6.
    assert item["n"] == 2  # not four environment cells or sixteen probe trials
    old = next(row for row in summaries if row["population"] == "existing")
    assert old["metrics"]["joint_weak_recovery_absolute_interaction"]["ci_low"] is None


def test_missing_environment_cell_rejects_unbalanced_training_block():
    endpoints = []
    for seed in (101, 102):
        for noise in (.01, .2):
            endpoints += plots.interactions([metric_row(seed=seed, noise=noise, delay=d) for d in (.05, .1)], .05, .1)
    with pytest.raises(ValueError, match="noise-condition grid"):
        plots.block_inference(endpoints[:-1], protocol())


def test_null_cell_retained_and_invalidates_complete_population_statistic():
    endpoints = []
    for seed in (101, 102):
        for noise in (.01, .2):
            endpoints += plots.interactions([metric_row(seed=seed, noise=noise, delay=d) for d in (.05, .1)], .05, .1)
    endpoints[0]["joint_weak_recovery_absolute_interaction"] = None
    _, summaries = plots.block_inference(endpoints, protocol())
    item = summaries[0]["metrics"]["joint_weak_recovery_absolute_interaction"]
    assert item["n"] == 2 and item["valid_n"] == 1
    assert item["mean"] is item["ci_low"] is None


@pytest.mark.parametrize("changed", ["delay", "duration"])
def test_all_corrections_are_frozen_across_delays_and_horizons(changed):
    low, high = bundle(), bundle(delay=.1) if changed == "delay" else bundle(duration=12.)
    high[0]["frozen_settings"]["weak_scalar"]["gain"] = 2.
    with pytest.raises(ValueError, match="single 50 ms preparation"):
        flatten([low, high])


def test_raw_means_recomputed_and_complete_task_failures_retained():
    item = bundle()
    for row in item[0]["rollouts"]:
        row["pulse"]["task_failure"] = True
    for summary in item[0]["summary"].values():
        summary["pulse"]["task_failure"] = 1.
    result = flatten([item])[0]
    assert result["task_failures"] == 20
    assert result["joint_weak_recovery_absolute"] == -1.
    item[0]["summary"]["joint_weak"]["paired_recovery"]["normalized_position_energy"] = 123.
    with pytest.raises(ValueError, match="disagrees"):
        flatten([item])


def test_local_identity_is_labeled_validation_and_derivative_components_cover_all_axes():
    key = ("fresh", .2, .01, 101)
    native = np.arange(7, dtype=float)
    weak = native * .8
    record = {"anchor": {"native_gradient": native.tolist(), "weak_gradient": weak.tolist()}, "variants": {
        "weak_scalar": {"correction": {"gain": 1.1}},
        "weak_kernel": {"correction": {"gain": 1.2, "kernel": (native - 1.2 * weak).tolist()}}}}
    rows = plots.kernel_rows({key: record}, .05)
    assert len(rows) == 7
    assert [(r["component"], r["lag_seconds"]) for r in rows] == [
        ("position", .05), ("velocity", .05), ("captured_action", .05),
        ("position", 0.), ("velocity", 0.), ("captured_action", 0.), ("held_action", 0.)]
    assert [r["kernel"] for r in rows] == pytest.approx(native)
    components = plots.component_summaries(rows)[0]
    assert sum(value for name, value in components.items() if name.endswith("_squared_fraction")) == pytest.approx(1.)
    assert components["kernel_gradient_error_max"] < 1e-15


def test_unavailable_controls_remain_explicit_nulls_in_full_population():
    item = bundle()
    record, prepare, mechanism, passive = item
    key = plots.identity(record)
    missing = ("weak_scalar", "weak_kernel")
    record["unavailable_controls"] = {name: "degenerate_calibration" for name in missing}
    for name in missing:
        record["summary"][name] = None
        record["frozen_settings"].pop(name)
        prepare[key]["variants"].pop(name)
        mechanism[(*key, .05)]["variants"].pop(name)
    record["rollouts"] = [row for row in record["rollouts"] if row["variant"] not in missing]
    row = flatten([item])[0]
    assert row["weak_kernel_recovery_absolute"] is None
    assert row["weak_kernel_available"] is False
    assert row["joint_weak_recovery_absolute"] == -1.
    assert row["unique_rollouts"] == 18


def test_failed_training_keeps_a_null_model_row_for_all_variants():
    item = bundle()
    record, prepare, mechanism, passive = item
    key = plots.identity(record)
    record.update(status="upstream_unavailable", frozen_settings={}, summary={name: None for name in plots.VARIANTS},
                  rollouts=[], unavailable_controls={name: "training_failed" for name in plots.VARIANTS})
    record.pop("group")
    prepare[key] = {"status": "upstream_unavailable", "variants": {}}
    mechanism[(*key, .05)] = {"status": "upstream_unavailable"}
    row = flatten([item])[0]
    assert row["native_recovery_energy"] is None
    assert row["weak_kernel_recovery_absolute"] is None
    assert row["unique_rollouts"] == 0
    assert row["group_size"] is None


def test_preparation_hash_binds_identical_settings_and_anchor_metadata():
    import hashlib
    import json
    item = bundle()
    key = plots.identity(item[0])
    digest = hashlib.sha256(json.dumps(item[1][key], sort_keys=True, allow_nan=False).encode()).hexdigest()
    item[0]["prepared_sha256"] = digest
    item[2][(*key, .05)]["prepared_sha256"] = digest
    flatten([item])
    item[1][key]["changed_anchor_metadata"] = True
    with pytest.raises(ValueError, match="exact single preparation"):
        flatten([item])


def test_numerical_mechanism_failure_is_not_treated_as_model_censoring():
    item = bundle()
    key = plots.identity(item[0])
    item[2][(*key, .05)]["status"] = "numerical_failure"
    with pytest.raises(ValueError, match="Numerical mechanism analysis"):
        flatten([item])
