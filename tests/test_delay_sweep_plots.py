"""Reporting tests target scientific pairing, missingness and cohort integrity."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "experiments/plot_delay_sweep.py"
SPEC = importlib.util.spec_from_file_location("delay_plots", PATH)
plots = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plots)


def simple_record(seed=11, delay=0., native=2., weak=1.):
    settings = {variant: {"branch_scales": {}, "gain": 1., "center": 0., "offset": 0.,
                          "magnitude_matched": True, "response_direction_concordant": True}
                for variant in plots.VARIANTS}
    record = {"tau": .1, "noise_tau": .2, "seed": seed, "delay": delay, "group": ["L0H0"],
              "frozen_settings": settings, "rollouts": [], "summary": {}, "fixed_history": {"variants": {}}}
    for variant in plots.VARIANTS:
        energy = weak if variant == "joint_weak" else native
        sham = {"position_rms": 1., "position_mean": 0., "action_rms": .2, "action_mean": 0.,
                "saturation_fraction": 0., "task_failure": False, "censored": False}
        recovery = {"complete": True, "normalized_position_energy": energy}
        for tape in (101, 102):
            for amplitude in (-.02, .02):
                record["rollouts"].append({"variant": variant, "noise_seed": tape, "amplitude": amplitude,
                                            "sham": deepcopy(sham), "pulse": deepcopy(sham),
                                            "paired_recovery": deepcopy(recovery)})
        record["summary"][variant] = {"sham": deepcopy(sham), "pulse": deepcopy(sham),
                                       "paired_recovery": recovery, "independent_noise_seeds": 2}
    return record


def passive():
    record = simple_record()
    native = record["summary"]["native"]
    return {(.1, .2): {"summary": {"zero_policy": native}}}


def test_endpoint_interactions_pair_models_before_aggregation():
    records = [simple_record(11, 0., 2., 1.), simple_record(11, .1, 10., 12.),
               simple_record(22, 0., 4., 5.), simple_record(22, .1, 1., .5)]
    rows = plots.flatten(records, passive(), [101, 102], [-.02, .02])
    endpoints = plots.endpoint_rows(rows, 0., .1)
    assert endpoints[0]["joint_weak_recovery_absolute_interaction"] == 3.
    assert endpoints[0]["joint_weak_recovery_percent_interaction"] == pytest.approx(70.)
    assert endpoints[0]["joint_weak_helpful_to_harmful"]
    assert endpoints[1]["joint_weak_recovery_absolute_interaction"] == -1.5
    assert endpoints[1]["joint_weak_harmful_to_helpful"]
    summary = plots.summarize(rows, endpoints, 0., .1)
    assert summary["primary_interactions"]["joint_weak"]["recovery_absolute_interaction"]["mean"] == .75
    # Mean of within-model percentages differs from the pooled-energy ratio.
    assert summary["primary_interactions"]["joint_weak"]["recovery_percent_interaction"]["mean"] == pytest.approx(-2.5)


def test_missing_model_endpoint_invalidates_declared_mean_and_sign_switch():
    first = simple_record(11, .1)
    for row in first["rollouts"]:
        if row["variant"] == "joint_weak":
            row["paired_recovery"]["normalized_position_energy"] = None
    first["summary"]["joint_weak"]["paired_recovery"]["normalized_position_energy"] = None
    records = [simple_record(11, 0.), first, simple_record(22, 0.), simple_record(22, .1)]
    rows = plots.flatten(records, passive(), [101, 102], [-.02, .02])
    endpoints = plots.endpoint_rows(rows, 0., .1)
    assert endpoints[0]["joint_weak_helpful_to_harmful"] is None
    field = "joint_weak_recovery_absolute_interaction"
    summary = plots.aggregate(endpoints, ("tau", "noise_tau"))[0]
    assert summary[field + "_mean"] is None
    assert summary[field + "_valid_n"] == 1
    assert summary[field + "_n"] == 2


def test_shared_shams_count_once_and_disagreement_is_rejected():
    record = simple_record()
    totals = plots.check_trials(record, [101, 102], [-.02, .02])
    assert totals["unique_rollouts"] == 5 * 2 * 3
    record["rollouts"][0]["sham"]["position_rms"] = 3.
    with pytest.raises(ValueError, match="shared sham"):
        plots.check_trials(record, [101, 102], [-.02, .02])


def test_completed_task_failure_is_retained_in_numeric_outcomes():
    record = simple_record()
    for row in record["rollouts"]:
        row["pulse"]["task_failure"] = True
    for summary in record["summary"].values():
        summary["pulse"]["task_failure"] = 1.
    rows = plots.flatten([record], passive(), [101, 102], [-.02, .02])
    assert rows[0]["task_failures"] == 20
    assert rows[0]["censored"] == 0
    assert rows[0]["joint_weak_recovery_percent"] == -50.


def test_changed_frozen_setting_across_delays_is_rejected():
    low, high = simple_record(delay=0.), simple_record(delay=.1)
    high["frozen_settings"]["joint_weak"]["offset"] = .1
    with pytest.raises(ValueError, match="changed across delays"):
        plots.flatten([low, high], passive(), [101, 102], [-.02, .02])


def test_control_cohort_depends_on_calibration_not_shifted_matching():
    records = [simple_record(11), simple_record(22)]
    records[0]["fixed_history"]["variants"]["gain_matched"] = {"matched_on_shifted_histories": False}
    records[1]["fixed_history"]["variants"]["gain_matched"] = {"matched_on_shifted_histories": True}
    records[1]["frozen_settings"]["gain_matched"]["response_direction_concordant"] = False
    rows = plots.flatten(records, passive(), [101, 102], [-.02, .02])
    summary = plots.aggregate(rows, ("tau", "noise_tau", "delay"))[0]
    assert summary["gain_matched_minus_weak_recovery_percent_n"] == 1
    assert summary["gain_matched_minus_weak_recovery_percent_mean"] == 50.
    assert summary["gain_matched_shifted_matched_on_shifted_histories_mean"] == 0.


def test_ratio_floor_and_nonfinite_outcome_are_not_imputed():
    assert plots.percent(1., 1e-12) is None
    assert plots.percent(1., 0.) is None
    assert plots.ratio(np.inf, 1.) is None
    assert plots.stats([1., None])["mean"] is None
    assert plots.stats([1., np.nan])["valid_n"] == 1


def test_duplicate_trial_and_incorrect_summary_are_rejected():
    record = simple_record()
    record["rollouts"].append(deepcopy(record["rollouts"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        plots.check_trials(record, [101, 102], [-.02, .02])
    record = simple_record()
    record["summary"]["native"]["paired_recovery"]["normalized_position_energy"] = 99.
    with pytest.raises(ValueError, match="disagrees"):
        plots.check_trials(record, [101, 102], [-.02, .02])
