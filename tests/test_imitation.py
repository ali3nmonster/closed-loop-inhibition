"""Check the information boundary between simulation, teacher, and learner."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from closed_loop_inhibition.controllers import PredictorPDController
from closed_loop_inhibition.imitation import (
    SharedTeacher,
    collect_demonstrations,
    encode_snapshot,
    shared_snapshot,
)
from closed_loop_inhibition.plants import Oscillator
from closed_loop_inhibition.records import AppliedAction, Observation, PolicyInput


def snapshot():
    observations = tuple(
        Observation(i, i * 0.05, i * 0.05 + 0.01, 0.1 * i, -0.2, 0.5, 0.3, 0.7)
        for i in range(15)
    )
    return PolicyInput(
        0.8,
        0.95,
        observations,
        (AppliedAction(0.0, 0.0, None), AppliedAction(0.6, 0.7, 1), AppliedAction(0.75, -0.4, 2)),
    )


def test_encoder_exposes_only_declared_scaled_features_and_right_padding():
    record = PolicyInput(
        0.8, 0.95,
        (Observation(3, 0.7, 0.72, 1.2, -0.6, 0.8, 0.2, -2.0),),
        (AppliedAction(0.75, 3.0, 5),),
    )
    tokens, valid = encode_snapshot(record, tau=0.5, action_limit=5.0, max_tokens=3)
    np.testing.assert_allclose(tokens[0], [1.2, -0.3, 0.8, 0.1, -0.4, 0.2, 0.04, 0.3, 0.6, 0.1])
    np.testing.assert_array_equal(valid, [True, False, False])
    np.testing.assert_array_equal(tokens[1:], 0)
    assert tokens.dtype == np.float32
    assert valid.dtype == np.bool_


def test_teacher_and_encoder_discard_identical_old_information():
    record = snapshot()
    bounded = shared_snapshot(record, max_tokens=4, history_seconds=0.5)
    assert tuple(obs.index for obs in bounded.observations) == (11, 12, 13, 14)
    assert bounded.actions == (record.actions[-1],)
    with pytest.raises(FrozenInstanceError):
        bounded.time = 100

    altered = replace(
        record,
        observations=(replace(record.observations[0], position=1e5),) + record.observations[1:],
        actions=(replace(record.actions[0], value=1e5),) + record.actions[1:],
    )
    for original, changed in zip(encode_snapshot(record, max_tokens=4), encode_snapshot(altered, max_tokens=4)):
        np.testing.assert_array_equal(original, changed)
    teacher = SharedTeacher(PredictorPDController(), max_tokens=4)
    assert teacher(record) == teacher(altered)
    assert teacher(record) == pytest.approx(np.clip(PredictorPDController()(bounded), -5, 5))


def test_window_is_relative_to_latest_capture_and_keeps_boundary_sample():
    bounded = shared_snapshot(snapshot(), max_tokens=100, history_seconds=0.5)
    assert tuple(obs.index for obs in bounded.observations) == tuple(range(4, 15))
    assert shared_snapshot(bounded, max_tokens=100, history_seconds=0.5) == bounded
    empty = replace(snapshot(), actions=())
    assert shared_snapshot(empty).actions == (AppliedAction(0.0, 0.0, None),)


@pytest.mark.parametrize("change", [
    lambda s: replace(s, planned_apply_time=0.7),
    lambda s: replace(s, observations=s.observations[:-1] + (replace(s.latest, capture_time=0.9),)),
    lambda s: replace(s, observations=s.observations[:-1] + (replace(s.latest, available_time=0.9),)),
    lambda s: replace(s, observations=(replace(s.observations[0], available_time=0.9),) + s.observations[1:]),
    lambda s: replace(s, actions=s.actions + (AppliedAction(0.9, 0.0, 9),)),
    lambda s: replace(s, observations=tuple(reversed(s.observations))),
    lambda s: replace(s, actions=tuple(reversed(s.actions))),
])
def test_teacher_and_encoder_reject_unavailable_or_unordered_information(change):
    bad = change(snapshot())
    with pytest.raises(ValueError):
        encode_snapshot(bad)
    with pytest.raises(ValueError):
        SharedTeacher(PredictorPDController())(bad)


def test_collection_uses_same_bounded_teacher_and_clipped_training_targets():
    seen = []

    def teacher(record):
        seen.append(record)
        return 20.0

    dataset = collect_demonstrations(
        Oscillator(), teacher,
        [{"id": 17, "duration": 0.16, "compute_duration": 0.0, "reference_initial": 0.8}],
        max_tokens=2, action_limit=5.0,
    )
    assert dataset["tokens"].shape == (4, 2, 10)
    np.testing.assert_array_equal(dataset["targets"], np.ones(4))
    np.testing.assert_array_equal(dataset["episode_ids"], [17] * 4)
    assert all(len(record.observations) <= 2 and len(record.actions) == 1 for record in seen)
    assert seen[1].actions[0].value == 5.0
    for index, record in enumerate(seen):
        tokens, valid = encode_snapshot(record, max_tokens=2)
        np.testing.assert_array_equal(dataset["tokens"][index], tokens)
        np.testing.assert_array_equal(dataset["valid"][index], valid)


def test_demonstration_targets_do_not_anticipate_future_reference_or_disturbance():
    teacher = PredictorPDController()
    base = {"id": "episode", "duration": 0.2, "compute_duration": 0.025}
    first = collect_demonstrations(Oscillator(), teacher, [base])
    second = collect_demonstrations(Oscillator(), teacher, [{
        **base, "reference_changes": [(0.125, 0.8)], "disturbance_changes": [(0.125, 0.3)],
    }])
    for key in ("tokens", "valid", "targets"):
        np.testing.assert_array_equal(first[key][:3], second[key][:3])
    assert first["targets"][3] != second["targets"][3]


def test_collection_is_repeatable_and_keeps_episode_membership():
    episodes = [
        {"id": 8, "duration": 0.15, "initial_state": (0.5, -0.2)},
        {"id": 9, "duration": 0.15, "reference_initial": 0.3, "schedule": "fixed_cadence"},
    ]
    first = collect_demonstrations(Oscillator(), PredictorPDController(), episodes)
    second = collect_demonstrations(Oscillator(), PredictorPDController(), episodes)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert set(first["episode_ids"]) == {8, 9}


def test_invalid_or_censored_demonstrations_raise_instead_of_silently_selecting_data():
    with pytest.raises(ValueError, match="distinct"):
        collect_demonstrations(Oscillator(), PredictorPDController(), [{"id": 1}, {"id": 1}])
    with pytest.raises(ValueError, match="censored"):
        collect_demonstrations(Oscillator(), PredictorPDController(), [{"id": 1, "initial_state": (2e6, 0)}])
    with pytest.raises(ValueError, match="finite"):
        SharedTeacher(lambda _: float("nan"))(snapshot())


@pytest.mark.parametrize("kwargs", [{"max_tokens": 0}, {"max_tokens": True}, {"tau": 0}, {"action_limit": -1}, {"history_seconds": 0}])
def test_invalid_encoder_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        encode_snapshot(snapshot(), **kwargs)
