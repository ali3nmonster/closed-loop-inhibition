"""Integration checks for paired experiments and the train/evaluate boundary."""

from collections import defaultdict
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from closed_loop_inhibition.neural import make_model
from closed_loop_inhibition.plants import Oscillator
from closed_loop_inhibition.records import AppliedAction, Observation, PolicyInput


@pytest.fixture(scope="module")
def runner():
    path = Path(__file__).resolve().parents[1] / "experiments" / "train_transformer.py"
    spec = importlib.util.spec_from_file_location("transformer_runner_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def config():
    return {
        "duration": 6.0,
        "schedules": ["serial", "fixed_cadence"],
        "compute_durations": [0.0, 0.05, 0.2],
        "splits": {
            "train": {"scenarios": 4, "seed": 104729},
            "validation": {"scenarios": 3, "seed": 130363},
            "test": {"scenarios": 3, "seed": 155921},
        },
        "encoding": {"max_tokens": 3, "history_seconds": 0.5, "action_limit": 5.0},
        "model": {"input_dim": 10, "max_tokens": 3, "width": 8},
        "training": {
            "epochs": 2, "patience": 10, "batch_size": 4,
            "learning_rate": 0.001, "weight_decay": 0.0001,
            "gradient_clip": 1.0,
        },
    }


def scenario_signature(episode):
    return json.dumps({name: episode[name] for name in (
        "initial_state", "reference_changes", "disturbance_changes", "duration"
    )}, sort_keys=True)


def test_exogenous_scenarios_are_paired_across_all_timing_conditions(runner, config):
    episodes = runner.make_episodes(config, "train")
    grouped = defaultdict(list)
    for episode in episodes:
        grouped[episode["scenario_id"]].append(episode)
    assert len(grouped) == config["splits"]["train"]["scenarios"]
    assert len({episode["id"] for episode in episodes}) == len(episodes)
    expected_conditions = {
        (schedule, delay)
        for schedule in config["schedules"] for delay in config["compute_durations"]
    }
    for family in grouped.values():
        assert {(episode["schedule"], episode["compute_duration"]) for episode in family} == expected_conditions
        assert len(family) == len(expected_conditions)
        assert len({scenario_signature(episode) for episode in family}) == 1


def test_split_sampling_is_repeatable_disjoint_and_independent(runner, config):
    episodes = {split: runner.make_episodes(config, split) for split in config["splits"]}
    for split, family in episodes.items():
        assert runner.make_episodes(config, split) == family
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert {episode["id"] for episode in episodes[first]}.isdisjoint(
            episode["id"] for episode in episodes[second]
        )
        assert {scenario_signature(episode) for episode in episodes[first]}.isdisjoint(
            scenario_signature(episode) for episode in episodes[second]
        )
    changed = deepcopy(config)
    changed["splits"]["train"]["seed"] += 1
    changed["splits"]["train"]["scenarios"] += 2
    assert runner.make_episodes(changed, "train") != episodes["train"]
    assert runner.make_episodes(changed, "validation") == episodes["validation"]
    assert runner.make_episodes(changed, "test") == episodes["test"]


@pytest.fixture
def small_cpu_training():
    previous_threads = torch.get_num_threads()
    numpy_state = np.random.get_state()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        yield
    torch.set_num_threads(previous_threads)
    np.random.set_state(numpy_state)


@pytest.mark.parametrize("kind", ["transformer", "mlp"])
def test_training_restores_selected_checkpoint_and_predictions(
    runner, config, tmp_path, monkeypatch, small_cpu_training, kind
):
    tokens = torch.randn(8, 3, 10)
    valid = torch.arange(3).unsqueeze(0) < torch.tensor([1, 2, 3, 1, 2, 3, 2, 3]).unsqueeze(1)
    train = (tokens, valid, torch.linspace(-0.7, 0.8, 8))
    validation = (torch.randn(4, 3, 10), valid[:4], torch.linspace(-0.4, 0.6, 4))
    checked_states = []
    actual_mse = runner.mse_on

    def controlled_selection(model, tensors):
        # Force epoch one to win while still exercising real validation and
        # two distinct optimizer updates. The returned model must be restored.
        assert np.isfinite(actual_mse(model, tensors))
        checked_states.append({name: value.clone() for name, value in model.state_dict().items()})
        return float(len(checked_states))

    monkeypatch.setattr(runner, "mse_on", controlled_selection)
    model, history, summary = runner.train_model(kind, 7, config, train, validation, tmp_path)
    assert len(history) == 2
    assert summary["best_epoch"] == 1
    assert any(not torch.equal(checked_states[0][name], value) for name, value in checked_states[1].items())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, checked_states[0][name], atol=0, rtol=0)

    checkpoint = tmp_path / f"{kind}_7.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["epoch"] == 1
    assert payload["kind"] == kind
    assert payload["model_config"] == config["model"]
    assert payload["encoding"] == config["encoding"]
    assert summary["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    restored = make_model(payload["kind"], **payload["model_config"])
    restored.load_state_dict(payload["state_dict"])
    restored.eval()
    with torch.inference_mode():
        torch.testing.assert_close(
            restored(validation[0], validation[1]), model(validation[0], validation[1]),
            atol=0, rtol=0,
        )


def test_policy_converts_normalized_prediction_to_physical_action(runner):
    class FixedPrediction(torch.nn.Module):
        def forward(self, tokens, valid):
            assert not self.training
            assert not torch.is_grad_enabled()
            assert tokens.shape == (1, 3, 10)
            assert valid.tolist() == [[True, False, False]]
            return torch.full((tokens.shape[0],), 1.25)

    snapshot = PolicyInput(
        0.1, 0.2,
        (Observation(0, 0.0, 0.0, 0.2, -0.1, 0.5, 0.0, 0.0),),
        (AppliedAction(0.0, 0.0, None),),
    )
    policy = runner.NeuralPolicy(
        FixedPrediction(), Oscillator(),
        {"max_tokens": 3, "history_seconds": 0.5, "action_limit": 5.0},
    )
    # Clipping belongs to the shared simulator, not the neural adapter.
    assert policy(snapshot) == pytest.approx(6.25)


@pytest.fixture
def frozen_run(runner, config, tmp_path, monkeypatch):
    output = tmp_path / "results"
    artifacts = tmp_path / "artifacts"
    output.mkdir()
    (artifacts / "checkpoints").mkdir(parents=True)
    runner.write_json(output / "config.json", config)
    runner.write_json(output / "episodes.json", {
        split: runner.make_episodes(config, split) for split in config["splits"]
    })
    runner.write_json(output / "training_summary.json", [{"model": "mlp", "seed": 7}])
    (output / "training_history.csv").write_text("model,seed,epoch\nmlp,7,1\n")
    (output / "validation_rollouts.csv").write_text("model,censored\nmlp,False\n")
    for split in ("train", "validation"):
        np.savez_compressed(artifacts / f"{split}_data.npz", targets=np.array([0.1, 0.2]))
    torch.save({"state_dict": {}}, artifacts / "checkpoints/mlp_7.pt")
    source_digest = "frozen-test-source"
    monkeypatch.setattr(runner, "digest_sources", lambda: source_digest)
    manifest = {
        "source_sha256": source_digest,
        "output_sha256": {
            path.name: runner.file_digest(path) for path in sorted(output.iterdir())
        },
        "artifact_sha256": {
            str(path.relative_to(artifacts)): runner.file_digest(path)
            for path in sorted(artifacts.rglob("*")) if path.is_file()
        },
        "test_evaluated": False,
    }
    runner.write_json(output / "training_manifest.json", manifest)
    return output, artifacts, manifest


def test_unchanged_training_run_verifies_without_modifying_artifacts(runner, config, frozen_run):
    output, artifacts, manifest = frozen_run
    assert runner.verify_training_run(output, artifacts, config) == manifest
    assert json.loads((output / "training_manifest.json").read_text()) == manifest


def test_evaluation_refuses_changed_configuration(runner, config, frozen_run):
    output, artifacts, _ = frozen_run
    altered = deepcopy(config)
    altered["splits"]["test"]["seed"] += 1
    with pytest.raises(ValueError, match="configuration differs"):
        runner.verify_training_run(output, artifacts, altered)


def test_evaluation_refuses_changed_source(runner, config, frozen_run, monkeypatch):
    output, artifacts, _ = frozen_run
    monkeypatch.setattr(runner, "digest_sources", lambda: "modified-source")
    with pytest.raises(ValueError, match="source changed"):
        runner.verify_training_run(output, artifacts, config)


@pytest.mark.parametrize("location,name", [
    ("output", "episodes.json"),
    ("output", "training_summary.json"),
    ("output", "training_history.csv"),
    ("output", "validation_rollouts.csv"),
    ("artifacts", "train_data.npz"),
    ("artifacts", "validation_data.npz"),
    ("artifacts", "checkpoints/mlp_7.pt"),
])
def test_evaluation_refuses_modified_record_dataset_or_checkpoint(
    runner, config, frozen_run, location, name
):
    output, artifacts, _ = frozen_run
    path = (output if location == "output" else artifacts) / name
    path.write_bytes(path.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="Training artifact changed"):
        runner.verify_training_run(output, artifacts, config)
