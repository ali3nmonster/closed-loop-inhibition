"""Force-noise grid helpers with causal information and paired random streams.

This experiment trains a different controller in every plant/noise cell by
imitation of a delay-aware reference. It does not optimize the plant objective
directly. Scales below are physical constants declared in the experiment config.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .centered_calibration import centered_commands
from .controllers import PredictorPDController
from .imitation import SharedTeacher, encode_snapshot
from .neural import make_model
from .plants import Oscillator
from .signals import PiecewiseConstant
from .structured_signals import ou_tape, rectangular_pulse
from .timing import TimingConfig, simulate


def cell_id(tau, noise_tau):
    return f"tau_{tau:g}_noise_{noise_tau:g}"


def make_plant(config, tau):
    return Oscillator(tau=float(tau), zeta=config["zeta"])


def make_timing(config, duration, sample_interval=.01):
    return TimingConfig(
        duration=duration, observation_interval=config["period"],
        decision_interval=config["period"], compute_duration=config["delay"],
        schedule="fixed_cadence", history_seconds=config["history_seconds"],
        sample_interval=sample_interval, action_limit=config["action_limit"],
        max_abs_state=config["evaluation"]["max_abs_state"],
    )


def teacher_policy(config, plant):
    return SharedTeacher(
        PredictorPDController(plant, kp=2., kd=1.),
        max_tokens=config["max_tokens"], history_seconds=config["history_seconds"],
        action_limit=config["action_limit"],
    )


def encode(snapshot, plant, config):
    """Keep the existing causal encoder, changing only declared unit scales."""
    if config["state_scale"] <= 0 or config["output_scale"] <= 0:
        raise ValueError("state_scale and output_scale must be positive")
    tokens, valid = encode_snapshot(
        snapshot, plant.tau, config["action_limit"], config["max_tokens"],
        config["history_seconds"],
    )
    tokens[:, :4] /= config["state_scale"]
    tokens[:, [4, 8]] *= config["action_limit"] / config["output_scale"]
    return tokens, valid


@dataclass
class NeuralPolicy:
    model: object
    plant: Oscillator
    config: dict
    offset: float = 0.
    gain: float = 1.
    center: float = 0.

    def __post_init__(self):
        self.model.eval()

    def __call__(self, snapshot):
        tokens, valid = encode(snapshot, self.plant, self.config)
        with torch.inference_mode():
            raw = float(self.model(torch.from_numpy(tokens)[None],
                                   torch.from_numpy(valid)[None])[0])
        return float(centered_commands(
            np.asarray([raw * self.config["output_scale"]]),
            action_limit=self.config["action_limit"], center=self.center,
            gain=self.gain, offset=self.offset,
        )[0])


def run_episode(config, tau, noise_tau, policy, seed, duration=None,
                initial_state=(0., 0.), pulse=None, record=False):
    """Run one precomputed OU force tape, optionally with a physical pulse.

    Reusing a seed gives identical noise across controllers and pulse/sham arms.
    Across correlation times, it pairs standardized Gaussian innovations, not
    the realized force itself. Plant speed cannot change the forcing clock.
    Recorded commands are exactly what the policy returned at dispatch.
    """
    duration = config["evaluation"]["duration"] if duration is None else duration
    plant = make_plant(config, tau)
    tape = ou_tape(duration=duration, dt=config["noise_dt"],
                   correlation_time=noise_tau, std=config["noise_std"], seed=seed)
    disturbance = tape.to_piecewise_constant()
    if pulse is not None:
        square = rectangular_pulse(onset=pulse["onset"], duration=pulse["width"],
                                   amplitude=pulse["amplitude"])
        # Work in integer clock ticks so an on-grid pulse boundary cannot be
        # duplicated by floating point representation of the noise clock.
        ticks = sorted({round(t * 1e9) for t in tape.times}
                       | {round(t * 1e9) for t, _ in square.changes if t <= duration})
        changes = [(tick / 1e9, tape.at(tick / 1e9) + square.at(tick / 1e9))
                   for tick in ticks if tick > 0]
        disturbance = PiecewiseConstant(tape.at(0.) + square.at(0.), tuple(changes))
    rows = {"tokens": [], "valid": [], "decision_times": [], "commands": []}

    def recorder(snapshot):
        command = float(policy(snapshot))
        tokens, valid = encode(snapshot, plant, config)
        rows["tokens"].append(tokens)
        rows["valid"].append(valid)
        rows["decision_times"].append(snapshot.time)
        rows["commands"].append(command)
        return command

    run = simulate(plant, recorder if record else policy,
                   make_timing(config, duration, config.get("sample_interval", .01)), initial_state=initial_state,
                   disturbance=disturbance)
    if not record:
        return run
    arrays = {key: np.asarray(value) for key, value in rows.items()}
    return run, arrays


def metrics(config, run, burn_in=None):
    """True-position metrics and exact held-command integrals on one window.

    State integrals use the declared reporting grid (trapezoidal quadrature).
    A censored rollout has null window metrics, never a flattering short-prefix
    RMS. Task failure also flags a completed trajectory above the RMS threshold.
    """
    start = config["evaluation"]["burn_in"] if burn_in is None else float(burn_in)
    end = run.config.duration
    if not 0 <= start < end:
        raise ValueError("burn_in must lie in [0, duration)")
    times = np.asarray([row["time"] for row in run.samples])
    position = np.asarray([row["position"] for row in run.samples])
    if (not np.all(np.isfinite(times)) or not np.all(np.isfinite(position))
            or np.any(np.diff(times) <= 0)
            or (len(times) and (times[0] < 0 or times[-1] > end + 1e-10))):
        raise ValueError("Sample times and positions must be finite, ordered and within the horizon")
    censored = (run.failure_reason is not None or not run.samples
                or run.samples[0]["time"] > 1e-10
                or run.samples[-1]["time"] < end - 1e-10)
    result = {"censored": bool(censored), "failure_reason": run.failure_reason,
              "task_failure": bool(censored), "position_rms": None,
              "position_mean": None, "position_max": None, "action_rms": None,
              "action_mean": None, "saturation_fraction": None,
              "duration_observed": run.samples[-1]["time"] if run.samples else 0.}
    if censored:
        return result
    use = (times > start) & (times < end)
    t = np.concatenate(([start], times[use], [end]))
    q = np.concatenate(([np.interp(start, times, position)], position[use],
                        [np.interp(end, times, position)]))
    window = end - start
    position_rms = float(np.sqrt(np.trapezoid(q * q, t) / window))
    first = second = saturated = 0.
    actions = run.applied_actions
    for index, action in enumerate(actions):
        following = actions[index + 1].time if index + 1 < len(actions) else end
        held = max(0., min(end, following) - max(start, action.time))
        first += held * action.value
        second += held * action.value ** 2
        saturated += held * (abs(action.value) >= config["action_limit"] - 1e-10)
    result.update(position_rms=position_rms,
                  position_mean=float(np.trapezoid(q, t) / window),
                  position_max=float(np.max(np.abs(q))),
                  action_rms=float(np.sqrt(second / window)),
                  action_mean=float(first / window),
                  saturation_fraction=float(saturated / window),
                  task_failure=bool(position_rms > config["evaluation"]["failure_rms"]))
    return result


def collect_dataset(config, tau, noise_tau, split):
    """Teacher trajectories with independent episode-level train/valid splits."""
    settings = config["training"]
    if split in {"train", "training"}:
        count, seed = settings["episodes"], settings["train_seed"]
    elif split == "validation":
        count, seed = settings["validation_episodes"], settings["validation_seed"]
    else:
        raise ValueError("split must be train or validation")
    # Separate initial-state randomness from the OU tape stream. The same
    # standardized q and tau*v initial states are paired across all grid cells.
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    plant = make_plant(config, tau)
    teacher = teacher_policy(config, plant)
    rows = {"tokens": [], "valid": [], "targets": [], "episode_ids": []}
    for episode in range(count):
        initial = rng.normal(0., settings["initial_std"], size=2)
        initial[1] /= tau
        run, probes = run_episode(
            config, tau, noise_tau, teacher, seed + episode,
            duration=settings["duration"], initial_state=tuple(initial), record=True,
        )
        if run.failure_reason is not None:
            raise RuntimeError(f"Censored teacher episode {split}/{episode}: {run.failure_reason}")
        rows["tokens"].append(probes["tokens"])
        rows["valid"].append(probes["valid"])
        rows["targets"].append((probes["commands"] / config["output_scale"]).astype(np.float32))
        rows["episode_ids"].append(np.full(len(probes["commands"]), episode, dtype=np.int64))
    return {key: np.concatenate(values) for key, values in rows.items()}


def _config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


def prepare_data(config, tau, noise_tau, run_dir):
    """Reuse byte-identical demonstrations for all initialization seeds."""
    directory = Path(run_dir) / "data" / cell_id(tau, noise_tau)
    directory.mkdir(parents=True, exist_ok=True)
    datasets = []
    for split in ("train", "validation"):
        path = directory / f"{split}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if saved["config_sha256"].item() != _config_digest(config):
                    raise ValueError(f"Cached dataset config mismatch: {path}")
                data = {key: saved[key].copy() for key in ("tokens", "valid", "targets", "episode_ids")}
        else:
            data = collect_dataset(config, tau, noise_tau, split)
            np.savez_compressed(path, **data, config_sha256=np.asarray(_config_digest(config)))
        datasets.append(data)
    return tuple(datasets)


def load_checkpoint(path, config=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if config is not None and payload["config_sha256"] != _config_digest(config):
        raise ValueError(f"Checkpoint config mismatch: {path}")
    model = make_model("transformer", **payload["model_config"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _mse(model, data, batch_size=1024):
    model.eval()
    total = 0.
    with torch.inference_mode():
        for start in range(0, len(data[2]), batch_size):
            prediction = model(data[0][start:start + batch_size], data[1][start:start + batch_size])
            total += float(torch.sum((prediction - data[2][start:start + batch_size]) ** 2))
    return total / len(data[2])


def train_cell(config, tau, noise_tau, seed, run_dir):
    """Fixed-budget imitation; checkpoint selection uses only validation MSE.

    Initialization, fixed intermediate, last epoch and validation-selected
    checkpoints are all retained. No closed-loop confirmation result enters
    fitting or selection. Equal seeds initialize identical weights across cells.
    """
    train, validation = prepare_data(config, tau, noise_tau, run_dir)
    train = tuple(torch.from_numpy(train[key]) for key in ("tokens", "valid", "targets"))
    validation = tuple(torch.from_numpy(validation[key]) for key in ("tokens", "valid", "targets"))
    torch.manual_seed(seed)
    model = make_model("transformer", **config["model"])
    settings = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"],
                                 weight_decay=settings["weight_decay"])
    generator = torch.Generator().manual_seed(seed + 1000)
    directory = Path(run_dir) / "checkpoints" / cell_id(tau, noise_tau) / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)

    def save(name, epoch, validation_mse):
        path = directory / f"{name}.pt"
        torch.save({"state_dict": model.state_dict(), "kind": "transformer",
                    "seed": seed, "tau": tau, "noise_tau": noise_tau,
                    "epoch": epoch, "validation_mse": validation_mse,
                    "model_config": config["model"], "config": config,
                    "config_sha256": _config_digest(config)}, path)
        return path

    initial_mse = _mse(model, validation)
    save("initial", 0, initial_mse)
    best, best_epoch = float("inf"), 0
    history = []
    started = time.perf_counter()
    x, mask, target = train
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        permutation = torch.randperm(len(target), generator=generator)
        loss_sum = 0.
        for start in range(0, len(target), settings["batch_size"]):
            indices = permutation[start:start + settings["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x[indices], mask[indices])
            loss = torch.mean((prediction - target[indices]) ** 2)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Nonfinite training loss at {cell_id(tau, noise_tau)}/{seed}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
        val_loss = _mse(model, validation)
        row = {"tau": tau, "noise_tau": noise_tau, "seed": seed, "epoch": epoch,
               "train_mse": loss_sum / len(target), "validation_mse": val_loss}
        history.append(row)
        if val_loss < best:
            best, best_epoch = val_loss, epoch
            save("selected", epoch, val_loss)
        if epoch == settings["intermediate_epoch"]:
            save(f"epoch_{epoch:03d}", epoch, val_loss)
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps({"stage": "training", **row}), flush=True)
    save("final", settings["epochs"], val_loss)
    selected = directory / "selected.pt"
    model = load_checkpoint(selected, config)
    summary = {"tau": tau, "noise_tau": noise_tau, "seed": seed,
               "parameters": sum(p.numel() for p in model.parameters()),
               "epochs_run": settings["epochs"], "best_epoch": best_epoch,
               "validation_mse": best, "initial_validation_mse": initial_mse,
               "train_examples": len(train[2]), "validation_examples": len(validation[2]),
               "training_seconds": time.perf_counter() - started,
               "checkpoint": str(selected),
               "checkpoint_sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
               "initial_checkpoint": str(directory / "initial.pt"),
               "intermediate_checkpoint": str(directory / f"epoch_{settings['intermediate_epoch']:03d}.pt"),
               "final_checkpoint": str(directory / "final.pt")}
    return model, summary, history
