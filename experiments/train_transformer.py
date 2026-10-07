"""Collect demonstrations, train matched-history models, and evaluate feedback control.

All delays are virtual. Model selection uses held-out demonstration loss; test
rollouts are generated only after checkpoints have been selected.
"""

import argparse
import csv
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from closed_loop_inhibition.controllers import ClassicalPDController, PredictorPDController
from closed_loop_inhibition.imitation import SharedTeacher, collect_demonstrations, encode_snapshot
from closed_loop_inhibition.metrics import summarize
from closed_loop_inhibition.neural import make_model
from closed_loop_inhibition.plants import Oscillator
from closed_loop_inhibition.signals import PiecewiseConstant
from closed_loop_inhibition.timing import TimingConfig, simulate

ROOT = Path(__file__).resolve().parents[1]
matplotlib.rcParams["svg.hashsalt"] = "closed-loop-inhibition-neural"


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def digest_sources():
    digest = hashlib.sha256()
    for directory, suffix in (("src", ".py"), ("experiments", ".py"), ("configs", ".json")):
        for path in sorted((ROOT / directory).rglob(f"*{suffix}")):
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_manifest():
    return {
        "generated_at": datetime.now(ZoneInfo("UTC")).isoformat(), "python": sys.version,
        "platform": platform.platform(), "torch_device": "cpu", "cpu_threads": torch.get_num_threads(),
        "dependencies": {name: version(name) for name in ("torch", "numpy", "scipy", "matplotlib")},
        "source_sha256": digest_sources(),
        "source_commit_at_run": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_tree_dirty_at_run": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
    }


def verify_training_run(output, artifacts, config):
    """Refuse stale checkpoints, datasets, code, or altered split definitions."""
    manifest = json.loads((output / "training_manifest.json").read_text())
    if json.loads((output / "config.json").read_text()) != config:
        raise ValueError("Evaluation configuration differs from the training run")
    if manifest["source_sha256"] != digest_sources():
        raise ValueError("Experiment source changed after training; use a separate run")
    for location, entries in ((output, manifest["output_sha256"]), (artifacts, manifest["artifact_sha256"])):
        for name, expected in entries.items():
            if file_digest(location / name) != expected:
                raise ValueError(f"Training artifact changed: {location / name}")
    return manifest


def make_episodes(config, split):
    """Independent exogenous scenarios, paired across delays and schedules."""
    settings = config["splits"][split]
    rng = np.random.default_rng(settings["seed"])
    episodes = []
    for index in range(settings["scenarios"]):
        scenario = f"{split}-{index:03d}"
        initial = [float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1))]
        if index % 3 == 0:
            references = []
        else:
            references = [[when, float(rng.uniform(-1, 1))] for when in (1.0, 3.0, 4.5)]
        pulse_start = float(rng.uniform(1.5, 2.0))
        amplitude = float(rng.uniform(-0.6, 0.6))
        disturbances = [[pulse_start, amplitude], [pulse_start + 0.25, 0.0]]
        for schedule in config["schedules"]:
            for delay in config["compute_durations"]:
                episodes.append({
                    "id": f"{scenario}-{schedule}-{delay:g}", "scenario_id": scenario,
                    "initial_state": initial, "reference_changes": references,
                    "disturbance_changes": disturbances, "schedule": schedule,
                    "compute_duration": delay, "duration": config["duration"],
                })
    return episodes


class NeuralPolicy:
    def __init__(self, model, plant, encoding):
        self.model = model.eval()
        self.plant = plant
        self.encoding = encoding

    def __call__(self, snapshot):
        tokens, valid = encode_snapshot(snapshot, tau=self.plant.tau, **self.encoding)
        with torch.inference_mode():
            normalized = self.model(torch.from_numpy(tokens[None]), torch.from_numpy(valid[None])).item()
        return float(normalized * self.encoding["action_limit"])


def tensor_data(data):
    return tuple(torch.from_numpy(data[key]) for key in ("tokens", "valid", "targets"))


def mse_on(model, tensors, batch_size=1024):
    model.eval()
    x, mask, target = tensors
    total = 0.0
    with torch.inference_mode():
        for start in range(0, len(target), batch_size):
            prediction = model(x[start:start + batch_size], mask[start:start + batch_size])
            total += float(torch.sum((prediction - target[start:start + batch_size]) ** 2))
    return total / len(target)


def train_model(kind, seed, config, train, validation, checkpoint_dir):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model(kind, **config["model"])
    settings = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    generator = torch.Generator().manual_seed(seed + 1000)
    x, mask, target = train
    best = float("inf")
    best_epoch = -1
    history = []
    checkpoint = checkpoint_dir / f"{kind}_{seed}.pt"
    started = time.perf_counter()
    for epoch in range(settings["epochs"]):
        model.train()
        permutation = torch.randperm(len(target), generator=generator)
        loss_sum = 0.0
        for start in range(0, len(target), settings["batch_size"]):
            indices = permutation[start:start + settings["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x[indices], mask[indices])
            loss = torch.mean((prediction - target[indices]) ** 2)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss: {kind}, seed {seed}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
        val_loss = mse_on(model, validation)
        row = {"model": kind, "seed": seed, "epoch": epoch + 1,
               "train_mse": loss_sum / len(target), "validation_mse": val_loss}
        history.append(row)
        if val_loss < best:
            best, best_epoch = val_loss, epoch
            torch.save({"state_dict": model.state_dict(), "kind": kind, "seed": seed,
                        "model_config": config["model"], "encoding": config["encoding"],
                        "validation_mse": best, "epoch": epoch + 1}, checkpoint)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(json.dumps({"stage": "training", **row}), flush=True)
        if epoch - best_epoch >= settings["patience"]:
            break
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, history, {"model": kind, "seed": seed, "parameters": sum(p.numel() for p in model.parameters()),
                            "best_epoch": best_epoch + 1, "epochs_run": len(history),
                            "validation_mse": best, "training_seconds": time.perf_counter() - started,
                            "checkpoint": str(checkpoint.relative_to(ROOT)) if checkpoint.is_relative_to(ROOT) else str(checkpoint),
                            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}


def evaluate(plant, policy, episodes, encoding, name, seed):
    rows = []
    for episode in episodes:
        timing = TimingConfig(
            duration=episode["duration"], observation_interval=0.05, decision_interval=0.05,
            sample_interval=0.01, history_seconds=encoding["history_seconds"],
            action_limit=encoding["action_limit"], schedule=episode["schedule"],
            compute_duration=episode["compute_duration"],
        )
        run = simulate(plant, policy, timing, initial_state=tuple(episode["initial_state"]),
                       reference=PiecewiseConstant(changes=episode["reference_changes"]),
                       disturbance=PiecewiseConstant(changes=episode["disturbance_changes"]))
        last = max([0.] + [pair[0] for pair in episode["reference_changes"] + episode["disturbance_changes"]])
        stats = summarize(run, last)
        decision_intervals = np.diff([job["start_time"] for job in run.jobs])
        applied = [job for job in run.jobs if job["status"] == "applied"]
        application_intervals = np.diff([job["apply_time"] for job in applied])
        ages = [job["apply_time"] - job["capture_time"] for job in applied]
        rows.append({"model": name, "seed": seed, "episode_id": episode["id"],
                     "scenario_id": episode["scenario_id"], "schedule": episode["schedule"],
                     "compute_duration": episode["compute_duration"],
                     "mean_decision_interval": float(np.mean(decision_intervals)) if len(decision_intervals) else None,
                     "mean_application_interval": float(np.mean(application_intervals)) if len(application_intervals) else None,
                     "mean_observation_age_at_application": float(np.mean(ages)) if ages else None,
                     **stats})
    return rows


def latency_probe(model, validation, threads):
    x, mask, _ = validation
    index = int(torch.argmax(mask.sum(dim=1)))
    args = (x[index:index + 1], mask[index:index + 1])
    durations = []
    with torch.inference_mode():
        for _ in range(20):
            model(*args)
        for _ in range(100):
            start = time.perf_counter()
            model(*args)
            durations.append((time.perf_counter() - start) * 1000)
    return {"median_forward_ms": float(np.median(durations)), "p95_forward_ms": float(np.percentile(durations, 95)),
            "cpu_threads": threads, "probe_tokens": int(mask[index].sum()),
            "scope": "CPU batch-one forward only, excludes encoding and physical I/O"}


def plots(output, history, rows, config):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = {"transformer": "#246596", "mlp": "#aa5c35", "teacher": "#222222", "pd": "#777777"}
    for kind in config["models"]:
        for i, seed in enumerate(config["model_seeds"]):
            selected = [row for row in history if row["model"] == kind and row["seed"] == seed]
            axes[0].plot([row["epoch"] for row in selected], [row["validation_mse"] for row in selected],
                         color=colors[kind], alpha=0.7, label=kind if i == 0 else None)
    axes[0].set(xlabel="Epoch", ylabel="Normalized action MSE", title="Validation imitation loss", yscale="log")
    for axis, schedule in zip(axes[1:], config["schedules"]):
        for kind in ["teacher", "pd"] + config["models"]:
            means, low, high = [], [], []
            for delay in config["compute_durations"]:
                selected = [r for r in rows if r["model"] == kind and r["schedule"] == schedule
                            and r["compute_duration"] == delay and not r["censored"]]
                seed_means = [np.mean([r["tracking_rmse"] for r in selected if r["seed"] == seed]) for seed in sorted({r["seed"] for r in selected})]
                means.append(float(np.mean(seed_means)) if seed_means else float("nan"))
                low.append(float(np.min(seed_means)) if seed_means else float("nan"))
                high.append(float(np.max(seed_means)) if seed_means else float("nan"))
            axis.plot(config["compute_durations"], means, "o-", color=colors[kind], label=kind)
            if kind in config["models"]:
                axis.fill_between(config["compute_durations"], low, high, color=colors[kind], alpha=0.15)
        axis.set(xlabel="Virtual computation duration (s)", ylabel="Tracking RMSE", title=f"Held-out control: {schedule.replace('_', ' ')}")
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "pilot_results.png", dpi=160)
    svg = output / "pilot_results.svg"
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/transformer_pilot.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/transformer_pilot")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "runs/transformer_pilot")
    parser.add_argument("--stage", choices=["train", "evaluate", "all"], default="all")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.stage == "evaluate":
        training_manifest = verify_training_run(args.output, args.artifacts, config)
    elif (args.output / "training_manifest.json").exists():
        raise ValueError("Completed training run already exists; choose a new --output and --artifacts")
    args.output.mkdir(parents=True, exist_ok=True)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    checkpoints = args.artifacts / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    torch.set_num_threads(config["training"]["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    source_at_start = digest_sources()
    plant = Oscillator(**config["plant"])
    teacher = PredictorPDController(plant, **config["teacher"])
    shared_teacher = SharedTeacher(teacher, **config["encoding"])
    episodes = {split: make_episodes(config, split) for split in config["splits"]}
    if args.stage != "evaluate":
        write_json(args.output / "config.json", config)
        write_json(args.output / "episodes.json", episodes)
    datasets = {}
    for split in ("train", "validation"):
        file = args.artifacts / f"{split}_data.npz"
        if args.stage == "evaluate":
            data = dict(np.load(file, allow_pickle=False))
        else:
            print(json.dumps({"stage": "collect", "split": split, "episodes": len(episodes[split])}), flush=True)
            data = collect_demonstrations(plant, teacher, episodes[split], **config["encoding"])
            np.savez_compressed(file, **data)
        datasets[split] = tensor_data(data)
        print(json.dumps({"stage": "dataset", "split": split, "examples": len(data["targets"])}), flush=True)
    history, summaries, models = [], [], {}
    if args.stage == "evaluate":
        summaries = json.loads((args.output / "training_summary.json").read_text())
        with (args.output / "training_history.csv").open() as stream:
            for row in csv.DictReader(stream):
                history.append({"model": row["model"], "seed": int(row["seed"]), "epoch": int(row["epoch"]),
                                "train_mse": float(row["train_mse"]), "validation_mse": float(row["validation_mse"])})
    for kind in config["models"]:
        for seed in config["model_seeds"]:
            if args.stage == "evaluate":
                payload = torch.load(checkpoints / f"{kind}_{seed}.pt", map_location="cpu", weights_only=True)
                model = make_model(kind, **config["model"])
                model.load_state_dict(payload["state_dict"])
                model.eval()
            else:
                model, curve, summary = train_model(kind, seed, config, datasets["train"], datasets["validation"], checkpoints)
                history.extend(curve)
                summary.update(latency_probe(model, datasets["validation"], config["training"]["cpu_threads"]))
                summaries.append(summary)
                write_json(args.output / "training_summary.json", summaries)
                write_csv(args.output / "training_history.csv", history)
            models[(kind, seed)] = model
    if args.stage != "evaluate":
        print(json.dumps({"stage": "validation_control", "episodes_per_controller": len(episodes["validation"])}), flush=True)
        validation_rows = evaluate(plant, shared_teacher, episodes["validation"], config["encoding"], "teacher", -1)
        for (kind, seed), model in models.items():
            validation_rows.extend(evaluate(plant, NeuralPolicy(model, plant, config["encoding"]),
                                            episodes["validation"], config["encoding"], kind, seed))
        write_csv(args.output / "validation_rollouts.csv", validation_rows)
        if source_at_start != digest_sources():
            raise RuntimeError("Experiment source changed during training")
        training_manifest = runtime_manifest()
        training_manifest.update({
            "output_sha256": {name: file_digest(args.output / name) for name in
                              ("config.json", "episodes.json", "training_summary.json", "training_history.csv", "validation_rollouts.csv")},
            "artifact_sha256": {str(path.relative_to(args.artifacts)): file_digest(path) for path in
                                [args.artifacts / "train_data.npz", args.artifacts / "validation_data.npz", *sorted(checkpoints.glob("*.pt"))]},
            "training_examples": len(datasets["train"][2]), "validation_examples": len(datasets["validation"][2]),
            "test_evaluated": False,
        })
        write_json(args.output / "training_manifest.json", training_manifest)
    if args.stage == "train":
        print(json.dumps({"stage": "trained", "checkpoints": len(models), "test_evaluated": False}), flush=True)
        return
    print(json.dumps({"stage": "test_evaluation", "episodes_per_controller": len(episodes["test"])}), flush=True)
    test_data = collect_demonstrations(plant, teacher, episodes["test"], **config["encoding"])
    test_tensors = tensor_data(test_data)
    imitation_rows = []
    for (kind, seed), model in models.items():
        mse = mse_on(model, test_tensors)
        imitation_rows.append({"model": kind, "seed": seed, "test_normalized_mse": mse,
                               "test_action_rmse": config["encoding"]["action_limit"] * mse ** 0.5})
    write_csv(args.output / "test_imitation.csv", imitation_rows)
    rows = evaluate(plant, shared_teacher, episodes["test"], config["encoding"], "teacher", -1)
    shared_pd = SharedTeacher(ClassicalPDController(**config["teacher"], tau=plant.tau), **config["encoding"])
    rows.extend(evaluate(plant, shared_pd, episodes["test"], config["encoding"], "pd", -1))
    for (kind, seed), model in models.items():
        rows.extend(evaluate(plant, NeuralPolicy(model, plant, config["encoding"]), episodes["test"],
                             config["encoding"], kind, seed))
        print(json.dumps({"stage": "evaluated", "model": kind, "seed": seed}), flush=True)
    write_csv(args.output / "test_rollouts.csv", rows)
    plots(args.output, history, rows, config)
    manifest = runtime_manifest()
    manifest.update({
        "training_manifest_sha256": file_digest(args.output / "training_manifest.json"),
        "test_rollouts": len(rows), "test_censored": sum(row["censored"] for row in rows),
        "training_examples": len(datasets["train"][2]), "validation_examples": len(datasets["validation"][2]),
        "test_imitation_examples": len(test_data["targets"]), "model_seeds": config["model_seeds"],
        "checkpoint_location": str(checkpoints),
    })
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({"stage": "complete", "output": str(args.output), "test_rollouts": len(rows),
                      "test_censored": manifest["test_censored"]}), flush=True)


if __name__ == "__main__":
    main()
