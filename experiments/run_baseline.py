"""Run the deterministic classical-control validation and save reviewable artifacts.

Usage: .venv/bin/python experiments/run_baseline.py
"""

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime
import gzip
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["svg.hashsalt"] = "closed-loop-inhibition"

from closed_loop_inhibition.controllers import ClassicalPDController, sampled_closed_loop_matrix
from closed_loop_inhibition.metrics import summarize
from closed_loop_inhibition.plants import Oscillator
from closed_loop_inhibition.signals import PiecewiseConstant
from closed_loop_inhibition.timing import TimingConfig, simulate

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"serial": "#b85430", "fixed_cadence": "#246596"}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fingerprint() -> str:
    digest = hashlib.sha256()
    for directory, suffix in (("src", ".py"), ("experiments", ".py"), ("configs", ".json")):
        for path in sorted((ROOT / directory).rglob(f"*{suffix}")):
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def save_figure(fig, output: Path, name: str) -> None:
    fig.savefig(output / f"{name}.png", dpi=160, bbox_inches="tight")
    svg = output / f"{name}.svg"
    fig.savefig(svg, bbox_inches="tight", metadata={"Date": None})
    # Matplotlib emits trailing spaces in path data; keep generated files clean.
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def analytical_validation(plant, controller, config, output):
    settings = config["validation"]
    period = settings["period"]
    gain = np.array([controller.kp, controller.kd * plant.tau])
    ad, bd = plant.discretize(period)
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    angles = np.linspace(0, 2 * np.pi, 200)
    axes[1].plot(np.cos(angles), np.sin(angles), color="0.7", lw=1, label="Unit circle")
    for steps in settings["delay_steps"]:
        timing = TimingConfig(
            duration=settings["duration"], observation_interval=period,
            decision_interval=period, sample_interval=period, compute_duration=steps * period,
            schedule="fixed_cadence", action_limit=None,
        )
        run = simulate(plant, controller, timing, initial_state=(1.0, 0.0))
        expected = [np.array([1.0, 0.0])]
        for k in range(len(run.samples) - 1):
            command = -float(gain @ expected[k - steps]) if k >= steps else 0.0
            expected.append(ad @ expected[-1] + bd * command)
        observed = np.array([[row["position"], row["velocity"]] for row in run.samples])
        error = float(np.max(np.abs(observed - np.array(expected))))
        matrix = sampled_closed_loop_matrix(plant, controller.kp, controller.kd, period, steps)
        poles = np.linalg.eigvals(matrix)
        radius = float(np.max(np.abs(poles)))
        passed = run.failure_reason is None and error < settings["trajectory_atol"]
        rows.append({"delay_steps": steps, "delay_seconds": steps * period,
                     "spectral_radius": radius, "max_abs_state_error": error,
                     "passed": bool(passed)})
        times = [row["time"] for row in run.samples]
        line, = axes[0].plot(times, observed[:, 0], label=f"Delay {steps * period:g} s")
        axes[0].plot(times[::8], np.array(expected)[::8, 0], ".", color=line.get_color(), ms=3)
        axes[1].scatter(poles.real, poles.imag, s=28, color=line.get_color(), label=f"Delay {steps * period:g} s")
    axes[0].set(xlabel="Physical time (s)", ylabel="Position", title="Event simulation and analytical samples")
    axes[0].set_yscale("symlog", linthresh=1)
    axes[1].set(xlabel="Real part", ylabel="Imaginary part", title="Augmented sampled-system eigenvalues")
    axes[1].set_aspect("equal", adjustable="box")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Classical validation: unsaturated fixed-cadence feedback", fontsize=12)
    fig.tight_layout()
    save_figure(fig, output, "analytical_validation")
    if not all(row["passed"] for row in rows):
        raise RuntimeError(f"Analytical trajectory validation failed: {rows}")
    return rows


def timing_plot(run, output):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.3]})
    samples = [row for row in run.samples if row["time"] <= 1.2]
    axes[0].plot([row["time"] for row in samples], [row["position"] for row in samples], color="#263949")
    axes[0].axvspan(0, run.config.compute_duration, color="#edc781", alpha=0.4)
    axes[0].set(ylabel="Position", title="The environment moves during the first inference interval")
    jobs = [job for job in run.jobs if job["start_time"] < 1.1]
    for i, job in enumerate(jobs):
        axes[1].broken_barh([(job["start_time"], job["complete_time"] - job["start_time"])],
                            (i - 0.3, 0.6), facecolors="#5c8498")
        axes[1].plot(job["capture_time"], i, "o", color="#246596", ms=5,
                     label="Observation captured" if i == 0 else None)
        axes[1].plot(job["apply_time"], i, ">", color="#b85430", ms=7,
                     label="Action applied" if i == 0 else None)
    axes[1].set(yticks=range(len(jobs)), yticklabels=[str(job["id"]) for job in jobs],
                ylabel="Inference job", xlabel="Physical time (s)", xlim=(0, 1.2))
    axes[1].legend(loc="upper left", fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(f"Serial scheduling with {run.config.compute_duration * 1000:g} ms virtual computation", fontsize=12)
    fig.tight_layout()
    save_figure(fig, output, "timing_timeline")


def sweep_plot(rows, config, output):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, task in zip(axes, config["tasks"]):
        for schedule in config["schedules"]:
            selected = [row for row in rows if row["task"] == task["name"] and row["schedule"] == schedule]
            ax.plot([row["compute_over_tau"] for row in selected],
                    [row["tracking_rmse"] for row in selected], "o-", color=COLORS[schedule],
                    label=schedule.replace("_", " "))
        ax.set(title=task["name"].replace("_", " ").capitalize(),
               xlabel="Computation duration / plant timescale", ylabel="Tracking RMSE")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Frozen PD pilot: timing schedules at an action limit of {config['timing']['action_limit']}", fontsize=12)
    fig.tight_layout()
    save_figure(fig, output, "delay_sweep")


def report(output, rows, validation, convergence, manifest):
    lines = ["# Classical control and timing validation", "",
             "This is a deterministic simulator-validation pilot. No transformer was trained and no inhibitory mechanism was tested.", "",
             "## Analytical checks", "",
             "Unsaturated fixed-cadence control at 0.1 s intervals, compared against an independent sampled recurrence. The initial command is zero until the first delayed result arrives.", "",
             "| Delay in seconds | Spectral radius | Maximum trajectory error |",
             "|---|---|---|"]
    for item in validation:
        lines.append(f"| {item['delay_seconds']:.2f} | {item['spectral_radius']:.6f} | {item['max_abs_state_error']:.3g} |")
    lines += ["", "A spectral radius below one indicates local linear stability for this fixed-cadence, unsaturated model. These eigenvalues do not describe the serial or saturated pilot curves.", "",
              "![Analytical validation](analytical_validation.png)", "", "## Timing and pilot results", "",
              "![Physical time and inference](timing_timeline.png)", "", "![Delay sweep](delay_sweep.png)", "",
              f"The same frozen gains and exogenous signals are used in all {len(rows)} pilot rollouts. Fixed-cadence dispatch has idealized parallel throughput; serial inference changes both decision cadence and observation age. The action limit is specified in config.json. These illustrative curves are not an optimized-controller comparison or evidence for E/I organization.", "",
              "| Schedule | Compute seconds | Mean decision interval | Pulse tracking RMSE | Pulse action effort |",
              "|---|---|---|---|---|"]
    for row in rows:
        if row["task"] == "disturbance_pulse":
            lines.append(f"| {row['schedule']} | {row['compute_duration']:.3f} | {row['mean_decision_interval']:.3f} | {row['tracking_rmse']:.6f} | {row['action_effort']:.6f} |")
    lines += ["", "## Reporting grid convergence", "",
              "For each task, the fixed-cadence condition closest to 0.1 s computation is checked while halving the reporting interval. This is a representative convergence check, not refinement of every sweep condition. Error integrals and peaks are sampled estimates; held-action effort is integrated from application events. The numerical divergence guard is event-sampled, so censoring must also be checked when refining the reporting grid.", "",
              "| Task | Computation seconds | Relative change in tracking ISE | Relative change in peak error |",
              "|---|---|---|---|"]
    for row in convergence:
        lines.append(f"| {row['task']} | {row['compute_duration']:.3f} | {row['ise_relative_change']:.3g} | {row['peak_relative_change']:.3g} |")
    lines += ["", "## Reproduce and inspect", "",
              "From the repository root:", "", "```bash",
              ".venv/bin/python -m pytest -q",
              ".venv/bin/python experiments/run_baseline.py", "```", "",
              "- `config.json` preserves all pilot settings; `metrics.csv` contains every rollout's scores and measured decision timing.",
              "- `validation.json` and `convergence.json` contain the analytical and reporting-grid checks.",
              "- `manifest.json` records dependency versions and a SHA256 fingerprint of source, experiment scripts, and configs.",
              "- Selected complete event traces are written to ignored `runs/baseline` as compressed JSON; other rollouts are reproducible from the configuration.",
              "- A censored rollout has unavailable full-episode scores, not a deceptively small truncated error. No task-success threshold has been declared.", "",
              f"Source fingerprint: `{manifest['source_sha256']}`.", "",
              "The next experiment is a small causal transformer imitation pilot with an equally informed classical teacher. Suppressive-pathway discovery starts after competent control and independent evaluation are established.", ""]
    (output / "README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/baseline")
    parser.add_argument("--raw-output", type=Path, default=ROOT / "runs/baseline")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    args.raw_output.mkdir(parents=True, exist_ok=True)
    plant = Oscillator(**config["plant"])
    controller = ClassicalPDController(**config["controller"], tau=plant.tau)
    validation = analytical_validation(plant, controller, config, output)
    rows, convergence = [], []
    runs = {}
    for task in config["tasks"]:
        reference = PiecewiseConstant(changes=task["reference_changes"])
        disturbance = PiecewiseConstant(changes=task["disturbance_changes"])
        for schedule in config["schedules"]:
            for duration in config["compute_durations"]:
                timing = TimingConfig(**config["timing"], schedule=schedule, compute_duration=duration)
                run = simulate(plant, controller, timing, tuple(task["initial_state"]), reference, disturbance)
                stats = summarize(run, task["last_change_time"], config["settling_tolerance"])
                intervals = np.diff([job["start_time"] for job in run.jobs])
                ages = [job["apply_time"] - job["capture_time"] for job in run.jobs if job["status"] == "applied"]
                row = {"task": task["name"], "schedule": schedule, "compute_duration": duration,
                       "compute_over_tau": duration / plant.tau,
                       "mean_decision_interval": float(np.mean(intervals)) if len(intervals) else None,
                       "mean_observation_age_at_application": float(np.mean(ages)) if ages else None,
                       **stats}
                rows.append(row)
                runs[(task["name"], schedule, duration)] = run
                if task["name"] == "disturbance_pulse" and duration in (0.0, 0.2, 0.4):
                    trace = {"plant": asdict(plant), "controller": asdict(controller), "task": task,
                             **run.to_dict()}
                    with gzip.open(args.raw_output / f"{task['name']}_{schedule}_{duration:g}.json.gz", "wt") as stream:
                        json.dump(trace, stream, allow_nan=False)
        convergence_delay = min(config["compute_durations"], key=lambda value: abs(value - 0.1))
        coarse = runs[(task["name"], "fixed_cadence", convergence_delay)]
        fine = simulate(plant, controller, replace(coarse.config, sample_interval=coarse.config.sample_interval / 2),
                        tuple(task["initial_state"]), reference, disturbance)
        a = summarize(coarse, task["last_change_time"], config["settling_tolerance"])
        b = summarize(fine, task["last_change_time"], config["settling_tolerance"])
        if a["censored"] or b["censored"]:
            raise RuntimeError("Reporting-grid comparison is censored; inspect event-grid guard dependence")
        convergence.append({"task": task["name"],
                            "schedule": coarse.config.schedule,
                            "compute_duration": coarse.config.compute_duration,
                            "coarse_censored": a["censored"], "fine_censored": b["censored"],
                            "coarse_sample_interval": coarse.config.sample_interval,
                            "fine_sample_interval": fine.config.sample_interval,
                            "ise_relative_change": abs(a["tracking_ise"] - b["tracking_ise"]) / max(abs(b["tracking_ise"]), 1e-12),
                            "peak_relative_change": abs(a["peak_abs_error"] - b["peak_abs_error"]) / max(abs(b["peak_abs_error"]), 1e-12)})
    if any(row["censored"] for row in rows):
        raise RuntimeError("A pilot rollout was censored; inspect it before generating the comparison report")
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "config.json", config)
    write_json(output / "validation.json", validation)
    write_json(output / "convergence.json", convergence)
    manifest = {
        "generated_at": datetime.now(ZoneInfo("Europe/Berlin")).isoformat(),
        "python": sys.version, "platform": platform.platform(),
        "dependencies": {name: version(name) for name in ("numpy", "scipy", "matplotlib", "pytest")},
        "source_commit_at_run": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_tree_dirty_at_run": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "source_sha256": fingerprint(), "rollouts": len(rows),
        "randomness": "None: deterministic plant, fixed controller and fixed signals",
    }
    write_json(output / "manifest.json", manifest)
    sweep_plot(rows, config, output)
    timeline_delay = min(config["compute_durations"], key=lambda value: abs(value - 0.2))
    timing_plot(runs[("initial_recovery", "serial", timeline_delay)], output)
    report(output, rows, validation, convergence, manifest)
    print(json.dumps({"output": str(output), "rollouts": len(rows),
                      "max_analytical_error": max(row["max_abs_state_error"] for row in validation),
                      "max_reporting_ise_change": max(row["ise_relative_change"] for row in convergence)}, indent=2))


if __name__ == "__main__":
    main()
