# Suppressive organization in transformer feedback control

Research project investigating whether emergent suppression and explicit excitation–inhibition inspired mechanisms improve the stability and responsiveness of transformer controllers when the environment evolves during computation.

**Central question:** Do suppressive mechanisms change the combined controller–environment dynamics beneficially, and does their causal contribution depend on the relationship between environmental timescales, observation-to-action latency, and fresh-feedback intervals?

**Status on 7 October 2026:** The three-seed transformer/MLP imitation pilot is complete: 768 held-out feedback rollouts and 108 passing tests. Both learned controllers closely reproduce their causal teacher's tracking across the tested timing conditions. The earlier classical pilot contains 36 rollouts. No suppressive-pathway or E/I mechanism experiment has been performed.

## First results

Read the [trained transformer pilot report](results/transformer_pilot/README.md). Mean held-out tracking RMSE is 0.2624 for the predictor teacher, 0.2629 for the transformer and 0.2635 for the matched-history MLP; all runs reached their horizon. The similar neural results establish usable baselines on this fixed, fully observed plant, without evidence for an attention-specific advantage.

![Learned feedback control across computation delays](results/transformer_pilot/pilot_results.png)

Read the [baseline report and plots](results/baseline/README.md). The simulator reproduces analytical sampled-feedback trajectories, including a delay-induced unstable case. The pilot separates fixed-cadence delayed delivery from serial inference, where longer computation also reduces decision frequency.

![Delay and decision cadence in the classical pilot](results/baseline/delay_sweep.png)

These are frozen classical-controller checks with prescribed signals and gains. They validate the experimental platform; they are not evidence for the proposed transformer E/I mechanism.

The [transformer pilot protocol](docs/TRANSFORMER_PILOT.md) specifies the matched information boundary, independent episode splits, checkpoint selection and delayed feedback evaluation. [Results and provenance](results/transformer_pilot/) include configuration, training curves and per-episode scores. This is a fixed-plant feasibility experiment, not a test of architectural superiority or inhibition.

## Run locally

Python 3.11 or newer is required; the saved run used Python 3.12.7. Create an isolated environment and install the recorded dependency versions:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pytest -q
.venv/bin/python experiments/run_baseline.py
```

For an installation using compatible version ranges, use `pip install -e '.[dev]'` inside the environment instead. Exact dependency versions and source fingerprints accompany the saved run. The locked versions reflect this Linux/Python environment and may require a compatible interpreter on other platforms.

The runner reads [configs/baseline.json](configs/baseline.json). It saves compact reports, CSV scores, configuration and PNG/SVG figures in `results/baseline/`; selected complete traces go to ignored `runs/baseline/`. The configuration is for the declared three-task pilot and both timing schedules. Regenerating it replaces these generated artifacts.

For the neural pilot, install the pinned CPU PyTorch build in the same isolated environment. PyTorch is optional for the classical simulator:

```bash
.venv/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-train-lock.txt
.venv/bin/python -m pytest -q
.venv/bin/python experiments/train_transformer.py --stage train --output runs/transformer_reproduction/results --artifacts runs/transformer_reproduction/artifacts
```

The training stage collects training and validation demonstrations, saves the best validation-loss checkpoint for each model/seed, and runs closed-loop validation. Inspect `training_summary.json`, `training_history.csv` and `validation_rollouts.csv` in the chosen output directory before starting final test evaluation:

```bash
.venv/bin/python experiments/train_transformer.py --stage evaluate --output runs/transformer_reproduction/results --artifacts runs/transformer_reproduction/artifacts
```

Use a new output/artifact directory for a fresh run; training refuses to overwrite a completed training manifest. Evaluation verifies the saved configuration and hashes of source, data, checkpoints and recorded training outputs before using them. Keep code and configuration unchanged between stages. The default settings are in [configs/transformer_pilot.json](configs/transformer_pilot.json); six models use the same observation histories and paired test scenarios. Large datasets and checkpoints remain under ignored `runs/`, while the saved project report uses `results/transformer_pilot/`.

The pilot uses prescribed virtual computation delays. Its separately measured CPU forward-pass latency excludes feature encoding and physical I/O and does not establish a real-time hardware control rate.

## Read first

1. [Research background and proposal](docs/RESEARCH_PROPOSAL.md) — the complete conceptual argument, evidence, definitions, hypotheses, and intended contribution.
2. [Step by step starting plan](docs/START_HERE.md) — the first experiment, milestones, controls, measurements, and decision criteria.
3. [Neuroscience evidence](docs/notes/neuroscience_evidence.md) — cortical computation, population geometry, inhibition, and embodied dynamics.
4. [Transformer evidence](docs/notes/transformer_evidence.md) — recurrence, geometry, native suppression, explicit mechanisms, and real-time control.
5. [Source and provenance guide](docs/SOURCES.md) — primary references, version cautions, and connections to existing local research.
6. [Implemented experiment contract](docs/EXPERIMENT_CONTRACT.md) — timing conventions, information access, scoring, censoring, and the current pilot's limits.
7. [Transformer pilot protocol](docs/TRANSFORMER_PILOT.md) — teacher imitation, matched-history architectures, data splits, training and evaluation.

## Implementation

| Module | Purpose |
|---|---|
| `plants.py` | Exact propagation of the normalized oscillator |
| `timing.py` and `records.py` | Virtual physical time and immutable controller-visible snapshots |
| `controllers.py` | PD, bounded-assumption predictor-PD, and analytical stability calculations |
| `signals.py` | Exogenous forcing independent of the controller schedule |
| `metrics.py` | Tracking, effort, variation, finite-horizon settling and explicit censoring |
| `imitation.py` | Shared bounded information, physical feature scaling and teacher demonstrations |
| `neural.py` | Causally masked transformer and matched-history MLP |

Validation covers analytical dynamics, delayed-feedback trajectories, information causality, event ordering, saturation, deterministic replay, reporting-grid independence and metric definitions. Neural checks additionally cover future-token and padding isolation, shared-information encoding, gradients and checkpoint round trips. Virtual computation time is independent of Python runtime. The fixed-cadence schedule assumes idealized parallel throughput; it is not a single-processor deployment claim.

The divergence guard is checked at event times rather than continuously. The predictor knows applied commands but not pending commands, future disturbances or future references. The measured PD pilot does not use that predictor. See the experiment contract for the full limitations.

## Next milestone

Define an operational competence threshold on separate development scenarios, then begin suppressive-pathway discovery and causal interventions with matched controls. Preserve the completed pilot as the baseline and reserve fresh confirmation scenarios for outcome-driven revisions. Direct closed-loop training and explicit E/I-inspired architectural comparisons remain subsequent work.

The primary research outcome remains the **stability–responsiveness tradeoff**, including disturbance recovery and delay tolerance at useful tracking performance. A reward increase, smaller actions, negative weights, or a static cancellation score alone would not establish the proposed mechanism.
