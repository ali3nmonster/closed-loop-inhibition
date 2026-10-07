# Suppressive organization in transformer feedback control

Research project investigating whether emergent suppression and explicit excitation–inhibition inspired mechanisms improve the stability and responsiveness of transformer controllers when the environment evolves during computation.

**Central question:** Do suppressive mechanisms change the combined controller–environment dynamics beneficially, and does their causal contribution depend on the relationship between environmental timescales, observation-to-action latency, and fresh-feedback intervals?

**Status on 7 October 2026:** The first simulator and classical-control validation milestone is implemented. An exact continuous-time plant, two inference schedules, timestamped policy snapshots, PD and causal predictor-PD controllers, and explicit metrics are available. The deterministic PD pilot contains 36 rollouts; no transformer has been trained and no inhibitory mechanism has been tested.

## First results

Read the [baseline report and plots](results/baseline/README.md). The simulator reproduces analytical sampled-feedback trajectories, including a delay-induced unstable case. The pilot separates fixed-cadence delayed delivery from serial inference, where longer computation also reduces decision frequency.

![Delay and decision cadence in the classical pilot](results/baseline/delay_sweep.png)

These are frozen classical-controller checks with prescribed signals and gains. They validate the experimental platform; they are not evidence for the proposed transformer E/I mechanism.

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

## Read first

1. [Research background and proposal](docs/RESEARCH_PROPOSAL.md) — the complete conceptual argument, evidence, definitions, hypotheses, and intended contribution.
2. [Step by step starting plan](docs/START_HERE.md) — the first experiment, milestones, controls, measurements, and decision criteria.
3. [Neuroscience evidence](docs/notes/neuroscience_evidence.md) — cortical computation, population geometry, inhibition, and embodied dynamics.
4. [Transformer evidence](docs/notes/transformer_evidence.md) — recurrence, geometry, native suppression, explicit mechanisms, and real-time control.
5. [Source and provenance guide](docs/SOURCES.md) — primary references, version cautions, and connections to existing local research.
6. [Implemented experiment contract](docs/EXPERIMENT_CONTRACT.md) — timing conventions, information access, scoring, censoring, and the current pilot's limits.

## Implementation

| Module | Purpose |
|---|---|
| `plants.py` | Exact propagation of the normalized oscillator |
| `timing.py` and `records.py` | Virtual physical time and immutable controller-visible snapshots |
| `controllers.py` | PD, bounded-assumption predictor-PD, and analytical stability calculations |
| `signals.py` | Exogenous forcing independent of the controller schedule |
| `metrics.py` | Tracking, effort, variation, finite-horizon settling and explicit censoring |

The 58 validation tests cover analytical dynamics, delayed-feedback trajectories, information causality, event ordering, saturation, deterministic replay, reporting-grid independence and metric definitions. Virtual computation time is independent of Python runtime. The fixed-cadence schedule assumes idealized parallel throughput; it is not a single-processor deployment claim.

The divergence guard is checked at event times rather than continuously. The predictor knows applied commands but not pending commands, future disturbances or future references. The measured PD pilot does not use that predictor. See the experiment contract for the full limitations.

## Next milestone

Build a small causal transformer imitation pilot and compare it with an equally informed teacher and simple neural baseline. Preserve timestamp and information boundaries, and match action-history access before architectural comparisons. Begin suppressive-pathway discovery only after competent held-out control is established.

The primary research outcome remains the **stability–responsiveness tradeoff**, including disturbance recovery and delay tolerance at useful tracking performance. A reward increase, smaller actions, negative weights, or a static cancellation score alone would not establish the proposed mechanism.
