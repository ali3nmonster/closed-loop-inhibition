# First experiment contract: causal timing and classical validation

**Status: implementation milestone, 7 October 2026.** This contract implements the measurement groundwork in [START_HERE.md](START_HERE.md), before training a transformer or testing suppression. Its immediate claim is that the simulator faithfully measures delayed physical feedback. It supplies no evidence for transformer inhibition or E/I balance. Settings below are pilot choices, not optimized parameters or a confirmation preregistration.

## 1. Plant, units and controller

Use the normalized oscillator

\[
\dot q=v,\qquad \dot v=(-q-2\zeta\tau v+u+d)/\tau^2.
\]

Time is in seconds; position is in a fixed normalized position unit; velocity is position units per second. Applied command, reference and disturbance are in position-equivalent units, preserving unit static gain. Set `tau = 0.5 s`, `zeta = 0.15`. Propagate the linear plant exactly between input events. Reference and disturbance schedules are exogenous, piecewise constant and independent of controller cadence.

The initial controller is frozen PD:

\[
u_{\rm issued}=r+2(r-q)-\tau v.
\]

Here every input comes from the selected observation, including its captured reference; there is no access to current or future simulator state. The reference feedforward term permits zero steady-state error for a constant reference without a disturbance. Analytic checks use unsaturated commands. Operational pilot curves additionally use a symmetric command limit of 5. No per-delay gain tuning or choice of a best-performing gain is permitted in this milestone. A delay-aware teacher and learned policies remain subsequent work.

## 2. Tasks and frozen pilot timing

Run 12-second episodes with reporting and sensor-capture intervals of 0.01 s. Start with applied command zero and no pending jobs. This cold start is part of the protocol and can favor shorter delays in short episodes; distinguish startup transients from responses to later disturbances.

| Case | Initial state | Reference | Disturbance |
|---|---|---|---|
| Initial-condition recovery | `(q, v) = (1, 0)` | 0 | 0 |
| Reference step | `(0, 0)` | 0 before 2 s, then 1 | 0 |
| Disturbance pulse | `(0, 0)` | 0 | 1 on `[2, 2.25)` s, otherwise 0 |

The computation-duration grid is **0, 0.025, 0.05, 0.1, 0.2 and 0.4 s**, corresponding to duration/`tau` ratios 0, 0.05, 0.1, 0.2, 0.4 and 0.8. This is a pilot grid, not an asserted transition boundary. Sensor transport and actuator delays are separate configuration fields, initially zero. Nonzero values are exercised in causality checks. A seeded piecewise disturbance can be added as a separate reproducibility check; it must specify its seed, amplitudes and switching times and must not silently replace these cases.

## 3. Causal event and observation interface

Use deterministic virtual time, rounding scheduled timestamps to the nearest nanosecond. Wall-clock Python execution time is not the modeled computation duration. A policy computes its result from an immutable snapshot immediately in host execution, but that result becomes issuable only at the scheduled virtual completion. The plant evolves under its existing command throughout that interval.

At coincident timestamps, prioritize **signal changes, action application, observation capture, observation availability, inference completion, inference dispatch, then reporting**. Process newly created zero-delay consequences at the same timestamp before advancing time or reporting. They cannot alter an observation already captured: capture precedes a command created by that timestamp's completion or dispatch. Such a zero-delay command nevertheless applies before the final reported state/action pair. Plant state is continuous at command jumps; reported signals and commands use their values after the timestamp's events.

Each observation records capture and availability times, captured position, velocity, reference and physically applied command. Only observations whose availability time has arrived may enter a query. Copy the available history into each job; later arrivals cannot alter an in-flight query. The physical history window is 1 s, measured backward from the **newest available capture time**, with inclusive endpoints. Exclude older observations. Consequently, a delayed history may end before dispatch time; record both timestamps so its age remains visible. Applied-action records are immutable and timestamped; issued or pending commands must not masquerade as already applied commands.

Record captures, availability, snapshot selection, inference starts/completions, command issuance/application and expired history entries. Each call begins with a fresh controller episode; no mid-episode reset is implemented. Unselected observations remain available in history until expiry. The current applied-action history spans the full episode; bound it or match that extra information explicitly before comparing learned architectures. Preserve enough data and configuration to reconstruct policy inputs and replay the trajectory.

## 4. Two distinct dispatch schedules

**Serial latest-observation schedule:** allow one outstanding inference. Dispatch when idle with a newly available observation; while busy, collect arrivals. At completion, use the newest eligible observation for the next job. Do not repeatedly redispatch an unchanged observation at zero compute duration. Increasing computation duration changes both observation age and effective decision cadence.

**Fixed-cadence schedule:** allow snapshot dispatch every 0.01 s, independently of outstanding jobs, but require an observation newer than the previously selected one. Skip ticks without a new observation; never redispatch an unchanged observation. Each job completes after the configured duration. The pilot's matching sensor and dispatch intervals preserve a fixed decision cadence after startup. Slower sensing limits actual decisions; verify and report the realized cadence. This idealized concurrent schedule isolates latency when cadence is preserved. It assumes sufficient parallel throughput, uses a stateless history-window controller and makes no single-processor deployment claim.

For both, apply commands after the separately specified actuator delay and hold the last command between applications. Report observation age and actual decision/application intervals; capture cadence alone is not the fresh-feedback rate.

## 5. Scores, censoring and stability

Let `e = q - r`. On the uniform reporting grid, compute tracking ISE by trapezoidal integration of `e²`; RMSE is `sqrt(ISE / observed_duration)`. Report peak absolute tracking error, integral of applied `u²`, and total action variation as the sum of absolute **applied** command jumps, including the initial zero-to-command jump. Use event durations for action effort rather than inventing transitions between held commands. Keep reference-step overshoot distinct from peak absolute tracking error.

Settling time is measured from the last reference or disturbance change (or time zero for initial-condition recovery): the earliest reporting sample after which `abs(e) <= 0.02` at every remaining sample. If this condition is not met, report unset/censored. This is a finite-horizon, position-only measurement, not proof of asymptotic settling; the final sample alone gives only horizon-edge evidence. Report last-change time and remaining observation horizon.

State magnitudes exceeding `1e6` in the recorded coordinate units, or nonfinite state, trigger a numerical divergence guard checked at event times. This is not continuous threshold detection: a coarse event grid can miss an excursion between events, and extra reporting events can reveal it. Check censoring as well as score convergence when refining the reporting grid. Censor the episode and identify the reason. Do not fill missing samples with zeros or compare truncated integrals to complete-episode integrals as if horizons matched. This guard is not an operational task-success threshold. No task failure rate is claimed until a meaningful task criterion is frozen separately.

Sampling makes trajectory scores and sampled peaks approximations even with exact plant propagation. Check metric convergence on a finer reporting grid without changing controller timing. Analytical stability uses the appropriate sampled system: continuous-feedback roots are only a limiting check. Fixed-cadence, unsaturated integer-step delays permit an augmented linear transition; report its spectral radius separately from empirical boundedness. Do not transfer those roots to fractional-delay or serial schedules without constructing their actual transition maps.

## 6. Acceptance and next decision

Accept this milestone when analytical propagation and sampled-feedback predictions agree within recorded tolerances; timestamps demonstrate physical evolution during inference; future observations never enter a snapshot; replay is deterministic; distinct sensor, computation and actuator delays behave correctly; and metric convergence and censoring checks pass. Save configuration, machine-readable metrics, event traces and plots exposing one observation-to-action cycle and the pilot delay sweep.

Report measured results only after running these checks. Advance next to a delay-aware classical comparator and a small transformer pilot. Before confirmation, freeze the suppressive-pathway identification rule, intervention controls, task-failure criterion, independent data splits and a useful timing grid. The eventual primary hypothesis remains the held-out **intervention-by-temporal-demand interaction**, not superior PD performance or an oscillatory plot.
