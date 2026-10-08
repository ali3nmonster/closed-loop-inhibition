# Validating plant speed, feedback dynamics and structured perturbations

**Prospective protocol, 8 October 2026.** This milestone tests the physical and temporal properties needed for the research question, before training faster-loop transformer controllers. The earlier suppression studies used one plant speed and one pulse duration. Here we vary plant speed, feedback delay and perturbation correlation time separately. The [configuration](../configs/dynamics_validation.json) fixes the grids and acceptance criteria before simulation outcomes are inspected.

The intended properties are independently adjustable physical response times and forcing timescales; correct causal delayed feedback; measurable frequency-dependent responses; and a grid containing both stable and unstable feedback regimes. A stable passive plant can become unstable under delayed active control. This provides a test environment, not evidence about neural inhibition.

## System and information boundary

Use the existing exact normalized oscillator:

\[
\tau^2\ddot q+2\zeta\tau\dot q+q=u+d.
\]

Hold damping ratio `zeta = 0.15` and static force-to-position gain fixed. Sweep `tau = 0.5, 0.2, 0.1, 0.05 s`. The natural angular frequency is `1/tau`, the damped period is `2*pi*tau/sqrt(1-zeta^2)`, and the amplitude-envelope decay time is `tau/zeta`. These are distinct quantities.

Classical PD issues `u = -2*q - tau*v` at zero reference from the latest captured observation. Its dimensionless gains remain fixed across speeds. Sensor capture and command dispatch both have period `h = 0.05 s`. Use fixed-cadence dispatch and delays `L = m*h` with `m = 0, 1, 2, 4`, assuming idealized parallel inference throughput. No sensor transport or additional actuator delay is added. The plant moves during virtual computation, and commands are held between deliveries. Initial commands and the pending queue are zero. This milestone does not test the serial schedule.

The ratio `h/tau` spans `0.1–1`, and `L/tau` spans `0–4`. Compare the actual sampled system with its augmented delayed transition, rather than interpreting continuous-feedback poles as delayed-system poles. The controller here is a transparent reference, not a transformer. No neural checkpoint is evaluated on an unfamiliar faster plant and called competent.

Use unsaturated commands for the linear validation. A numerical state guard remains active; its crossing is distinct from asymptotic instability. Record peak commands so later bounded-actuator studies can set and check competence separately. Finite-window completion does not prove stability, and a finite guard crossing must not be turned into a complete-horizon performance score.

## Independent exogenous clocks and measurement corruption

Force waveforms are generated on a `2.5 ms` grid and held between changes. This clock is independent of controller capture and dispatch. The physical plant propagates exactly between input events. A sampled-and-held waveform is the actual specified input; it is not an exact continuously driven stochastic differential equation.

For colored inputs, sample the stationary Ornstein–Uhlenbeck process at the forcing grid by its exact AR(1) transition:

\[
d_0\sim\mathcal N(0,\sigma^2),\qquad
d_{j+1}=a\,d_j+\sigma\sqrt{1-a^2}\,\epsilon_j,
\quad a=\exp(-\Delta/t_c),\quad \epsilon_j\sim\mathcal N(0,1).
\]

Use `sigma = 0.02`, correlation times `t_c = 0.01, 0.05, 0.2, 1 s`, and seeds `101, 202, 303`. The same seed and correlation-time tape is reused across plants, controllers and injection channels. Each trace remains uncentered and unscaled: RMS is matched in expectation, and realized mean/RMS are reported. Different correlation times may share random-number innovations by seed; these comparisons are paired, not independent noise draws. Slow-noise finite records need not have their target RMS exactly.

Two injection channels remain separate:

- **Force disturbance:** the waveform enters `d` in the plant equation; observations are exact.
- **Position measurement noise:** add the waveform evaluated at each observation's capture time to its reported position. Velocity measurements remain exact. The physical force disturbance is zero. Repeated access to a historical observation returns the same corrupted value. Controller-visible timestamps and applied actions are preserved, and future noise values never enter a snapshot.

Store true plant measurements separately from the known noise tapes. The noise wrapper changes controller inputs, not the physical position or reporting trace. This explicit position-only sensor model does not claim a realistic joint position/velocity sensor process.

## A. Passive ringdown

For each plant, release it from `(q,v) = (0.01,0)` with zero control and zero disturbance. Run for eight passive envelope-decay times, reporting at `tau/100`. Compare the simulator with an independently implemented scalar underdamped solution at every recorded time, using `(q,tau*v)` to avoid unit-dependent coordinate scaling. The maximum trajectory discrepancy divided by the maximum reference state norm must be at most `1e-8`.

Estimate the oscillation period from successive positive position peaks and the envelope decay from a linear fit to their log amplitudes. Relative errors against the known values must each be at most `1%`. Record the measured physical times and normalized-time curves for all four speeds.

## B. Delayed feedback modes and trajectory validation

For all 16 plant × delay cells, construct the exact sampled transition for `u_k = -K*x_(k-m)`, with `K = [2,tau]`. Report its eigenvalues, spectral radius `rho`, and dominant continuous-equivalent growth rate `log(rho)/h`. The state includes the required delayed state history. Eigenvalue classifications apply to unsaturated linear dynamics around zero reference.

Run the event-driven simulator for `10 s` from `(q,v) = (1e-8,0)`, with a state guard of `1e4` and reporting at `h`. Independently reconstruct the same cold-start delayed commands and state sequence using a discrete recurrence. Compare every recorded pre-censoring sample, with maximum normalized relative trajectory error at most `1e-8`. Retain all unstable and censored cases.

Fit log state-norm growth over the latter half of usable recorded samples. Its sign must agree with the analytic dominant rate when that rate's magnitude exceeds `0.1 /s`; report the fitted value without asserting precise pole estimation from a short oscillatory trace. Include at least one analytically stable and one unstable grid cell, and confirm a delay-driven crossing at fixed plant parameters. The exact trajectory comparison is the stricter quantitative test.

## C. Frequency response

Check the four passive plants and four predeclared stable PD cases: `(tau,m) = (0.5,0), (0.5,4), (0.1,0), (0.05,0)`. If an anchor is analytically unstable, retain that finding and withhold steady-response interpretation rather than replacing it.

Drive each case with a held sinusoidal force of amplitude `0.01` at `omega*tau = 0.5, 1, 2`. Warm up for eight dominant envelope times, then measure for eight input periods, rounding the episode boundary to the controller grid. Fit sine, cosine and constant components to sampled position after warmup. Compare the measured complex response, including phase, with the analytical discrete response using the actual `2.5 ms` input holds and `50 ms` observation/control grid. The complex relative discrepancy must be at most `0.5%`.

The forcing hold is included in the theoretical response; it must not be mistaken for an ideal continuous sine. The sampled transfer response includes feedback delays. These finite frequency checks do not constitute a complete robustness or stability-margin measurement.

## D. Input-statistics validation

For each correlation time and seed, generate a separate `300 s` stationary input tape on the same grid, without a plant simulation. Report sample mean, RMS and correlation at lag nearest to one nominal correlation time. Pool the three traces for each correlation time, accounting for within-trace lag pairs only. The pooled RMS relative error must be at most `10%`; absolute correlation error against `a^lag` must be at most `0.1`.

These finite-sample checks complement exact seeded-recurrence tests. Keep all tapes unnormalized. They validate the input generator, not the controller's dynamics.

## E. Colored-input responses and injection channels

Run every stable PD anchor × four correlation times × two injection channels × three seeds: 96 rollouts. Also run the four passive plants under force noise at every correlation time and seed: 48 rollouts. Episodes last `60 s`; discard the first `20 s` only from the declared measurement window, retaining the complete simulation. Report at `0.01 s` and score true physical position and applied command. This is a finite-window response comparison; slow closed-loop modes and finite records can affect variance estimates.

Report physical position mean and RMS, command RMS, position temporal correlation, and actual input mean and RMS. Aggregate with the range across seeds, without treating time samples as independent experiments. For the passive plants, compare measured statistics descriptively with exact stationary covariance predictions for the augmented sampled-and-held plant/OU process. Finite-sample agreement here is reported rather than used to tune signal parameters.

For the first seed in each PD anchor, correlation time and injection channel, additionally compare captured physical states against the independent fine-grid recurrence, including the same held force and capture-time measurement noise. Require maximum normalized relative trajectory error at most `1e-8`. Record completion and command/state peaks for every run. No actuator clipping is enabled; no claim of bounded-actuator competence follows from this stage.

## Validation integrity and next decision

Test the signal transition, stationary initialization, replay, clock independence, capture-time sensor corruption, scalar ringdown, delayed recurrence, frequency response and covariance calculation before running the full protocol. Freeze source, configuration and protocol fingerprints at launch and verify them after all stages. Save failures alongside successes, figures, full per-condition summaries and representative traces. Large tapes/traces belong under ignored `runs/`; compact results belong under `results/dynamics_validation/`.

The main simulation total is 188 rollouts: 4 passive ringdowns, 16 delayed-mode cases, 24 sinusoidal response cases and 144 colored-input cases. Input-quality tape generation and independent algebraic/reference calculations are not additional physical rollouts.

Four predeclared supplemental rollouts repeat PD anchors `(tau,m) = (0.05,0)` and `(0.5,4)`, both noise channels, `t_c = 0.01 s` and seed `101`, with reporting refined from `0.01` to `0.005 s`. Keep physical inputs, controller timing, duration and measurement window fixed. Compare shared-time states and commands, position RMS and exact held-action RMS against the main runs. Position RMS must change by at most `0.5%`, and applied-command behavior must agree to floating-point tolerance. Record these results separately; the combined planned total is **192 rollouts**. This tests reporting accuracy for a fast plant and a slowly damped delayed loop, without changing the actual forcing hold model.

Accept the setup for subsequent controller experiments only if the predeclared analytical, input-quality and causal checks pass and the grid demonstrates the intended range of temporal demands. Use observed failures to diagnose the setup explicitly, not to discard unfavorable cells. A later neural experiment must establish a competent baseline across the chosen plant/timing distribution, account for the controller's learned temporal response, and reserve fresh perturbations for confirmation. This milestone makes no claim that suppression or an E/I architecture benefits fast feedback.
