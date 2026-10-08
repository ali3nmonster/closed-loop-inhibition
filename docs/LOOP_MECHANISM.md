# Complete-loop dynamics and gain rescue

This is a targeted, exploratory follow-up to the [completed delay sweep](../results/delay_sweep/README.md). Its population and endpoints are chosen from that result, before new calibration or confirmation. It is a mechanistic test on existing models, not independent training replication. The historical pooled absolute interaction was negative; the positive 200 ms plant subgroup must not replace that broader result.

## Question and fixed scope

For the 200 ms plant, weakening the previously selected suppressive branches improved recovery at zero latency and worsened it at 100 ms latency. Scalar gain also produced a crossover. Does the complete feedback loop show corresponding changes in local modes and disturbance responses? How much of the intervention is explained by ordinary output gain, and how much remains after compensating it?

Use all 12 inherited selected checkpoints at plant tau 0.2 s: three initialization seeds crossed with four training-noise correlation times. These are three initialization seeds, not twelve independent initialization replicates. No retraining or pathway discovery occurs. Test physical delays 0, 25, 50, 75 and 100 ms at the fixed 50 ms observation and decision cadence. Keep the explicit delay input at its training value, 50 ms, as in the previous sweep. All realized histories and applied actions remain truthful. Delays above the cadence use the inherited idealized command pipeline.

The [machine-readable configuration](../configs/loop_mechanism.json) is authoritative. The runner binds the selected checkpoints and intervention groups to the original training and collective-suppression manifests and verifies the completed delay experiment against commit `2c6177e`. Earlier scientific sources, configurations and results remain unchanged.

## Interventions and independent calibration

Keep the inherited `native`, `joint_weak` (selected residual branches multiplied by 0.9), and `joint_strong` (1.1) policies, including their previously calibrated offsets. Add two controls fitted separately for every model and physical delay:

1. `gain_matched`: unchanged native branches with a positive scalar output gain fitted to reproduce the weak policy's pulse-minus-sham command waveform on the same native histories.
2. `weak_rescue`: weakened branches with a positive scalar output gain fitted to reproduce the native pulse-minus-sham command waveform on those histories.

Generate calibration histories by running the native policy in its own physical loop on two fresh OU tapes, each with a sham and both pulse signs. Freeze all 60 preparations before starting confirmation. Calibration streams are 5010001–5010002; confirmation streams are 5030001–5030004. Development uses different streams and separate output directories.

Fit the gain by equal-probe-weight least squares to the entire post-pulse command-response waveform, constrained to [0.25, 4]. Fit the constant offset separately to match the native clipped sham-command mean on calibration histories. The policy remains `clip(center + gain * (raw - center) + offset)`. Report the gain bound status, response energy, waveform residual, cosine similarity, mean mismatch and clipping. A successful scalar fit is not presumed. Report unavailable controls explicitly if calibration is censored or degenerate, retaining the inherited variants.

The fit targets paired command responses on common histories, not closed-loop recovery performance. It cannot guarantee preservation of the actual equilibrium or mean on the control's own trajectory. The rescue changes both gain and offset; attribution to gain alone requires the recorded mean and equilibrium checks. Calibration uses native histories, so confirmation also measures matching drift and waveform residuals on independent native histories. No controls are retuned or excluded based on confirmation quality.

## Physical confirmation and primary comparisons

Use the original OU force process (standard deviation 0.02 and each model's original noise correlation time), four-second trials, pulse onset at one second, pulse width 0.1 s, and amplitudes ±0.02. Share the exact disturbance tape among policies and between each pulse and its sham. Reuse one physical sham for both signs. Score the interval [1, 4] s. Include a passive zero-command reference on the same new tapes, once for each of the four noise conditions.

The primary performance measure is the mean amplitude-normalized integrated squared pulse-minus-own-sham position,

`E_v(L) = mean_integral((q_v,pulse - q_v,sham)^2 / amplitude^2)`.

This is a position-error measure, not mechanical energy. For each model report the absolute effect `D_v(L) = E_v(L) - E_native(L)` and the predeclared endpoint interaction `I_v = D_v(0.1) - D_v(0)`. The principal new contrast is `I_weak_rescue` versus `I_joint_weak`; also report residual effects at each endpoint and `I_gain_matched` versus `I_joint_weak`. A reduced rescue interaction is descriptive evidence for a gain/mean explanation, not proof of complete mediation. Show all twelve model values, averages within initialization seed, and percentage effects as secondary measures. Do not divide by near-zero denominators or pool signed ratios as evidence of complete rescue.

The truthful current-action-age feature differs across schedule phases: it is 50 ms at zero latency, 25 ms at latencies 25/75 ms, and zero at latencies 50/100 ms. Development verification identified different equilibrium positions across these phases even with the planned-delay cue fixed. Also report 25→75 ms and 50→100 ms as secondary comparisons with the same action-age phase. They help separate added command latency from this direct timing input for the three inherited policies. They do not replace the primary endpoints; calibrated controls have separately fitted settings at each delay.

Retain sham accuracy, action effort, task failures, censoring, and native-to-passive ratios. A completed trajectory above the inherited RMS failure threshold retains numeric error metrics; a censored trajectory has null full-window metrics. Aggregates requiring missing observations remain unavailable instead of averaging only survivors. All conditional control comparisons state their denominator.

## Complete-loop state and linearization

Construct a repeating 50 ms dispatch-to-dispatch map after the observation history has filled. Its state contains eleven chronological observations (position, velocity and captured applied action), the current held command, and commands pending application. The latest observation supplies the plant state. Fixed reference is zero. Timestamp and action-age features are fixed by each delay's phase within the schedule. The explicit delay feature remains 50 ms.

Propagate the plant exactly between command-delivery events. Respect capture-before-completion ordering at equal timestamps: the captured action can differ from the current action available at dispatch. Fractional delays split a cycle into two held-input intervals. This augmented map, including history shifts and the pending-command queue, is the object differentiated. A residual-layer or immediate-readout Jacobian is not substituted for it.

Analyze each policy at its own disturbance-free equilibrium. Record the root residual, equilibrium position and command, clipping, and whether the linearization is valid. Use the trained float32 weights represented in double precision for differentiation; validate the resulting map against the original float32 event-driven policy. Root convergence alone does not prove uniqueness, global attraction or nonlinear stability.

For a valid equilibrium record the complete Jacobian, poles, spectral radius, dominant modal decay/growth rate `-log(abs(pole))/period`, and oscillation frequency `abs(arg(pole))/(2*pi*period)`. A spectral radius below one indicates local asymptotic stability of this fixed-phase map. Report transient amplification only in the declared coordinates: position/state_scale, tau*velocity/state_scale, and command/output_scale. Euclidean transient amplification depends on these scales and on the history-state representation.

Compute the linear disturbance-to-position response of the complete closed loop, with an external disturbance held over each decision interval. Its frequency response is that discrete input/output system, not a continuously varying sinusoidal-force experiment and not the previous fixed-teacher-history assay. The configuration lists 25 frequencies from 0.1 to 9 Hz, concentrated around 1 Hz and below the 10 Hz Nyquist frequency. Phase and magnitude concern the closed loop; they are not open-loop gain/phase margins.

## Transients and validation

Starting at each equilibrium, apply a 100 ms square force pulse with amplitudes ±0.0001 and ±0.02. Follow the complete nonlinear map for twelve seconds and compare with its linear prediction. The small pulse tests local validity; the larger pulse tests finite-amplitude deviation. Report sampled position traces, error integrals, peak response, sign crossings and settling within 2% of each response's peak for the remainder of the finite horizon. An unsettled response is explicitly marked. These map trajectories have mature equilibrium histories and no OU force; they are not the noisy, zero-initialized four-second confirmation trials.

An additional ±0.000001 pulse pair checks the numerical linearization over the first 0.5 seconds, with a declared 1% relative prediction-error tolerance. Retain the full twelve-second mismatch as well. Record deviations explicitly; growth into a nonlinear regime around an unstable equilibrium is not evidence that the derivative itself is wrong.

Validate every policy/delay map with the event-driven simulator using a separate deterministic off-equilibrium pulse trajectory. Compare the next mature state and current command, including queue contents, across the same physical schedule. Record physical validation rollout counts separately from map iterations. Additional tests check classical-controller limits, finite-difference derivatives and small-perturbation prediction where applicable. Numerical tolerance failures stop publication rather than becoming missing favorable results.

## Interpretation boundaries

Agreement between the recovery crossover, modal damping changes and gain rescue would support an ordinary feedback-gain explanation of suppressive functionality. Persistence after gain/offset compensation would motivate examining the remaining history-dependent and nonlinear response; it would not by itself establish a unique E/I algorithm or biological E/I balance. Local equilibrium analysis does not establish stability along noisy trajectories or throughout the nonlinear state space. Independent training replication, direct closed-loop learning, other controller architectures and explicit inhibitory mechanisms remain subsequent experiments.

The sampled-state formulation follows standard feedback-system analysis; see [Åström and Murray, Feedback Systems](https://fbsbook.org/). The implementation and its simulator-parity checks, rather than an assumed continuous-time approximation, define the actual experiment.
