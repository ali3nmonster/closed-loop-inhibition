# Small centered interventions in suppressive transformer heads

**Prospective protocol, 8 October 2026, Europe/Berlin.** This follow-up asks whether the disturbance-response effects of previously identified suppressive attention heads persist when the intervention is smaller and its mean command shift is removed on a fixed calibration distribution. The [first suppression pilot](../results/suppression_pilot/README.md) found reproducible functional suppression, but halving a selected head also caused autonomous drift, and controls matched poorly across timing conditions. Those findings motivate this new experiment; its confirmation scenarios are fresh.

The experiment changes three frozen transformer controllers. It tests a local functional role of selected pathways, not a newly trained E/I architecture. A preserved mean command on calibration histories does not establish a preserved equilibrium, unchanged feedback gain, or E/I balance. Calibration decisions are frozen before confirmation outcomes are evaluated.

## Frozen models and disturbance task

Use the existing transformers with training seeds `11`, `22`, and `33`, retaining their checkpoint hashes, encoding, and controller-visible information boundary. Reuse the candidate and alternative heads in the selection artifact committed at `bebc9bd`; verify that artifact against the committed version and record its hash. There is no head search or replacement in this experiment.

| Training seed | Candidate layer and head | Alternative layer and head |
|---:|---|---|
| 11 | 1, 1 | 1, 0 |
| 22 | 1, 1 | 1, 0 |
| 33 | 1, 3 | 1, 1 |

Indices are zero based. The alternatives were selected by residual contribution magnitude in the previous pilot. Some also met its functional suppression criterion; they are alternative pathways, not established nonsuppressive nulls.

The plant is the same normalized oscillator with time constant `0.5 s`, damping ratio `0.15`, and actuator limit `5`. Five-second episodes start at zero position and velocity and maintain zero reference. A constant disturbance lasts `0.25 s`, with onset drawn uniformly from the grid `1.00, 1.05, ..., 1.40 s`. Its magnitude is uniform on `[0.3, 0.6]`, with an independent equally probable positive or negative sign. Each disturbance is paired with a zero-disturbance sham having the same controller, intervention, initial state, reference, and timing.

Observation capture and fixed-cadence dispatch use `0.05 s` intervals. Sensor transport and actuator delay remain zero; computation durations are `0`, `0.05`, `0.1`, and `0.2 s`. Evaluate fixed-cadence and serial schedules separately. The reporting interval is `0.01 s`, and the observation history remains bounded to `0.5 s` and eleven tokens. Virtual computation delay determines environmental evolution; host inference runtime does not set that delay. Fixed-cadence execution assumes idealized parallel throughput. Serial execution changes both latency and decision cadence.

## Independent calibration and confirmation

Generate and save complete scenario specifications using these frozen streams:

| Stage | Scenarios | Generation seed | Scenario prefix |
|---|---:|---:|---|
| Calibration | 4 | 32452843 | `centered_calibration` |
| Confirmation | 8 | 49979687 | `centered_confirmation` |

Verify that the exogenous scenarios and generation streams differ from prior discovery, calibration, and confirmation data. Changing an identifier alone does not make a scenario independent. The same scenario is deliberately paired across model seeds, timing conditions, and variants. Its timing repetitions are not independent environmental draws.

Calibration uses native-controller pulse and sham histories to choose offsets and matched-control strengths. It also evaluates native and centered candidate controllers in their own closed loops. No closed-loop recovery outcome is an objective for choosing an offset, control strength, head, or threshold. Preserve unsuccessful matching, competence, drift, and direction checks rather than changing the design or dropping a model.

## Seven controller variants

Scale a selected head by multiplying its columns in the attention output-projection matrix, preserving the shared projection bias, attention computation, masks, and downstream operations. Construct each variant from the original checkpoint. A scale of one is an exact identity control.

| Variant | Head or output change | Constant command correction |
|---|---|---|
| Native | Original transformer | None |
| Raw weakened candidate | Candidate scale `0.9` | None |
| Raw strengthened candidate | Candidate scale `1.1` | None |
| Centered weakened candidate | Candidate scale `0.9` | Fitted on native calibration sham histories |
| Centered strengthened candidate | Candidate scale `1.1` | Fitted on native calibration sham histories |
| Centered matched alternative | Frozen alternative head at a calibrated scale | Fitted on native calibration sham histories |
| Centered matched gain | Native command with a calibrated gain | Fitted on native calibration sham histories |

All final commands obey the same actuator limit. Raw and centered candidate arms separate the effects of head scaling from those of its additive command correction. The strengthened arm checks the opposite local perturbation descriptively; nonlinear response costs need not reverse exactly when the sign of the weight change reverses.

## What command centering preserves

Collect paired native histories at dispatch times in the inclusive window from disturbance onset through onset plus `2 s`. The sham has no physical disturbance, but its calibration window uses the paired scenario's onset. This window is the same existing probe window used for immediate-effect measurement.

For a fixed model, timing condition, and variant, let `z_v(H)` be its physical command before final clipping and let `u_0(H)` be the physically clipped native command. Let `C(x) = clip(x, -5, 5)`. Fit one scalar offset `b_v` such that

\[
\frac{1}{N}\sum_{s=1}^{N}\frac{1}{n_s}\sum_{t\in W_s}
C\!\left(z_v(H_{s,0}(t))+b_v\right)
=
\frac{1}{N}\sum_{s=1}^{N}\frac{1}{n_s}\sum_{t\in W_s}
u_0(H_{s,0}(t)).
\]

Here `H_s,0` is a native sham history, `W_s` is its probe window, and `n_s` counts its dispatches. Each scenario receives equal weight, and dispatches within a scenario receive equal weight. Use deterministic bisection with absolute mean-residual tolerance `1e-10`, retaining the fitted offset and achieved residual. Apply the offset before the final common clip, so clipping is included in the calibration equation.

The offset is constant throughout a rollout and is specific to the model, timing condition, and variant. It receives no future disturbance, pulse onset, future state, or native trajectory during confirmation. It is not updated online. The centered policy acts on its own evolving observations.

This procedure preserves the **mean clipped command on the specified native calibration sham histories**. It does not preserve the mean on pulse histories, the full-episode command mean, the time average of held actuator commands, or the mean on the centered controller's own trajectory. Equal dispatch weighting is not elapsed-time weighting. Since the sham dynamics are deterministic for each model and timing condition, repeated calibration scenarios do not provide independent sham trajectories; their windows can differ. Report actual closed-loop drift and held-action means independently of the fitted residual.

## Controls matched within each timing condition

For each model and timing condition, first fit the centered weakened candidate. Its immediate-effect target is the RMS change from native clipped commands on fixed native pulse and sham histories. Give pulse and sham equal weight within each scenario and scenarios equal weight:

\[
M_v^2=\frac{1}{N}\sum_s\frac{1}{2n_s}\sum_{t\in W_s}
\left[(u_v(H_{s,p}(t))-u_0(H_{s,p}(t)))^2
+(u_v(H_{s,0}(t))-u_0(H_{s,0}(t)))^2\right].
\]

Match each control to `M_centered weakened` separately for every model × schedule × computation-duration cell. This replaces the previous pilot's pooled matching, which concealed large cell-specific mismatches. Matching uses only immediate command effects on calibration histories, never confirmation behavior or a favorable recovery score.

For the alternative head, search scales `0.500, 0.505, ..., 1.000`. Refit the sham offset for each scale before measuring its RMS effect. Select the scale with the smallest absolute discrepancy from the target, resolving exact ties deterministically in ascending parameter order. The head's identity remains fixed.

For the gain control, search the prospectively chosen amplification branch `g = 1.000, 1.001, ..., 1.500`. Let `mu_0` be the equally weighted native sham command mean in that cell. Before final clipping, the gain control is

\[
z_g(H)=\mu_0+g\,[z_0(H)-\mu_0].
\]

Here `z_0(H)` is the native physical command before actuator clipping. Apply the gain transformation and fitted offset before the one final common clip, consistently in calibration and deployment. Fit an additional scalar offset using the same mean-preservation equation, including clipping, for every gain. Choose the gain with the smallest RMS discrepancy, with the same tie rule. Centering around `mu_0` expresses an amplification of deviations from the calibrated mean; the extra offset accounts for any mean change introduced by clipping.

A control matches only when its RMS discrepancy is at most `5%` of the candidate target. Record the target, achieved effect, relative discrepancy, parameter, offset, and match status for every cell. If the target is below `1e-8`, mark the ratio and matching interpretation unavailable, retain an identity control, and do not inflate the target with a replacement denominator. Do not extend either search range after observing data. A failed match remains visible and withholds the corresponding matched-control claim.

RMS matching controls the magnitude of the immediate intervention, not its sign, waveform, state dependence, effort, or full feedback law. The alternative head can itself be suppressive. A result that differs from these controls excludes only these particular explanations.

## Checks before confirmation

Assess the candidate's small-step functional direction on independent calibration histories at the previous discovery timing: fixed-cadence dispatch with `0.05 s` computation duration. Compute the amplitude-normalized RMS pulse-minus-sham command response as in the [first protocol](SUPPRESSION_PILOT.md). Centered weakening replicates the direction only if its median fractional increase over native is positive and at least `75%` of scenarios show a positive increase. The native normalized response must exceed `1e-8`; otherwise its ratio is unavailable. Do not impose the previous `5%` discovery threshold on a perturbation one fifth as large. Record raw and centered responses, and retain the same candidate regardless of outcome.

Native competence retains the previous bounds in every model × timing cell: every pulse and sham must complete; peak absolute position must stay strictly below `1`; final-second absolute position RMSE must be at most `0.05`; and mean normalized response error must be at most `1.5` times the shared-information predictor-PD teacher on the same scenarios. The teacher has explicit plant knowledge; its role is a control reference, not an identical learned prior.

Run both centered candidate variants in their own calibration closed loops as well. Record their completion and absolute competence. A separate secondary drift criterion passes only when the mean across scenarios of `sham RMSE_variant - sham RMSE_native` is at most `0.005` in that model × timing cell. This absolute tolerance avoids unstable ratios to a nearly zero native sham error. It is an operational check, not evidence of exact mean or equilibrium preservation. Report the same diagnostic for confirmation variants, with no retuning or selective omission.

Failure of a native competence, directional replication, drift, or matching check limits the corresponding interpretation. All frozen confirmation variants and failures remain reportable; failed checks do not authorize replacing a head or selecting only favorable delays. Save readiness diagnostics and freeze configuration, source fingerprints, checkpoint identities, scenario specifications, and all calibration decisions before advancing to confirmation.

## Closed-loop endpoints and temporal contrast

Every confirmation controller receives observations from its own trajectory. Pair its pulse rollout with its own intervened sham. If `D` is signed disturbance amplitude and `t_end` is pulse offset, the primary response endpoint is

\[
J_v=\frac{1}{D^2}\int_{t_{\mathrm{end}}}^{t_{\mathrm{end}}+3\,\mathrm{s}}
\left[q_{v,p}(t)-q_{v,0}(t)\right]^2\,dt.
\]

Use the existing reporting-grid trapezoidal calculation, interpolating position at exact window boundaries before squaring. The three-second recovery window fits within every episode. With the declared plant units, `J` has units of seconds. It measures incremental disturbance recovery, not absolute tracking performance or asymptotic stability.

For each model and scenario, define `Delta J(d) = J_centered weakened(d) - J_native(d)`. The primary temporal contrast is

\[
I=\operatorname{mean}_{\mathrm{scenario}}
\left[\Delta J(0.2\,\mathrm{s})-\Delta J(0)\right]
\]

under **fixed-cadence** inference. A positive value means that weakening raises the incremental recovery cost more, or lowers it less, at the longer delay. Its sign alone does not show that weakening is harmful in either condition or that suppression benefits overall control.

Compare the same temporal contrast for the raw weakened candidate, centered matched alternative, and centered matched gain. Preserve scenario pairing in candidate-minus-control contrasts. Strengthening, intermediate delays, and every serial-schedule contrast are secondary descriptive results. Keep serial and fixed-cadence results separate because the serial comparison also changes command cadence.

Report per-model results and all paired scenario values, together with absolute pulse and sham tracking errors, peak errors, final-second recovery, action effort, action variation, signed time-average held action, clipping, numerical completion, and competence. Own-sham drift cannot be hidden by subtracting it from the pulse. Compute the held-action mean using applied-action durations, including initial and final hold intervals, rather than averaging commands or reporting samples. Censored or incomplete pairs have unavailable full-window response scores and explicit reasons; partial integrals must not be compared with complete scores.

Offsets and matched-control parameters vary with timing by design. Accordingly, the primary comparison concerns a family of controllers calibrated for each timing condition. It does not hold every component of the modified policy identical while changing only delay. Raw arms expose the contribution of this calibration, while the fixed head-scaling magnitude and within-cell controls constrain its interpretation.

## Run budget and numerical validation

The planned simulation counts are:

| Purpose | Rollouts |
|---|---:|
| Native pulse and sham probe collection for calibration | 192 |
| Calibration closed loops for native, both centered candidates, and teacher | 640 |
| Confirmation for seven transformer variants | 2688 |
| Confirmation teacher reference | 128 |
| Total before numerical checks | 3648 |

These counts include repeated deterministic shams as recorded executions, not independent experiments. Fixed-history evaluation of offsets and control grids adds model predictions but no new physical rollouts. Reserve the remaining approximate 4000-rollout budget for a representative reporting-grid convergence check and implementation checks, and report their count separately.

Before final interpretation, check representative native and centered candidate cases at half the reporting interval. Compare `J`, completion, and applied actions and their timestamps. Test the actual centering equation after clipping, sample/group weighting, gain transformation, identity cases, unavailable matching targets, causal history handling, same-intervention sham pairing, held-action integration, and stage-integrity guards. Deterministic restoration to the original checkpoint is an implementation check rather than independent evidence of biological rescue.

## Interpretation and retained artifacts

Save configuration and stage manifests; source, checkpoint, inherited-selection, scenario, and calibration fingerprints; all control-grid diagnostics; mean-preservation residuals; direction and competence checks; paired and cell-level outcomes; temporal contrasts; and compact figures. Keep confirmation independent of parameter choice. Changes motivated by confirmation results require a new experiment version and fresh confirmation data.

An informative outcome can be persistent functional suppression with little drift, a recovery change explained by generic gain, a response change shared by another suppressive head, residual drift despite calibration centering, or no consistent temporal interaction. Improved incremental recovery accompanied by greater effort or degraded absolute behavior is a tradeoff. Three frozen imitation-trained models, one fully observed linear plant, and eight disturbance scenarios do not establish a general transformer mechanism, emergence driven by fast-loop training, or an advantage of explicit E/I architecture. This experiment narrows those next questions by separating small pathway effects from an observable command-bias confound.
