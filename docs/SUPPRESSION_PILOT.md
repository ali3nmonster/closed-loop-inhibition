# First causal suppression pilot

**Status: protocol frozen before discovery and confirmation, 7 October 2026.** This experiment follows the [completed transformer imitation pilot](../results/transformer_pilot/README.md) and implements the first functional-role study in [START_HERE.md](START_HERE.md). Its question is whether an identifiable attention-head contribution suppresses disturbance-evoked action responses and whether weakening that contribution changes the coupled controller–plant response differently at short and long computation delays.

This is an exploratory mechanism pilot with prospectively frozen rules, not a formal external preregistration. It uses three existing frozen transformers, not newly trained models. It cannot establish emergence caused by fast-loop training, general architectural superiority, inhibitory cell types, or E/I balance. The explicit E/I-inspired architecture comparison remains later work.

## 1. Frozen models and physical task

Use the saved two-block, four-head-per-block transformers with training seeds `11`, `22` and `33`. Preserve checkpoint hashes, architecture, feature scaling and the shared causal information boundary from [TRANSFORMER_PILOT.md](TRANSFORMER_PILOT.md). No weights are optimized. Each policy call receives only its immutable available history and known action information; pending commands and future disturbances remain unavailable.

The plant remains the normalized oscillator with `tau = 0.5 s`, damping ratio `0.15`, and applied-action limit `5`. Episodes last `5 s`, start at position and velocity zero, and have reference zero throughout. Each disturbance scenario specifies:

- An onset sampled uniformly from the grid `1.00, 1.05, ..., 1.40 s`.
- A magnitude sampled uniformly from `[0.3, 0.6]`, with an independent equally probable positive or negative sign.
- A constant pulse lasting `0.25 s`, followed by zero disturbance.

Every pulse rollout has a paired zero-disturbance rollout with identical initial state, timing, controller and intervention. These pairs distinguish disturbance-evoked behavior from intervention-induced baseline drift. Use the same disturbance realization across timing conditions and interventions. No reference changes or randomized initial states are added to this pilot.

Sensor capture and fixed-cadence dispatch remain `0.05 s`; sensor transport and actuator delay are zero. The computation-duration grid is `0`, `0.05`, `0.1` and `0.2 s`. Evaluate both fixed-cadence and serial schedules. Closed-loop reporting uses a `0.01 s` grid. The plant evolves during each prescribed virtual computation interval; measured host runtime does not set that interval. Fixed-cadence inference assumes idealized parallel throughput, while serial inference also changes decision cadence.

## 2. Independent stages and frozen competence criteria

All scenario streams are new and disjoint from the preceding imitation experiment. Preserve explicit scenario specifications and audit split membership.

| Stage | Scenarios | Generation seed | Use |
|---|---:|---:|---|
| Discovery | 8 | 196613 | Identify one candidate head per transformer and its alternative head |
| Calibration | 4 | 216091 | Check competence, replicate the suppression direction, and match control strengths |
| Confirmation | 8 | 262147 | Evaluate frozen interventions and the temporal interaction |

Discovery uses only fixed-cadence inference at `0.05 s` computation duration. Calibration and confirmation cover both schedules and all four delays. Calibration uses the native controller's pulse/sham histories to assess immediate counterfactual action effects; control strengths are never selected by maximizing a closed-loop outcome.

Before examining confirmation outcomes, evaluate native transformer competence on calibration scenarios in every seed × schedule × delay condition. Require:

1. Every native pulse and sham rollout reaches its full horizon.
2. Every such rollout has peak absolute position strictly below `1`.
3. Every such rollout has absolute position RMSE over its final second at most `0.05`.
4. Mean normalized disturbance-response error, defined below, is at most `1.5` times that of the shared-information predictor-PD teacher on the same scenarios and condition.

These are declared operational bounds for a small pilot, not a stability theorem or an optimality guarantee. Record every failure. Do not tune thresholds, discard failing seeds or conditions, or replace a head after inspecting these results. A failed competence or directional-replication check withholds a positive mechanism interpretation for the affected condition; it does not license selective reporting. Report confirmation competence again alongside intervention outcomes.

Freeze resolved configuration, source fingerprint, checkpoint identities, scenario manifests, selected heads and calibration decisions before evaluating confirmation. If an outcome motivates a revised design, give it a new version and fresh confirmation scenarios.

## 3. The causal intervention

For head `j`, multiply its block of columns in the attention output-projection matrix by a factor `a`. Leave the shared output-projection bias unchanged. This scales that head's projected contribution at every token while preserving its original queries, keys, values, masks and all subsequent computation. Other heads and MLP weights are unchanged. Make interventions from the original frozen checkpoint, not by cumulatively scaling an already modified model.

There are eight candidate heads per model. The main intervention is `a = 0.5`; `a = 1.5` tests a stronger contribution. Native control has `a = 1`. The selected candidate is never completely ablated in this pilot. The separately calibrated alternative-head control can reach scale zero under its declared matching grid.

Verify exact identity at scale one, appropriate padding and causal-mask behavior, and restoration of the original predictions after a perturb-and-restore cycle. Add a small trajectory replay check for the restored controller. Because the model is frozen and deterministic, restoration is an expected implementation check, not independent biological-style evidence of rescue.

## 4. Functional suppression, not weight sign

Run the native controller on a discovery pulse and its sham. At matching dispatch times `t`, retain both immutable histories, `H_p(t)` and `H_0(t)`. For any head scale `a`, evaluate the full downstream model on each of these **fixed native histories**. Here `u` is the physical command after restoring action units and clipping to the common limit `[-5, 5]`, not the raw scalar readout. Let

\[
R_{j,a}(t)=u_{j,a}(H_p(t))-u_{j,a}(H_0(t)).
\]

The corresponding native response is `R_1(t)`. On dispatches in the inclusive window from pulse onset through onset plus `2 s`, calculate the RMS of this action response divided by the absolute disturbance amplitude. The window is fixed in physical time, and discovery has one fixed dispatch cadence. Denote these per-scenario values by `A_{j,a}` and `A_1`. The discovery score is

\[
S_j=\operatorname{median}_{\text{scenario}}
\left(\frac{A_{j,0.5}}{A_1}-1\right).
\]

The amplitude normalization makes raw response magnitudes comparable; it cancels in the within-scenario fractional ratio. The fixed numerical guard is `1e-8`. A group whose normalized native response is at or below this guard has an undefined ratio, and any such group makes the candidate ineligible. Do not drop those groups or divide by a small replacement denominator to create a large discovery score.

A head qualifies only if its median fractional increase exceeds `0.05` and at least `75%` of discovery scenarios have a positive increase. Select the qualifying head with the highest score for each transformer, breaking ties lexicographically by layer index and then head index. Do not force a selection when no head qualifies; record that outcome explicitly.

This definition identifies a contribution that causally limits a specified disturbance-evoked action signal on these histories. It does not identify suppression from a negative parameter, a negative motor command, or an arbitrary residual-stream coordinate. It also does not establish that limiting the action response is beneficial: that is the separate closed-loop test.

On independent calibration scenarios at the same discovery timing, require a positive median fractional increase and at least `75%` positive scenarios. Do not substitute a different head if this directional replication fails. Preserve the failed result and withhold a verified-suppression interpretation.

The fixed-history intervention is used only for discovery and immediate-effect calibration. During confirmation, every intervened controller receives observations from its **own resulting physical trajectory**. Do not replay activations or observations from the native trajectory after the worlds have diverged.

## 5. Controls selected without confirmation outcomes

Use six learned-controller variants when a discovery candidate exists:

| Variant | Definition |
|---|---|
| Native | Original transformer, scale one |
| Candidate weakened | Selected head scaled to `0.5`; primary intervention |
| Candidate strengthened | Selected head scaled to `1.5` |
| Norm-matched alternative | Alternative head in the same block scaled to `0.5` |
| Output-effect-matched alternative | That alternative head at a calibrated scale |
| Output gain control | Native physical command multiplied by a calibrated gain, then clipped |

Choose the alternative head on discovery histories by the nearest log RMS magnitude of its projected residual contribution to the candidate's. Exclude the shared output bias. The magnitude is `sqrt(mean_valid_token ||c||²)`, where `c` is the head's projected residual contribution, pooling pulse and sham discovery histories equally at their common cadence. Break ties lexicographically by layer index and then head index. Report the achieved norm ratio. With only three other heads in the block, a close magnitude match is not guaranteed.

This alternative is not necessarily random or nonsuppressive. Report its own functional suppression score and do not label it a nonsuppressive null merely because it was not selected as the candidate.

For immediate-effect calibration, compare **physically clipped commands** on the same fixed native pulse and sham histories. Let `delta_candidate` be the difference between the candidate-at-half command and the native command. Measure its RMS over the specified response window. Pool calibration effects with equal weights for scenario × timing-condition pairs, averaging within each pair before averaging across pairs; report per-condition effects as well as the pooled value.

For the output-effect-matched alternative, search the fixed head-scale grid `0, 0.05, ..., 1.0`. Select the scale whose RMS immediate command change is closest to that of the candidate-at-half intervention. Use a fixed numerical tie break. A match is satisfactory only within `10%` of the target RMS; report the achieved ratio and any failed match. Do not silently describe a poorly matched control as equivalent.

For the gain control, search the fixed gain grid `1.0, 1.025, ..., 3.0` and minimize the same RMS-change discrepancy, clipping after multiplication. Amplifying gain is chosen prospectively to mirror the targeted increase in disturbance-evoked action amplitude. Its direction is not selected from closed-loop outcomes. Report the achieved pooled and per-condition match and the same `10%` tolerance. Clipping can prevent an exact match; record that limitation rather than extending the grid after observing results.

All control settings are selected on calibration data and then fixed across confirmation scenarios and timing conditions. RMS matching controls perturbation magnitude, not the complete direction, waveform or state dependence of the policy change. Consequently, a residual difference is evidence against this particular simple gain explanation, not proof that every possible gain-based model has been excluded.

## 6. Closed-loop disturbance-response endpoint

For a controller/intervention variant `v`, let `q_v,p(t)` be position in its pulse rollout and `q_v,0(t)` position in its own paired sham. If pulse offset is `t_end` and signed amplitude is `D`, define

\[
J_v=\frac{1}{D^2}
\int_{t_{\mathrm{end}}}^{t_{\mathrm{end}}+3\,\mathrm{s}}
\left[q_{v,p}(t)-q_{v,0}(t)\right]^2\,dt.
\]

The window fits within all five-second episodes. With the declared normalized input/output units, `J` has units of seconds. It quantifies the post-pulse disturbance-evoked position response, rather than an absolute tracking loss or proof of asymptotic stability. Calculate it on the reporting grid with a documented endpoint/interpolation rule.

The same intervention must be present in both members of a pair. Subtracting the native sham from an intervened pulse would confound altered baseline behavior with disturbance response. Conversely, subtraction must not hide an unusable baseline: separately report absolute sham position/effort, absolute pulse tracking, numerical completion, operational competence, peak disturbance response, final-window recovery, action effort and action variation.

Censored or incomplete pairs have unavailable full-window response scores and explicit reasons. Do not replace them with zero or compare a partial integral with a complete integral. Keep all intervention failures visible; reduced movement alone is not an improvement in the stability–responsiveness tradeoff.

## 7. Primary temporal interaction and interpretation

For each frozen model and confirmation scenario, calculate the paired intervention effect

\[
\Delta J(d)=J_{\mathrm{candidate},0.5}(d)-J_{\mathrm{native}}(d).
\]

The primary contrast is the mean across confirmation scenarios of

\[
I=\Delta J(0.2\,\mathrm{s})-\Delta J(0).
\]

Use **fixed-cadence inference** for this primary contrast. Positive values mean weakening the identified suppression increases disturbance-response error more at the longest tested delay than at zero computation delay. This sign alone does not establish useful suppressive organization; candidate replication, native competence, matched controls, absolute behavior and consistency across model seeds also matter.

Compute the analogous temporal contrast for each control and compare it with the candidate's contrast on the same scenarios. If effects are explained by the matched alternative or gain control, report that limitation. Middle delays, strengthening to `1.5`, and all serial-schedule contrasts are secondary descriptive results. Serial inference combines changes in latency and decision cadence and must remain separate in figures and summaries.

Report all three model-seed effects and the underlying paired scenario values. Eight scenarios from one frozen model are not eight independent models. This pilot has no formal power calculation and supports no population-level claim of architectural superiority. Selection among eight heads occurs only on discovery data, not on confirmation interaction values.

If all three seeds yield a candidate, confirmation comprises `8 scenarios × 3 models × 2 schedules × 4 delays × 2 pulse/sham members × 6 variants = 2,304` learned-controller rollouts, plus `128` predictor-teacher rollouts. A missing candidate reduces the applicable intervention count and must be reported as such, not filled by an arbitrary head. Calibration failures do not justify dropping selected models or timing conditions from the recorded results.

## 8. Recorded outcomes and next decision

Save configuration and stage manifests; checkpoint/source/data fingerprints; full scenario specifications; candidate scores for all heads; calibration replication and matching diagnostics; per-variant and per-condition metrics; paired primary contrasts; and compact figures. Preserve enough snapshot and trajectory information to reproduce the selection and endpoint calculation. Freeze discovery/calibration decisions before confirmation and verify their integrity when advancing stages.

Valid outcomes include no reproducible suppressive candidate, suppression without a beneficial control role, stronger effects attributable to generic gain, a timing-dependent effect that also damages baseline behavior, and a beneficial temporal interaction surviving the declared controls. None establishes E/I balance. A promising result motivates more independent models, improved dynamical analysis and fresh confirmation data before an explicit architectural comparison; a negative result narrows the hypothesis without triggering unreported redesign.
