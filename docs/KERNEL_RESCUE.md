# Equilibrium, gain and local-response rescue with training replication

This experiment follows the [complete-loop dynamics and partial gain-rescue result](../results/loop_mechanism/README.md). That experiment found a timing-dependent damping reversal in twelve existing models, with only partial recovery after independently calibrated gain/mean compensation. Residual fixed-history waveform mismatch, operating-point shifts and three initialization seeds limit interpretation. The present experiment separates those explanations and adds independent training replication.

The [configuration](../configs/kernel_rescue.json), source files and this protocol are frozen before production. Existing models support an exploratory bridge to previous results; newly trained models form the primary replication population. Development smoke runs use separate seeds and directories and never enter research aggregates.

## Populations and physical conditions

Use the 200 ms plant with damping ratio 0.15, 50 ms observations/decisions, eleven tokens covering 0.5 s, action limit 0.5 and the original force-noise amplitude 0.02. Compare physical command latencies 50 and 100 ms. Both have the same current-action-age feature at mature dispatch; the planned-delay cue remains 50 ms. Observation/action histories remain truthful. Constant latency above the decision interval uses the existing idealized pipeline. This is a targeted replication in the previously identified regime, not a test of universality across plant timescales.

The existing population contains the original three initialization seeds crossed with four training-noise correlation times (0.01, 0.05, 0.2 and 1 s): twelve frozen selected checkpoints. Their native and joint-weak policies retain exact inherited branch gates, gain, offset and center.

The fresh population contains ten new initialization seeds crossed with those same four noise conditions: forty models. Keep architecture, teacher, training duration, example counts, optimizer, fifty-epoch budget and validation-MSE checkpoint selection unchanged. Generate independent training and validation demonstration streams for every new model condition. A replicate shares an initialization seed across its four noise conditions but does not share demonstration tapes between those conditions. Save exact per-model training configurations, datasets, training histories and initial/intermediate/final/selected checkpoints. Training and validation streams do not overlap earlier studies, pathway discovery, calibration or confirmation.

No model is selected based on closed-loop test performance. Training failures, unavailable fits, task failures and censoring remain explicit. Do not average only surviving conditions.

## Pathway discovery and base intervention

For the twelve existing models, reuse previously selected suppressive groups unchanged. For each fresh selected checkpoint, apply the same ten-branch assay: eight projected attention-head contributions and two complete MLP residual outputs. Use a shared teacher-history bank from new discovery streams with background correlation time 0.2 s and standard deviation 0.01. The four pulse amplitudes and physical timing are specified in the configuration.

A branch is eligible if multiplying it by 0.9 produces more than a 1% median increase in pulse-minus-sham command RMS, at least 75% of probe effects are positive, and every native response exceeds 0.0001. Select only from discovery data. An empty or unclassifiable group remains recorded; its planned weak intervention is the identity, and the model stays in the all-model population. Report eligibility prevalence and any conditional eligible-group summaries separately. Do not manufacture a positive branch selection.

For fresh models, retain native gain one and offset zero. Fit the weak policy's clipped command offset on a separate native-history calibration bank at 50 ms, using new offset-calibration streams. This defines its base `joint_weak` policy; all later corrections start from that frozen policy. Existing weak offsets are inherited unchanged. Discovery, base-offset calibration and correction calibration use disjoint streams.

## One anchor and one calibration for both delays

Find the native disturbance-free equilibrium at 50 ms. With mature histories, denote its physical position and command by `q* = u*`, and its accessible normalized history vector by `h*`. The vector has 34 coordinates: eleven triples of position/state_scale, tau*velocity/state_scale and captured_action/output_scale, followed by current_action/output_scale. Pending commands are excluded from this controller input vector. The complete physical map also contains the command pipeline where required.

Let `f_w(h)` denote the pre-clipping output of the frozen weak policy, including its inherited affine transformation. Let `K_n` and `K_w` be the native and weak pre-clipping command derivatives at the common anchor, in the declared normalized coordinates. The native anchor must be interior to its actuator limit; the weak derivative is taken before clipping even if its uncorrected output would saturate at that anchor. Define all corrections at 50 ms and carry their coefficients, anchor and scalar gain unchanged to 100 ms.

| Label | Command before the final actuator clip | Purpose |
|---|---|---|
| `native` | Original native policy | Reference |
| `joint_weak` | Original weak policy | Original intervention |
| `weak_equilibrium` | `u* + f_w(h) - f_w(h*)` | Restore the resting operating point by a constant correction |
| `weak_scalar` | `u* + g [f_w(h) - f_w(h*)]` | Add compensation of overall response strength |
| `weak_kernel` | `u* + g [f_w(h) - f_w(h*)] + (K_n - g K_w)(h-h*)` | Restore the full local command response as well |

Apply the same actuator limit to all variants. The scalar gain `g` is a positive bounded least-squares fit of the weak pulse-minus-sham response to the native response on independent native histories at 50 ms. Fit the entire response waveform with equal weight per probe; do not fit closed-loop recovery error or refit at 100 ms. Record gain-bound hits, degenerate fits, clipping and independent confirmation mismatch. Exact equilibrium anchoring replaces a calibration-sham mean match for the three corrected variants.

For runtime evaluation, save anchor values consistent with the original float32 network implementation. For smooth local analysis, retain the same trained weights represented in float64 and corresponding float64 anchor values. Validate their physical command agreement within the declared tolerances. During startup, the network receives its ordinary available history; the additive correction uses only available observations aligned to their actual lags. Missing old observations contribute no correction. No future observation, pending command, simulated live state or artificial full-history token enters the policy.

When there is no eligible group, preserve exact native identity rather than interpreting tiny floating-point anchoring differences as an intervention. Record this case explicitly.

## What the local construction establishes

At an unclipped differentiable anchor, the full correction has the native command value and derivative. Combined with the unchanged plant, history shift and delay pipeline, the complete local closed-loop Jacobian should therefore match native at each tested delay. This pole match is an implementation check, not independent evidence for a discovered mechanism.

The scientific test is whether a correction constructed at one operating point and one delay generalizes to finite disturbances, noisy trajectories and the longer physical latency. Success would support a local linear-feedback explanation of the remaining effect. Failure would motivate nonlinear or state-dependent effects or limits of the common-anchor approximation. Neither result alone establishes transformer specificity or biological E/I organization.

Save command sensitivities and correction contributions separately for position, velocity, captured/current actions and observation lag. Use the fixed physical normalizations. A non-scalar residual can reflect a changed position-versus-velocity balance as well as weighting of older observations; do not label all non-scalar changes as memory or temporal filtering. Report correction magnitudes and clipping on each policy's own trajectories, not only the native calibration histories.

## Confirmation and primary contrasts

Seal every correction preparation before any production confirmation. Use four new OU tapes per condition, shared across policies and pulse/sham arms. A pulse begins at 1 s, lasts 0.1 s, and has amplitude ±0.02. Reuse one physical sham per policy/tape/horizon for both pulse signs. Use the original force standard deviation and each condition's noise correlation time. Include shared passive zero-command references.

The primary horizon is 4 s and scores [1, 4] s. The secondary horizon is 12 s and scores [1, 12] s; its noise tapes share their initial segment with the short trials. These two horizons are not independent replicates. The longer horizon tests effects that may remain below the task threshold at four seconds. Preserve complete failed trials numerically and censored full-window quantities as null.

For a model and policy `v`, define `E_v(L)` as the mean amplitude-normalized integral of squared pulse-minus-own-sham position. Define `D_v(L) = E_v(L) - E_native(L)` and `I_v = D_v(100 ms) - D_v(50 ms)`. Compute model-level contrasts before averaging.

The two principal fresh-population, four-second estimates are:

1. `I_joint_weak`: replication of the timing-dependent weakening effect.
2. `I_weak_kernel - I_weak_scalar`: incremental effect of correcting the remaining local command-response shape after equilibrium and scalar compensation.

Report every policy's absolute and relative effect at both delays, so a smaller interaction cannot hide a constant performance penalty. The kernel-minus-native residual, intermediate correction steps, existing-model outcomes, eligible-group-only outcomes and twelve-second results are secondary. Also retain sham position RMS, action effort, means, clipping, task failures, censoring and native-to-passive performance.

Use initialization blocks as the replication units: first average the four noise conditions within each fresh seed, then average the ten seed-block effects. Show all model and seed-block values. For the fresh population, report a descriptive 95% percentile bootstrap interval from 10,000 resamples of whole seed blocks, using the dedicated frozen statistics seed. These intervals condition on the fixed task/noise grid and shared confirmation tapes. They are not forty-model or episode-level IID intervals, multiplicity-adjusted hypothesis tests, or evidence of equivalence merely because zero is included. Existing three-seed summaries have no bootstrap interval. A required missing cell makes its block and the full-population aggregate unavailable; retain explicit missing counts rather than dropping it.

## Local analysis and verification

Analyze the complete event-scheduled map at each policy's own equilibrium, with the three corrected policies analyzed at their constructed native anchor. Include observation history, held action and pending commands; validate parity against the original event simulator at both delays. Save all matrices, spectra, equilibrium residuals and kernel-versus-native Jacobian discrepancies. Respect the original capture/completion/application ordering.

Use short ±0.000001 pulses to validate the numerical linearization over the first 0.5 s. Also retain four-second zero-background ±0.02 map responses and their local linear predictions. These map analyses are distinct from noisy physical confirmation trials and are not counted as additional event-simulator trials. Unstable equilibria remain recorded; a finite formal frequency resolvent has no stable steady-state interpretation in such cases.

The runner verifies all inherited scientific manifests, records a clean launch commit and hashes the new scientific sources, configurations and protocol. Training artifacts, discovery banks, preparations and complete results are sealed independently; changed or unsealed dependencies cannot be resumed silently. Plotting and reporting stay outside the scientific freeze and cannot retune the experiment.
