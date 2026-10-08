# Suppressive pathways under a within-controller delay sweep

Physical-latency experiment, 8 October 2026. This follows the
[collective suppression results](../collective_suppression/README.md) using
the same 48 trained models and identified pathways. The new experiment varies
physical command latency **within each fixed controller**, while keeping its
plant, noise condition and 50 ms observation/decision cadence unchanged.
Scientific implementation and protocol were committed at `4740ffc` before the
full confirmation run. There is no retraining, pathway selection or calibration.

**Complete: 336 model–delay conditions, 20,352 new physical trials and 12 figure
sets.** The experiment finds a clear within-controller crossover for the 200 ms
plant, but does not support a universal or monotonic increase in suppression's
usefulness with latency. The prespecified pooled absolute interaction and its
percentage-based counterpart have opposite signs. Generic output-gain changes
also reproduce the clearest crossover.

## Main result: a causal crossover in a competent control regime

For plant tau 200 ms, increasing physical delay from zero to 100 ms changes the
effect of the same frozen pathways in the same controllers:

| Frozen intervention | Recovery effect at zero delay | Recovery effect at 100 ms delay |
|---|---:|---:|
| Weaken suppressive group by 10% | −9.65% | +26.50% |
| Strengthen suppressive group by 10% | +9.75% | −9.42% |
| Gain control calibrated at 50 ms | −9.81% | +13.53% |

Positive values mean worse pulse recovery relative to the native controller at
that delay. Each entry averages 12 models: three initialization seeds at each
of four noise timescales. Weakening switches from helpful to harmful in
**11/12 models**, and all 12 have a positive absolute high-minus-low interaction.
Strengthening switches from harmful to helpful in 10/12, with negative absolute
interactions in all 12.

This crossover occurs with useful native control. Mean native noise RMS is
0.417 times passive at zero delay and 0.559 times passive at 100 ms. Native,
weakening, strengthening and gain variants have **no task failures at these
endpoints**. The selected groups still have suppressive fixed-history effects
in all 12 models at both endpoints: their weakening increases command-response
RMS, and strengthening decreases it.

The gain comparison limits the mechanistic interpretation. All 12 gain controls
qualified at the original calibration; eight also switch from helpful to
harmful. At 100 ms, gain has lower recovery energy than internal weakening in
all 12 models, by an average 12.97 percentage points of native-relative effect.
Thus timing changes the causal usefulness of these internal pathways, while
a scalar gain manipulation reproduces the qualitative crossover. This does
not identify a uniquely inhibitory internal algorithm or isolate damping/phase
as its mechanism. Calibration matching is not guaranteed at the new delays.

## The prespecified primary result is heterogeneous

The primary contrast was fixed before confirmation: the weakening-minus-native
recovery-energy difference at 100 ms, minus that difference at zero delay.
Here "absolute" means an absolute difference in the amplitude-normalized
recovery metric defined below, not mechanical energy.

| Plant tau | Mean absolute interaction | Mean percentage-point interaction | Helpful → harmful | Harmful → helpful |
|---|---:|---:|---:|---:|
| 50 ms | −0.0714723 | −4.99 | 0/12 | 7/12 |
| 100 ms | −0.6302842 | −14.11 | 0/12 | 2/12 |
| 200 ms | +0.0160328 | +36.15 | 11/12 | 0/12 |
| 500 ms | +0.0000688 | +6.08 | 0/12 | 0/12 |
| **All 48 models** | **−0.1714137** | **+5.78** | **11/48** | **9/48** |

The absolute interaction is positive in 27/48 models and negative in 21/48;
the percentage-point interaction is positive in 28/48. A positive interaction
can indicate weakening becoming less helpful without becoming harmful, as in
the 500 ms group. The two mean signs differ because each percentage uses a
different native baseline, while large absolute changes in the faster plants
dominate the absolute mean. The positive percentage-point mean does not replace
or rescue the negative prespecified absolute mean.

Across all models, the strengthening interaction averages +0.2301555 in the
absolute metric and +0.55 percentage points, also without a uniform predicted
sign. All models retain complete endpoint outcomes; these comparisons do not
exclude task-failed trials or select a surviving cohort. The result supports
regime-dependent effects, not a general rule that increasing delay makes
suppression more beneficial.

## Full curves, competence and failures

The fastest plant reproduces the earlier local result on fresh environmental
streams: at the original 50 ms delay, weakening worsens recovery by **15.48%**
on average, with 9/12 models harmed. This resembles the previous 17.19% mean,
but it is a fresh-probe repeat on the same trained models, not a new training
replication. Across the full delay sweep, its mean weakening effects are:

| Physical delay | 0 ms | 25 ms | 50 ms | 75 ms | 100 ms | 150 ms | 200 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fastest-plant recovery change | +1.51% | +6.12% | +15.48% | −1.35% | −3.47% | +2.82% | −9.94% |

This is nonmonotonic. At 75 ms and above, all 48 native noise-only trials in
this fastest-plant group exceed the task-failure threshold. An improved
incremental pulse response there does not establish acceptable control of the
underlying noisy trajectory. Even being below the failure threshold does not
guarantee improvement over passive motion; both comparisons remain visible.

Across all variants, **5,436 of 20,160 own-loop trials fail the task threshold**.
None is numerically censored, and none of the 192 passive trials fails. All
numeric outcomes are retained. No actuator clipping occurs in native sham
trials; poor control cannot simply be attributed to clipping at the action
bound.

| Delay | Native models with all 12 trials passing | Native recovery better than passive | Native noise RMS better than passive |
|---|---:|---:|---:|
| 0 ms | 48/48 | 48/48 | 36/48 |
| 25 ms | 48/48 | 48/48 | 36/48 |
| 50 ms | 48/48 | 48/48 | 45/48 |
| 75 ms | 36/48 | 24/48 | 26/48 |
| 100 ms | 27/48 | 23/48 | 24/48 |
| 150 ms | 23/48 | 12/48 | 12/48 |
| 200 ms | 13/48 | 12/48 | 11/48 |

These counts are descriptive competence checks, not exclusion criteria.
Recovery and noise regulation are different endpoints and can disagree.
The 200 ms plant at 100 ms delay supplies a reversal at the prespecified primary
endpoints while control remains useful; the failed regimes require different
interpretation. The complete 500 ms plant curves show a later crossover:
weakening changes from −12.92% at zero delay to +24.26% at 200 ms, with all
12 models switching sign. Strengthening changes from +12.68% to −11.73%.
This later endpoint comparison is descriptive secondary evidence, not a
replacement for the prespecified 0–100 ms primary contrast.

## Figures

PNG and SVG figures live in [`plots/`](plots/):

- [Weakening and strengthening: recovery effects across delay](plots/delay_recovery_effects.png)
- [Absolute pulse-recovery energy](plots/delay_recovery_energy.png)
- [Recovery relative to passive motion](plots/delay_passive_ratios.png)
- [Noise-regulation effects](plots/delay_noise_effects.png)
- [Absolute noise-regulation error and passive reference](plots/delay_noise_accuracy.png)
- [Noise error relative to passive motion](plots/delay_noise_passive_ratios.png)
- [Task failures for noise-only and pulse trials](plots/delay_task_failures.png)
- [Actuator saturation](plots/delay_saturation.png)
- [Applied action effort](plots/delay_action_effort.png)
- [Frozen calibrated controls versus weakening](plots/delay_controls.png)
- [Matching drift on changed histories](plots/delay_match_drift.png)
- [Prespecified within-model interactions](plots/delay_interactions.png)

Each curve panel fixes the plant and noise condition. Its horizontal axis is
physical command latency; the 50 ms training delay is marked. Individual model
curves and ranges show variation across three initialization seeds per cell.
Shared environmental tapes are paired probes, not additional trained models.
The seven tested delays are 0, 25, 50, 75, 100, 150 and 200 ms.

## What stays fixed

For each model, use exactly its previously selected checkpoint, suppressive
branch group and five saved controller variants: native, joint weakening,
joint strengthening, output gain and alternative group. Every gate scale,
command offset, gain and centering constant is copied unchanged from the
previous 50 ms calibration. No setting is adjusted at a new delay.

The encoder normally receives planned application delay as a feature. This
experiment holds that explicit cue at its training value of 50 ms before
encoding, while the simulator applies commands at the actual swept delay.
Thus changing latency does not also instruct the network to use a different
delay cue. Realized observations and applied-action histories remain truthful
and change naturally through feedback. This tests a latency perturbation
without updating the cue; it is not a comparison of controllers optimized for
each deployment latency. At 50 ms, the policy agrees with the inherited one.

The plant, damping, disturbance standard deviation, force-noise correlation
time, force clock, feature scales, history length and action limit remain
fixed within each model's sweep. The command limit is 0.5. The forcing clock
is 2.5 ms, independent of controller timing. At latencies exceeding the 50 ms
decision period, multiple stateless inference jobs are pending. This is an
idealized pipeline with fixed throughput, not a serial processor taking longer
between decisions. Per-trial job records verify dispatch spacing and actual
command-application lag.

## Paired probes and the primary question

Four new OU force streams (3010001–3010004) are paired across delays, variants
and pulse/sham arms, with standard deviation 0.02 and each model's original
noise correlation time. Four-second episodes start from zero state. Pulses
have amplitudes ±0.02, start at 1 s and last 0.1 s; scoring covers [1,4] s.
Each seed has one unique sham reused for both pulse signs. Passive zero-command
trials provide an independent performance reference for each plant/noise cell,
shared across model seeds and delays.

Recovery energy E is the integral of squared pulse-minus-own-sham position
difference, divided by squared force-pulse amplitude, then averaged over the
four tapes and two signs. It measures incremental disturbance recovery over
the declared horizon; it is not mechanical energy or total task utility.

For each model, define Δ(d) = E_weak(d) − E_native(d). The **prespecified
primary interaction** is Δ(100 ms) − Δ(0 ms). Positive values mean weakening
becomes more harmful or less helpful over that contrast. A positive interaction
alone is not a sign reversal; helpful-to-harmful and harmful-to-helpful counts
are reported separately. Strengthening provides a complementary intervention.

Percentage effects use each delay's own native controller:
100 × [E_variant(d)/E_native(d) − 1]. Their high-minus-low difference is in
percentage points. Absolute metric differences and native energies remain
visible to distinguish denominator changes. Ratios require a native denominator
above 1e-12. All seven delays are shown, without assuming monotonic effects or
selecting endpoints after observing outcomes.

Noise-only position RMS, applied action RMS, means, saturation, passive-relative
performance and task failures accompany the recovery metric. A completed trial
with position RMS above 0.1 is a task failure and retains its numeric outcome.
The state-divergence guard is 100; incomplete/censored outcomes have undefined
required metrics. Aggregates never silently drop missing tapes or model seeds.

## Meaning of the control comparisons

The inherited calibration qualifies 47 gain controls and 15 alternative-group
controls for magnitude-and-direction matching at 50 ms. These exact model
cohorts stay fixed across the delay sweep. A control remains plotted even if
its effective matching changes on the newly encountered histories.

At each delay, native histories are recorded during already-required native
rollouts. Replaying all frozen variants on those identical, cue-clamped inputs
measures immediate command-change RMS, response direction and mean drift.
These measurements diagnose the changing match; they never refit parameters
or select a confirmation cohort. They add no physical simulations. A control
calibrated at 50 ms is not automatically magnitude-matched at another delay.

For example, of the 47 originally qualified gain controls, 7 remain within the
magnitude/direction matching criteria on fresh zero-delay histories, 37 at
50 ms, 11 at 100 ms and three at 200 ms. The cohort remains 47 throughout;
no comparison silently replaces it with a newly selected matched subset.

The suppressive label itself also depends on encountered histories. The frozen
groups have positive weakening-induced command-response changes in 48/48
models at 50 ms, 38/48 at zero delay and 39/48 at 100 ms. The 200 ms plant's
groups remain suppressive in all 12 models at both primary endpoints, which
helps distinguish that crossover from a change in the group's functional sign.

## Reproduction and provenance

See the [frozen protocol](../../docs/DELAY_SWEEP.md),
[`config.json`](config.json), [`base_config.json`](base_config.json) and
[`run_manifest.json`](run_manifest.json).

```bash
.venv/bin/python experiments/run_delay_sweep.py --output runs/delay_reproduction/results --artifacts runs/delay_reproduction/artifacts
.venv/bin/python experiments/plot_delay_sweep.py --input runs/delay_reproduction/results
```

The prior collective experiment and all inherited artifacts must be available
at the paths recorded in their manifests, including model checkpoints,
demonstration data and common-history banks. Those large server artifacts are
excluded from Git. The runner verifies the committed parent manifest and its
original training lineage. A fresh clone alone does not contain all artifacts
needed to reproduce these exact saved models.

- `confirmation/`: one sealed record for each model and physical delay, including all trial outcomes, fixed settings, input-history checks and timing audits.
- `passive/`: one sealed zero-command reference per plant/noise cell.
- `model_metrics.csv`, `delay_cells.csv`: model and cell delay curves, including control matching and competence.
- `model_interactions.csv`, `interaction_cells.csv`: paired primary contrasts and sign reversals.
- `summary.json`: descriptive counts, complete-cohort means, controls and per-plant/per-delay results.
- `run_manifest.json`: scientific source freeze, inherited records, checkpoints and completed-result fingerprints.
- `plot_manifest.json`: plotting input/output and implementation fingerprints.
- `dynamics_audit.json`, `provenance_audit.json`: independent recomputations, competence checks and frozen-parameter audits.
- `validation.json`, `artifact_manifest.json`: final checks and publication-artifact hashes.

These are finite-horizon causal interventions within previously studied,
expert-imitation controllers. They do not measure asymptotic stability, damping,
closed-loop eigenmodes, biological E/I balance or the benefit of an explicit
inhibitory architecture. A new training-seed replication would be needed to
establish how broadly the result generalizes. How training produces suppression
is a separate question from how a frozen pathway's usefulness changes with
latency.

## Validation and accounting

The final software suite passes **477 tests**. Tests include fixed-cue
invariance, exact agreement with the inherited 50 ms policy, independent
fractional-delay classical dynamics, immutable lineage and missing-outcome
aggregation.

The scientific source freeze contains **26 fingerprints**, all matching commit
`4740ffc`; the run started with a clean worktree. All **354 sealed artifacts**
match, as do the collective parent's 24 source/250 artifact hashes and the
original experiment's 21 source/422 artifact hashes. Each result binds the
exact prior discovery, preparation and selected checkpoint.

All 20,352 physical timing audits pass. Independent checks reproduce 37,312
outcome summaries and 37,312 native-relative deltas, with maximum arithmetic
discrepancy 1.42×10⁻¹⁴, and verify identical groups/settings across delays.
Recorded native commands replay within 9.54×10⁻⁸ physical command units
(prespecified tolerance 1e-6). A separate manual gate/feature/metric replay of
30 development trials agrees exactly; it is excluded from scientific results.

| Stage | New physical simulator trials |
|---|---:|
| 48 models × 7 delays × 5 variants × 4 tapes × (sham + 2 pulse signs) | 20,160 |
| 16 plant/noise cells × 4 tapes × (passive sham + 2 pulse signs) | 192 |
| **Total** | **20,352** |

The integration smoke run used different development streams and is also
excluded. Fixed-history diagnostics reuse recorded native trajectories and
add no simulator trials. No discovery, calibration, training or independent
hardware timing measurement is included in this total.
