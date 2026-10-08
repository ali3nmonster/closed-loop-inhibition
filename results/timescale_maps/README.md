# Fixed-delay plant/noise timescale maps

Force-disturbance experiment, 8 October 2026. **Complete: 48 trained models,
4,950 simulator rollouts and all four map sets.** The scientific implementation and protocol
were committed at `8f7bbdd` before the full run. No training or selection rule
was changed in response to confirmation outcomes.

## Plots

All figures live in [`plots/`](plots/), in PNG and SVG formats:

- [Control accuracy, effort and failures](plots/performance.png)
- [Functional suppression and initialization comparison](plots/suppression.png)
- [Causal effects of pathway interventions](plots/causal_effects.png)
- [Control matching, drift and interpretation checks](plots/control_quality.png)

The horizontal axis is plant tau (50, 100, 200, 500 ms); the vertical axis is
force-noise correlation time (10, 50, 200, 1000 ms). Both axes are logarithmic,
and every colored rectangle represents a measured grid cell, without spatial
interpolation. Update period and virtual computation delay are both 50 ms.
Thus delay/plant-tau is 1, 0.5, 0.25 and 0.1 from left to right. Plant tau is
inverse natural angular frequency, not oscillation period.

## Control results

We trained **48 ordinary transformers**, three independent initializations per
timing cell. Each has two blocks, four attention heads per block, width 32 and
17,889 parameters. Each cell uses 1,296 expert examples from 16 demonstrations,
486 validation examples from six separate episodes, and the same 50-epoch
optimization budget. Checkpoint selection uses supervised validation MSE only.

All **192 held-out transformer trials** completed with zero task failures and
zero numerical censoring. On cell means, transformer/passive position-RMS ratios
range from **0.338 to 0.953**: learned control improves over passive motion in
all 16 cells. Transformer/teacher ratios range from **0.987 to 1.073**. These
small differences do not establish superiority over the teacher. The hardest
fast-plant/fast-noise corner gives only about 4.7% improvement over passive
motion; being finite and below the failure threshold would not alone establish
useful control there.

The teacher is a causal delay-aware predictor-PD controller, with the same
bounded observations and applied-command information as the learner. It has
declared knowledge of the plant model but no future disturbance information.
Its mean RMS improves over passive in every cell, although at the hardest
corner it wins only two of four individual tapes.

Performance RMS is measured over 3–12 s from a zero initial state. It is a
finite-horizon comparison, not a stationary estimate or proof of asymptotic
stability. The four environmental tapes are paired across model seeds and
conditions. Figures report seed ranges and denominators; three model seeds
do not constitute twelve independent network replications.

## Suppression and training

Discovery identifies a qualifying pathway in **48/48 models**. Weakening the
selected head by 10% increases its fixed-history disturbance response by
1.33–4.31% on discovery data; selection makes these discovery magnitudes
optimistic, so confirmation is reported separately.

On the common teacher-input bank, mean eligible-head fractions are **57.29% at
initialization, 48.96% at epoch 25, and 47.40% at the selected checkpoint**.
From initialization to selection, the fraction increases in eight models,
decreases in 28 and is unchanged in 12. All four plant-speed averages decrease.
Thus functional suppression is already common at random initialization, and
this experiment does not show a general training-induced increase in its
prevalence. Heads and repeated initializations across timing cells are not
independent biological or statistical replicates.

All 48 alternative-head controls pass the prespecified 5% matching tolerance;
**32/48 output-gain controls pass**. Failed matches remain recorded, and the
search grids were not refined after seeing these outcomes. The maximum mean
command error after calibration centering is below 1e-10. This calibration
constraint does not guarantee zero confirmation drift.

All **48/48 selected pathways retain functional suppression on held-out
histories**. Their centered weakening raises pulse-evoked command RMS by
1.18–4.21%, averaging 2.34%. Strengthening produces the opposite response change.
This establishes the operational pathway effect; it does not establish benefit.

## Causal effects depend on the timescales

Across all 48 models, centered weakening changes noise-regulation position RMS
by **−0.986%**, command RMS by **+2.307%**, and paired pulse-recovery energy by
**−1.486%**. These are equal-model means of percentage changes relative to each
model's native controller. Noise RMS improves in 42 models and worsens in six.
Strengthening increases mean noise RMS by 1.045%.

The fastest plant reverses the usual pattern. With plant tau 50 ms and noise
correlation time 10 ms, weakening increases cell-mean noise RMS by **1.29%**;
at noise correlation time 50 ms, the increase is only **0.065%**. In the same
fastest plant, weakening worsens paired pulse-recovery energy at all four noise
timescales (cell means +0.88% to +1.76%). For the slower plants, mean pulse
recovery improves. Noise regulation and pulse recovery are distinct endpoints,
so their crossover patterns need not coincide.

On the **same 32 models with valid output-gain matches**, the gain control has
lower noise RMS than head weakening in 29/32 models and lower recovery energy
in 25/32. Its mean noise-RMS effect is 0.524 percentage points lower, recovery
effect 1.236 points lower, and effort effect 0.503 points lower than weakening.
The gain panel therefore cannot be compared by simply subtracting its mean
over 32 models from the weakening mean over all 48. The saved summary includes
the properly paired comparisons.

For the fastest plant and fastest noise, the two successfully matched models
have mean noise-RMS changes of **+1.740% under head weakening versus −0.032%
under scalar gain**. Their recovery-energy changes are +2.227% versus −1.314%.
Thus the fast-loop harm from altering the internal pathway is not reproduced
by an output-gain change matched for immediate command-effect magnitude. This
is compatible with a change in temporal filtering, but phase and damping were
not measured here.

The maps show a small, timing-dependent functional contribution, including
protective effects of the selected pathways in demanding conditions. They do
not establish a general advantage from more inhibition or an explicit E/I
architecture. Generic gain changes are an effective comparator, and the
fast-condition difference between head intervention and gain manipulation
needs targeted replication and measurements of the joint loop's dynamics.
No significance or precise boundary claim is made from this exploratory grid.

All **2,880 own-history intervention rollouts** complete without task failure
or numerical censoring. The command/position drift panels remain part of the
interpretation, even though mean corrections were fitted before confirmation.

## Measurement meaning

Functional suppression means that weakening a selected head's projected
attention contribution raises its disturbance-evoked command response on
identical fixed histories. Selection uses discovery tapes; offsets and matched
controls use separate calibration tapes; confirmation uses fresh tapes.

The initialization, epoch-25 and selected-checkpoint panels use identical
teacher histories with standardized pulses and zero background noise. Their
input bank is identical across training noise timescales at fixed plant tau.
These panels distinguish training-associated changes from functional
suppression already present at random initialization. Conditional tracking of
a head selected after training is not an unbiased emergence-rate estimate.

Closed-loop intervention maps instead let each policy act on its own evolving
history under the same exogenous input. Positive weak-minus-native RMS means
weakening harms accuracy; negative means it improves accuracy. Effort and drift
are separate measurements. Centering preserves mean command on calibration
histories only, so confirmation drift is measured directly. Matched-control
panels include only successful prespecified matches, with denominators shown;
all unmatched effects remain in the numeric records.

These networks learned by **expert imitation**, not direct optimization of
closed-loop reward. This first grid tests ordinary transformers under physical
force noise. It does not test an explicit E/I architecture or sensor noise, and
functional suppression does not establish biological excitation/inhibition
balance.

## Reproduction and files

See the [frozen protocol](../../docs/TIMESCALE_MAPS_PILOT.md) and
[configuration](config.json). To reproduce into fresh directories:

```bash
.venv/bin/python experiments/run_timescale_maps.py --output runs/timescale_reproduction/results --artifacts runs/timescale_reproduction/artifacts
.venv/bin/python experiments/plot_timescale_maps.py --input runs/timescale_reproduction/results
```

- `performance.csv`: all passive, teacher and transformer evaluation trials.
- `training.csv`, `learning_curves.csv`: model summaries and every training epoch.
- `prepared/`: all head assays, discovery selection and frozen calibration grids.
- `confirmation/`: fixed-history tests and seed-level own-history intervention outcomes.
- `map_cells.csv`, `model_metrics.csv`, `seed_ranges.csv`: plotted values and variability.
- `run_manifest.json`: source, protocol, data, checkpoint and stage-result hashes.
- `validation.json`, `artifact_manifest.json`: final checks and publication-artifact hashes.
- `reporting_convergence.json`: prespecified 10 ms versus 5 ms reporting checks.

Weights and demonstration arrays remain on this server under
`runs/timescale_maps/checkpoints/` and `runs/timescale_maps/data/`; they are
excluded from Git. Each model retains initialization, epoch 25, selected and
final weights. Resume requires unchanged scientific inputs and sealed artifacts.

## Validation and accounting

The final software suite passes **357 tests**. The six prospectively chosen
reporting-resolution repeats change position RMS by at most **0.0114%** when
reporting is refined from 10 ms to 5 ms, with physical forcing and controller
timing unchanged. All scientific-source and completed-artifact fingerprints
are checked independently after the run.

| Stage | Physical rollouts |
|---|---:|
| Passive and teacher performance | 128 |
| Training/validation teacher demonstrations, reused across model seeds | 352 |
| Neural performance confirmation | 192 |
| Common teacher-history probe arms | 240 |
| Native discovery probe arms | 288 |
| Native calibration probe arms | 288 |
| Held-out fixed-history source arms | 576 |
| Own-history intervention arms | 2,880 |
| Reporting-convergence repeats | 6 |
| **Total** | **4,950** |

Each confirmation sham is reused for both pulse signs: the 1,920 saved
pulse-pair records do not imply 1,920 independent sham trials. Development
pilots and the separate integration smoke test are excluded from this count.
