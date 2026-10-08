# Collective suppression in saved transformer controllers

Prospective protocol, 8 October 2026. This follows the
[48-model force-noise grid](../results/timescale_maps/README.md). It uses the
existing initialization, epoch-25 and validation-selected checkpoints without
retraining. Its configuration is
[`collective_suppression.json`](../configs/collective_suppression.json).
New scientific sources and this protocol are fingerprinted before launch;
inherited source, configuration and checkpoint bytes are verified against the
prior run manifest and committed parent results.

## Scope and component definition

The primary question is how much the model's residual pathways collectively
suppress a declared disturbance-evoked command response, and how that effect
changes during training. This is an architecture-wide branch assay, not an
architecture-independent measure of total inhibition or biological E/I balance.

Use ten nonoverlapping gates: eight attention heads and the two complete MLP
residual outputs. An attention gate scales only that head's columns of the
attention output projection, preserving the shared output bias. An MLP gate
scales both the final MLP weight and its bias, hence its complete residual
contribution. Native gates equal one. Embeddings, residual identity paths,
normalization, shared attention biases and the readout remain fixed. Their
downstream computation contributes to the measured causal effects, but they are
not separately classified. Whole MLP branches can internally mix suppressive
and amplifying activity; finer subdivisions would be a separate analysis.

The network architecture, feature scaling, physical plant and timing are
inherited: 48 models, three initialization seeds per 4×4 timing cell, 50 ms
delay and observation/update period, 2.5 ms forcing clock, actuator limit 0.5.
The inherited models were trained by expert imitation, not closed-loop reward.

## Common pulse histories and data separation

Generate teacher-controlled pulse/sham histories at each plant tau. These are
identical across training-noise timescales and model seeds at fixed plant tau,
and are reused unchanged across all three checkpoints. The common background
is OU force with correlation time 0.2 s and marginal standard deviation 0.01,
independent of a model's training noise condition. In 4 s trials add physical
pulses at 1 s, width 0.1 s, amplitudes ±0.01 and ±0.02. Use paired background
tapes for pulse/sham arms and score decision histories over [1,4) s. Save the
common banks and their hashes. They are common within a plant tau, not identical
encoded inputs across different plant dynamics.

Discovery seeds are 1010001/1010002. Common-history confirmation seeds are
1030001–1030004. Each checkpoint independently selects its group using only
discovery histories. For every gate, weakening to 0.9 must raise the median
paired command-response RMS by more than 1%, with positive effects in at least
75% of seed/amplitude groups, and native response RMS above 1e-4 in every group.
Selection is based on the whole declared probe bank, not a different gate set
for every input. Persist all selections before confirmation or spectral assays.

An unclassifiable checkpoint has an undefined group and undefined relative
aggregate effects. A classifiable checkpoint with no eligible branches has an
empty group: its collective intervention is identity and its collective effect
is exactly zero. Report these cases separately, with eligibility denominators.
Do not retune thresholds, checkpoints or groups after seeing confirmation.

## Strength, collective effect and learning change

Measure all ten single-branch effects at scales 0.9 and 1.1 on held-out common
histories, retaining physical RMS changes, changes divided by pulse amplitude,
relative changes, polarity consistency and classifiability. Approximate local
signed sensitivity with the symmetric finite difference

`s_j = [R_j(0.9) - R_j(1.1)] / (0.2 * R_native)`.

Report sums of positive and negative sensitivity magnitudes and the signed
net, summing the median per-probe sensitivity of each branch. Medians and sums
do not commute, so this is not the derivative of a pooled response norm.
These are local finite-difference aggregates under this gate definition;
they are not assumed to equal the finite joint intervention effect. Aggregate
seed/amplitude groups equally and retain per-group values.

Weaken the discovery-selected group jointly to 0.95, 0.9 and 0.8, and strengthen
it jointly to 1.1. The primary collective score is the native-relative change
at joint scale 0.9: `I = R_joint / R_native - 1`. A positive value means that
the intact group collectively restrained the measured response. Report the
full strength curve, physical and pulse-amplitude-normalized changes alongside
the percentage. Near-zero baseline responses remain undefined, not inflated.

At scale 0.9, compute the interaction residual per probe group:
`delta_R_joint - sum(delta_R_individual)` over the selected branches. Preserve
both physical and relative units, then aggregate. Joint effects may differ in
size or sign from sums of individually suppressive effects. No exhaustive
coalition/Shapley claim is made by this selected-group assay.

The primary learning-associated change is selected-checkpoint collective
score minus initial-checkpoint collective score, paired by model seed and
common input bank. Group membership is independently discovered at each
checkpoint; group sizes and identities must be visible. This definition can
reflect changes in prevalence, strength and interactions. Separately apply the
trained-selected group at every checkpoint, with an explicit selection-bias
qualification: this tracks a group chosen using the trained model and is not
an unbiased estimate of emergence. Epoch-25 is a fixed training snapshot;
the selected checkpoint's actual epoch is reported.

## Frequency-resolved direct controller assay

Use zero-background teacher histories driven by sampled-and-held sinusoids
at 0.5, 1, 2, 4 and 8 Hz, physical amplitude 0.01 and phases 0 and pi/2.
Trials last 12 s; analyze decision times [4,12) s. All frequencies are below
the 10 Hz observation Nyquist frequency. The forcing clock remains 2.5 ms.
Each frequency completes an integer number of periods in the scoring window.
These trials are repeated for each plant tau and paired with a zero-force sham.

Apply each checkpoint's pulse-discovered group at scale 0.9 to these fixed
histories. Do not select a new group per frequency. Fit sine, cosine and
intercept to pulse-minus-sham command output at the drive frequency; report
native and modified harmonic amplitudes, response RMS, relative amplitude
change and wrapped phase change. Ratios require native harmonic amplitude
above 1e-4; phase additionally requires a resolvable modified amplitude. Retain
absolute amplitudes below threshold and mark ratios/phase undefined. Verify
fitting conventions against known sinusoids in software tests.

This measures a controller response to teacher-generated histories, including
the plant/reference filter in those histories. It is not the transfer function
or damping of the neural controller's own closed loop. Compare training-noise
conditions within a plant tau to hold the probe history fixed; compare plant
taus with the physical-probe difference stated. Preserve frequency curves as
well as grid maps rather than hiding frequency dependence in one average.

## Closed-loop benefit and controls

For the selected checkpoint only, test its frozen common-discovery group on
its own histories under the cell's actual OU correlation time and standard
deviation 0.02. Calibration seeds are 1020001/1020002; new confirmation seeds
are 1040001–1040004. Use 4 s trials, physical pulses ±0.02 at 1 s with width
0.1 s, and matched noise-only sham arms. Score [1,4) s. Retain every failure.

Variants are native, jointly weakened (0.9), jointly strengthened (1.1),
matched output gain and an alternative group. Fit constant command offsets on
native calibration sham histories for each intervention. Match controls to the
centered weak group's immediate command RMS change across pulse/sham inputs,
with 5% relative tolerance. Preserve actual closed-loop drift separately.

Gain candidates range from 0.5 to 2 in steps of 0.005, centered on native
calibration sham command mean. Restrict gain candidates to the same sign of
median physical pulse-response RMS change as joint weakening on calibration
histories, then choose the closest immediate-effect RMS. If the target signed
change is within 1e-12 of zero, direction matching is unavailable. Retain failed
matches without refining the grid.

Choose one seeded alternative group with the same number of branches and
maximum difference from the selected group (seed 1100000 plus model seed).
For group size k at most five, sample k branches uniformly from its complement.
For k above five, take every complementary branch and sample the remaining
2k−10 branches uniformly from the selected group. This minimizes overlap;
report the actual overlap and Jaccard similarity.
An empty group or all-ten group has no distinct same-size alternative, so mark
that control unavailable. Match alternative strength on 0.5–1.5 by 0.01 using
immediate-effect magnitude. Report response-direction concordance and the
command-change cosine separately. Strong comparison claims require both a
valid magnitude match and concordant response direction; retain all raw control
effects and show denominators. Random group controls are not guaranteed to
isolate a different mechanism when overlap is large.

Persist calibration settings before own-loop confirmation. Report fixed-history
confirmation as well as noise-only accuracy, command effort, means, saturation,
task failure/censoring and own-sham incremental pulse recovery. Deduplicate
shams reused for the two pulse signs. Compare controls against group weakening
on identical matched model cohorts. Positive weak-minus-native RMS means
weakening harms control; positive response suppression alone does not imply a
beneficial dynamical role. No new asymptotic-stability or damping claim follows
from finite-horizon accuracy/recovery alone.

## Reporting and interpretation

Produce discrete log-axis 4×4 maps for collective suppression, change during
learning, group size, sensitivity, interactions, frequency dependence and
closed-loop usefulness. Show three-model seed ranges, valid denominators,
empty versus undefined groups, failed matches and complete task failures.
Missing required numeric outcomes invalidate their aggregate; do not average
only surviving rollouts. Matched-control comparisons explicitly condition on
their recorded eligibility. Environmental tapes are shared across models/cells.

This follow-up is exploratory and uses pretrained models whose earlier
behavior is already known. Fresh probes provide confirmation of newly frozen
assays, not an independent replication of training. A decrease in a scalar
amount can coexist with more useful or more selective organization. Neither
mixed-sign weights nor this aggregate establish cortical E/I balance.
