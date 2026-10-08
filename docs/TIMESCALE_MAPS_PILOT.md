# Fixed-delay timescale maps: first force-disturbance experiment

Prospective implementation protocol, 8 October 2026. This instantiates the
[map design](TIMESCALE_MAPS.md). Configuration is in
[`configs/timescale_maps.json`](../configs/timescale_maps.json). Sources,
configuration and this document are fingerprinted before the scientific run.
The exploratory grid is not a significance test or a survey of architectures.

## Physical task and reference

Hold both dispatch/observation period and virtual computation delay at 50 ms.
Use fixed cadence, zero sensor/actuator delay, zero reference, damping 0.15,
plant tau 50/100/200/500 ms and OU force correlation time 10/50/200/1000 ms.
The OU stationary marginal standard deviation is 0.02, with exact stationary
samples held on an independent 2.5 ms clock. Initial evaluation state is zero.
Actuator limit is 0.5; the event-time state guard is 100. Reporting interval is
10 ms. Plant tau is inverse natural angular frequency, not its oscillation period.
This first channel is physical force noise; no sensor noise is added.

Compare passive zero command and a delay-aware predictor-PD reference with
kp=2, kd=1. The reference receives the same causal bounded snapshot as the
learner: at most 11 observations spanning 0.5 s and the latest applied command.
It predicts to application under that command, without future force or pending
command access. At this one-step delay the hold assumption is exact apart from
unknown forcing. Its plant knowledge is a declared model-based prior. A separate
12 s competence pilot (seed 730001) improved RMS over passive in all 16 cells;
the ratios ranged from 0.288 to 0.967. This pilot is not confirmation data.

## Training

Train independent ordinary causal transformers for every cell, seeds 11/22/33:
48 models. Each has two blocks, four attention heads per block, width 32,
feedforward width 64, GELU, pre-normalization, learned positions and a scalar
final-token readout. No explicit inhibitory or E/I constraint is introduced.

Use the existing causal ten-feature encoder. Divide physical position,
reference and tau-scaled velocity features by 0.1, and express action features
and outputs in units of 0.1. Timing features retain their declared tau scaling.
Scaling is fixed prospectively, not fitted to evaluation data.

Per cell collect 16 independent 4 s expert demonstrations and six validation
episodes. Each starts with independent normal q and tau*v, standard deviation
0.05, then receives that cell's stationary OU force. Train and validation random
streams start at 710001 and 720001; episode indices specify subsequent seeds.
Reuse data across initialization seeds, and pair random draws across timing
cells. This common random-number design is not independent environmental
replication across cells. Save generated data and their checksums.

Train all models for 50 epochs using AdamW (lr 0.001, weight decay 0.0001),
batch size 128 and gradient clipping 1. Select the checkpoint with the lowest
supervised validation MSE. Save initialization, epoch 25, selected and final
checkpoints; retain all epoch losses and selection epochs. Equal optimization
budgets do not imply equal competence. Do not select checkpoints or retrain
using closed-loop confirmation results.

The objective is expert imitation. Results concern functional organization in
learned implementations of that policy, not inhibition emerging under direct
optimization of closed-loop task reward. Supervised loss alone is not competence.

## Performance confirmation

Evaluate each model and both references on the same four fresh 12 s OU tapes
(seeds 810001–810004), scoring [3,12] s. Record true position RMS and mean,
applied-command RMS and mean, saturation time, maximum position, numerical
censoring and task failure. An RMS above 0.1 or numerical censoring is a declared
absolute task failure; compare also with passive and teacher performance because
this threshold alone does not establish useful control. Censored RMS values are
undefined and remain visible, never replaced by zero or silently omitted.
This is a finite-horizon task from rest: the 3 s burn-in is shorter than one
passive decay-envelope time for the slowest plant, so RMS is not claimed to be
stationary. As a prespecified reporting-convergence check, repeat the fastest
plant with the shortest/longest noise correlation time, background seed 810001,
for passive, teacher and model seed 11 at 5 ms reporting. Compare position RMS
with 10 ms reporting while leaving forcing, observations and actions unchanged.

## Discovery, calibration and independent diagnostic confirmation

Physical pulse probes have onset 1 s, duration 0.1 s and amplitudes ±0.02,
in 4 s trials. Pair pulse/sham arms on identical background OU tapes. The direct
controller assay uses native pulse/sham histories over decision times [1,4);
both models see identical saved inputs. Closed-loop interventions act on their
own histories and are a different measurement.

Use discovery seeds 910001/910002, calibration seeds 920001/920002 and
confirmation seeds 930001–930004. For each of eight heads, weaken its projected
residual contribution by scaling output-projection columns to 0.9. Eligibility
requires every physical baseline pulse-minus-sham command RMS to exceed 1e-4,
a median fractional response increase above 1%, and positive increases in at
least 75% of seed/amplitude groups. Choose the eligible head with greatest
median increase; deterministic layer/head order breaks ties. Report eligible
head counts and candidate prevalence even when no head qualifies. Missing
candidate effects are undefined, not zero. No discovery-based pathway identity
is inherited from earlier experiments.

Separately assay all eight heads in initialization, epoch-25 and selected
checkpoints on a common bank of exact teacher histories with zero background
noise and pulse amplitudes ±0.01/±0.02. This bank is identical across training
noise timescales at fixed plant tau. Track eligible fractions unconditionally
and the trained-selected pathway conditionally. This tests training-associated
changes in functional contribution on controlled inputs, not random-initialized
closed-loop behavior. A pathway chosen after training is subject to selection
bias; its initial-to-trained trajectory alone is not an unbiased emergence rate.

Fit constant offsets for the weak (0.9) and strong (1.1) interventions on native
calibration sham histories, preserving clipped mean command there. Measure the
actual confirmation drift separately. Choose an alternative head in the same
layer by nearest projected residual RMS on calibration histories. Match its
scale (0.5 to 1.5 by 0.01) and a generic output-gain control (1 to 1.5 by 0.005)
to the centered weak intervention's immediate command RMS change across pulse
and sham histories. Gain is centered on the native calibration sham mean;
controls receive their own constant mean correction. Target relative matching
tolerance is 5%; retain and flag failed matches rather than tuning the grid.
Alternative pathways need not have the same functional sign. Persist all
discovery choices and calibration settings before confirmation.

On fresh diagnostic tapes, report raw and centered fixed-history weak/strong
response effects, absolute changes as well as ratios, and native, centered weak,
centered strong, gain-control and alternative-head own-history rollouts. Report
true sham/noise regulation, mean shifts, effort and failures along with paired
pulse-minus-sham recovery. Reusing the same sham for the two pulse signs is
pairing, not extra independent replication. Positive weak-minus-native RMS
means weakening harms accuracy; negative means it helps. A suppression effect
alone is not evidence that suppression helps control.

## Maps and limits

Plot discrete 4×4 cells on logarithmic plant/noise axes, with fixed timing
annotated. Show accuracy, reference ratios, effort/failures, eligible fractions,
held-out suppression strength and causal effects separately. Show seed-level
variation and valid/eligible denominators; preserve missing or failed cells.
Use a zero-centered diverging scale for signed changes. No interpolated region
boundaries or inferential significance claims from three model seeds.

These measurements operationalize suppressive attention pathways. They do not
establish biological E/I balance, neuron types, a benefit from explicit E/I
components, or a next-generation architecture. Fixed computation delay is
virtual, independent of wall-clock inference time. The cadence assumes adequate
throughput. The 12 s finite horizon does not establish asymptotic stability;
4 s pulse assays are short for the slowest plant's passive envelope. The fixed
physical pulse intentionally occupies different fractions of a plant cycle.
