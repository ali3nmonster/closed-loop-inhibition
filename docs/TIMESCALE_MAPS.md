# Two-timescale maps at fixed computation delay

**Design proposal, 8 October 2026. No new training or map experiment has been run.** This proposal follows the [validated dynamics setup](../results/dynamics_validation/README.md). It fixes the feedback interface and maps performance and learned suppressive organization against plant and perturbation timescales. It is not yet a frozen training/confirmation protocol: the competence baseline, training objective and diagnostic selection rules must be specified before launch.

## Axes and fixed conditions

Use fixed-cadence execution with observation and dispatch period `h = 50 ms` and computation delay `L = 50 ms`. One update of latency is interpretable and already makes the fastest validated plants challenging. It is a prescribed virtual delay, not a claim about the transformer forward-pass runtime. The schedule assumes enough throughput for the declared dispatch cadence.

| Plot component | Initial grid or setting |
|---|---|
| Horizontal axis: plant tau | 50, 100, 200, 500 ms |
| Vertical axis: noise correlation time | 10, 50, 200, 1000 ms |
| Axis scale | Logarithmic, with discrete tested cells clearly marked |
| Noise amplitude | Stationary marginal standard deviation 0.02 in expectation |
| Damping ratio | 0.15 |
| Physical forcing clock | 2.5 ms sampled-and-held input |
| Reference | Zero target for the first regulation study |

Keep update period fixed as well as latency. Annotate ratios `h/tau_plant`, `L/tau_plant` and `tau_noise/tau_plant`; the diagonal `tau_noise = tau_plant` helps read the temporal relationship. `tau_plant` remains inverse natural angular frequency, not the oscillation period. Do not interpolate a four-by-four pilot into apparently measured boundaries.

Keep force disturbances and position-measurement noise in separate map sets. A staged start can train on force disturbances first and then repeat the design for sensor noise; combining their outcomes would mix different input paths. The current position-noise model leaves velocity measurements exact. At 50 ms sampling, noise with a 10 ms correlation time is nearly uncorrelated across captures; this does not mean the controller resolves 10 ms fluctuations. Physical force still acts between observations.

Reuse paired exogenous tapes across models and interventions. Preserve the actual mean/RMS of each draw rather than rescaling every realization. Fixed marginal variance controls expected input power; changes in spectral distribution, predictability and the physical transfer path remain part of the manipulation.

## Learned emergence versus context-dependent use

For the emergence question, train an independent controller ensemble for each timescale pair, with the same architecture, initialization-seed set, information boundary, training sample budget and optimization budget. Three seeds on a four-by-four grid give 48 models per noise channel. This is an exploratory first map, not enough replication for precise region boundaries.

Save initialization, intermediate checkpoints and final checkpoints. A final pattern alone does not establish that it emerged through training. Compare change from initialization and record when suppression diagnostics and competent behavior appear. Training conditions may produce different learned temporal filters, which is an outcome rather than a reason to normalize away differences after training.

A separately useful experiment trains one shared controller across the grid and evaluates it cell by cell. Its map measures context-dependent use and generalization of a shared organization. It does not by itself show that each environment caused a different learned organization. If used, state whether plant/noise parameters are supplied or must be inferred from history. Keep these two questions distinct.

## Map families

| Map | Operational meaning | Important comparison |
|---|---|---|
| Control accuracy | True-position RMS over a fixed declared evaluation window | Passive behavior and a competent controller with matched observations |
| Cost and failure | Applied-command effort, saturation, task-failure and numerical-censoring rates | Accuracy at a useful effort level; visible failed cells |
| Functional suppressive contribution | Change in disturbance-evoked command response when a preselected pathway is weakened or strengthened | Identical fixed histories, matched small probes and initialization |
| Contribution cancellation | Opposing contributions under a fixed, explicit head/pathway decomposition | Descriptive diagnostic; depends on the decomposition and chosen readout |
| Gain modulation | How pathway intervention changes response sensitivity across prescribed probe amplitudes or backgrounds | Generic output-gain control and alternative-pathway controls |
| Temporal effect | Intervention-induced changes in measured response gain, phase, decay or oscillation | Full controller–plant loop with unchanged input/timing protocol |
| Causal usefulness | Change in true-state accuracy, effort and failure after intervening on the candidate suppression | Own-intervention sham, mean correction and matched-effect controls |

These are distinct measurements. More functional suppression need not improve control, and lower command magnitude need not imply better damping. Attention competition, negative weights or cancellation alone do not establish a biological excitation/inhibition balance. Prefer quantities tied to physical outputs or explicit interventions over arbitrary activation-coordinate signs.

For causal maps, define the sign explicitly, for example `Delta RMSE = RMSE_weakened - RMSE_native`: positive values mean weakening harms accuracy, while negative values mean it improves accuracy. Show effort and failure changes alongside this map. Use a diverging scale centered on zero for intervention effects, a shared scale across comparable accuracy panels, and seed-level uncertainty or ranges. Mark failed or undefined cells explicitly; do not replace them with zero or average only successful episodes.

## Probes and data separation

Different noise spectra change which dynamics are excited. Comparing raw activity statistics between cells therefore cannot alone identify different controller mechanisms. Use both each cell's operating-distribution probes and a predeclared common diagnostic probe family, and label their interpretations separately.

Keep the direct controller assay separate from deployment: fixed-history assays feed native and modified policies the same saved histories to measure immediate output effects; closed-loop assays let each policy act on its own evolving history. Their trajectories can diverge even with identical exogenous noise. The latter measures the combined controller–environment consequence.

For paired probes, compare the same background-noise tape with and without a small added physical disturbance under the same controller/intervention. Use the same tape again across native and modified policies. Freeze how probe amplitude and duration are defined in physical or normalized units. The choice of fixed physical duration versus a duration scaled with plant tau answers different questions and must not change opportunistically across cells.

Retain discovery, calibration and confirmation separation. Select pathways using discovery data; fit command-centering offsets and matched intervention strengths on calibration data; evaluate final maps on fresh confirmation noise. Selecting a different winning head in each cell requires its own held-out validation and must not be confused with tracking one fixed pathway. Report both prevalence of eligible suppression under a fixed selection rule and effect magnitude conditional on that rule, including cells with no eligible pathway.

Constant command correction preserves only the chosen calibration mean. Continue measuring actual closed-loop baseline drift and held-action means. Use matched output gain and alternative-pathway controls to distinguish a pathway effect from a general response-strength change. The previous small-intervention study provides methods, not selected heads transferable to newly trained models.

## Competence before neural maps

At `L = h = 50 ms`, the existing fixed-gain PD reference is unstable for plant tau 100 and 50 ms. That is a property of this reference controller, not proof that these conditions are uncontrollable. Establish a competent delay-aware classical baseline on the proposed grid before interpreting learned failures or selecting imitation targets.

The expert must receive the same captured noisy observations, timing information and allowed command history as the learned controller. Plant-model knowledge is a declared prior; future disturbances, clean hidden measurements or pending-command access unavailable to the learner must not become privileged targets. Use fixed documented feature scaling, including velocity units, across speeds.

Freeze the training objective before training. Teacher imitation maps architecture-specific realizations of a prescribed policy; direct closed-loop optimization additionally tests organization learned for the task objective. A causal claim about suppression should survive competence, effort and matched-control checks under the chosen training regime. The first useful result can be a clear negative map.

## Implementation order

1. Build fixed-delay classical accuracy/effort/failure maps over the complete two-axis grid and verify a suitable competence reference.
2. Freeze the per-cell neural training objective, budgets, information boundary, seeds, success criteria and independent data streams.
3. Train the first channel's controller ensembles and produce performance maps before interpreting inhibition diagnostics.
4. Apply frozen discovery/calibration/confirmation procedures and create separate suppression-strength and causal-usefulness maps.
5. Compare initialization/training trajectories, replicate across seeds, then repeat for the other noise channel. Refine the spatial grid only as a new exploratory or confirmation stage.

The first artifact should make the experiment's two axes and failure regions visible. Maps of inhibition emergence require the new neural training and causal measurements; the existing classical validation cannot supply them.
