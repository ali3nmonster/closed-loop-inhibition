# Small causal-transformer control pilot

**Status: completed feasibility pilot, 7 October 2026.** This milestone implements step 4 of [START_HERE.md](START_HERE.md), following the [classical timing contract](EXPERIMENT_CONTRACT.md). It asks whether a small transformer can imitate a causal delay-aware controller and maintain useful control on held-out episodes. It does not test inhibition, E/I balance, emergence under different training demands, or a transformer advantage. Numerical settings are pilot choices, not a confirmation preregistration. The executed [configuration](../configs/transformer_pilot.json), [result manifests](../results/transformer_pilot/) and [measured report](../results/transformer_pilot/README.md) document the completed run.

## 1. Scope and experimental unit

Use the existing exactly propagated, normalized mass–spring–damper plant with `tau = 0.5 s`, damping ratio `0.15`, and symmetric actuator limit `5`. Plant dynamics remain fixed in this milestone. Run six-second episodes. Sweep prescribed computation durations of `0`, `0.05`, `0.1`, and `0.2 s` in both serial and fixed-cadence schedules. This varies computation duration relative to one plant timescale; it does not establish generalization to faster plants or separate effects of changed plant mechanics.

Training and evaluation use immutable observation snapshots and the existing event ordering. Sensor capture and fixed-cadence dispatch both occur every `0.05 s`; sensor transport and actuator delays are zero. Closed-loop evaluation reports the trajectory every `0.01 s` without changing sensing or decision timing. The plant evolves throughout each modeled computation interval. Host execution happens separately from the virtual physical clock. Fixed-cadence concurrent inference is an idealized throughput control; the serial arm changes both latency and decision cadence. Save realized decision/application intervals and observation ages rather than equating sensor frequency with fresh-feedback frequency.

The units of replication are independent trained models and independent exogenous episode specifications. The initial target is three training seeds per architecture, `11`, `22`, and `33`. Three seeds are a feasibility check, not a power analysis or sufficient evidence for a general architectural advantage. Multiple test episodes from the same trained model are repeated measurements, not independent model seeds.

## 2. Shared information boundary

Both learned architectures receive the same bounded observation history, observation ages, known action information, and scheduled application time. Use a physical history duration of `0.5 s`, sampled every `0.05 s`, giving at most 11 observations including both endpoints. Valid tokens run from oldest to newest, followed by right padding with an explicit validity mask. Padding is not a real zero-valued observation.

Each token uses the following ten features in this order, where `start` is dispatch time and `apply` is the scheduled action-application time:

1. Captured position `q`.
2. Captured velocity multiplied by `tau`.
3. Captured reference `r`.
4. Captured reference velocity multiplied by `tau`.
5. Captured applied action divided by `5`.
6. Measurement age, `(start - capture) / tau`.
7. Observation transport delay, `(available - capture) / tau`.
8. Time until application, `(apply - start) / tau`.
9. Latest known applied action divided by `5`.
10. Age of that action record, `(start - last_applied_time) / tau`.

The last three features are shared across tokens within a snapshot. The fixed physical scales use no validation or test statistics. Measurement age differs from the additional time until the generated command will apply. Absolute time must not provide an episode identifier or reveal a future reference or disturbance event.

The teacher uses the same bounded information, including only the latest known applied-action record rather than an unlimited applied-action log. Neither teacher nor student can inspect current unobserved state, pending commands, future observations, or future reference and disturbance events. Include explicit tests for this boundary. A command already applied at the dispatch timestamp can be known even when the latest captured observation precedes it; event ordering must remain consistent with the simulator contract.

## 3. Teacher and its limitations

Use predictor PD with the fixed plant model and the existing baseline gains. Starting from the latest available captured state, reconstruct the permitted known command transition, then hold the last known command through the planned application time. Assume zero unknown disturbance and locally constant reference velocity during this prediction. Clip teacher targets to the same action limit applied to all policies.

The predictor is an approximate causal teacher. Overlapping jobs can apply pending commands during computation, and those commands are intentionally outside the shared information boundary. Unknown disturbances and reference changes can also invalidate its prediction. Teacher competence must therefore be measured separately for every timing condition; its name is not evidence of optimality or stability.

The teacher has an explicit model of the fixed plant. The learners acquire an approximation from data generated on that plant. This is a prior-knowledge asymmetry despite their matched runtime information. Comparisons assess imitation and closed-loop behavior, not the superiority of learning over model-based control. The original unlimited-applied-history predictor is not an equally informed comparison unless its interface is restricted as specified here.

## 4. Architectures and objective

The transformer baseline uses two pre-normalized causal attention blocks, width 64, four heads, feedforward width 128, GELU activations and no dropout. Learned position embeddings represent token order alongside the physical timing features. Read out an unbounded scalar from the final valid token; the policy adapter restores physical action units and the simulator applies the common action limit. Causal masking must prohibit later tokens from influencing earlier token representations. Padding must not introduce attention leakage or invalid outputs during cold start.

Compare against a similarly sized MLP receiving the same ordered history, feature values, padding information, and timing information. A memoryless MLP would be an information-confounded control. The configured transformer has 68,545 parameters and the MLP 69,057, approximately 0.7% more. Verify these counts in the run manifest. Record optimization budgets and measured inference runtimes; approximate parameter matching is not runtime matching.

Train both policies by mean squared error against normalized, clipped teacher commands. Normalizing by the common action limit makes the loss interpretable across architectures. Save the normalization in every checkpoint so deployment cannot silently apply a different scale. Use AdamW with learning rate `1e-3`, weight decay `1e-4`, batch size `256`, and gradient-norm clipping at `1`. Train for at most 50 epochs, stopping after 10 epochs without improved validation imitation loss. Restore the checkpoint with the lowest validation normalized-action MSE.

Check held-out closed-loop competence on validation scenarios before opening final test results. This is a readiness check; the stated checkpoint-selection rule remains validation imitation loss. Any changed selection rule or extra training requires a versioned configuration and a new report of the selection procedure. Do not select using final test control scores.

This objective is supervised imitation on collected snapshots. Evaluating the learned policy in closed loop does not turn its training into reinforcement learning or optimization through the physical feedback loop. Native suppression learned under direct control pressure remains a future question.

## 5. Episode generation and splitting

Generate randomized initial states and exogenous reference/disturbance schedules independently of controller decisions. Initial position and velocity are sampled independently and uniformly from `[-1, 1]`. Every scenario includes a disturbance pulse with amplitude drawn uniformly from `[-0.6, 0.6]`, onset drawn uniformly from `[1.5, 2.0] s`, and duration `0.25 s`. Scenario indices divisible by three have reference zero throughout; other scenarios change reference at `1.0`, `3.0` and `4.5 s`, with independent values drawn uniformly from `[-1, 1]`. These mixed scenarios combine initial recovery, disturbances and reference tracking. Preserve complete episode specifications and replay the same test episode across controllers and timing conditions for paired comparisons.

Allocate independent episode identifiers and exogenous realizations to training, validation and test before collecting snapshots. Split whole episodes, never randomly divide neighboring trajectory rows. All timing variants of the same exogenous episode belong to the same split. Different random realizations of the same task family establish held-out episodes within that family, not generalization to unseen disturbance families or plant parameters. Any later claim of family generalization requires withholding entire families.

The initial configuration uses 32 training, eight validation and 12 test exogenous scenarios, generated from split seeds `104729`, `130363` and `155921`, respectively. Each scenario is replayed across two schedules and four delays, producing 256 training, 64 validation and 96 test episodes per evaluated controller. The learned policies use training seeds `11`, `22` and `33`. Paired timing variants are related measurements, not eight independent exogenous scenarios.

Training and validation losses use an unweighted mean across snapshot rows. Conditions generating more decisions therefore receive greater influence, and serial long-delay episodes contribute fewer samples. Equal numbers of episodes per condition do not imply equal loss weight. Report per-condition evaluation separately; no episode- or condition-balancing scheme is used in this pilot.

Derive randomness deterministically from the full declared episode specification or stable split-specific seed streams. Do not use a process-randomized hash or silently reuse a scenario under another split name. Save split membership and audit disjoint identifiers and exogenous specifications. Hyperparameter tuning and any decision to add training data must use training/validation evidence. If test outcomes motivate a revision, label the existing test set exploratory and reserve a fresh test set for subsequent confirmation.

Teacher-generated trajectories may undersample states encountered by an imperfect learner. Separate held-out imitation loss from closed-loop performance: good loss on teacher trajectories does not establish recovery from the learner's own mistakes. If needed, make a later, explicitly versioned data-aggregation pass using training scenarios only. Do not hide such a change within the original pilot.

## 6. Closed-loop evaluation

Evaluate every trained checkpoint, the shared-information predictor teacher, and ordinary PD on identical held-out episode specifications for each schedule and computation duration. Apply the same actuator limit, signal schedules, reporting grid, cold-start convention and stopping rules. The primary feasibility endpoint is held-out closed-loop tracking across the mixed scenarios, reported together with action effort and completion/censoring status. Whole-episode tracking scores and settling after the final signal change do not isolate disturbance-pulse recovery; that requires a separately specified event-window metric.

Retain the existing metric definitions: tracking RMSE and ISE, peak absolute tracking error, exact held-action effort, action variation, and finite-horizon settling. Distinguish unset settling time from zero settling time. Report score units, episode duration and the realized timing. Use raw episode/model results as well as summaries so averaging does not conceal a failed seed or timing condition.

Plot fixed-cadence and serial results in separate panels. Any plotted band across three model seeds describes their range, not a confidence interval or a power calculation. Preserve paired scenario scores and report censored counts alongside summaries.

Numerically censored episodes cannot contribute deceptively low partial-horizon error integrals to a complete-episode comparison. Report their count, censoring reason and observed horizon separately. Completion means that the simulator reached the horizon; bounded or completed episodes are not automatically successful control. No operational failure probability or delay margin is claimed until a meaningful task threshold is specified independently.

A fully observed linear plant can be controlled using compact sufficient state and an approximate predictor. History attention may therefore offer no advantage over the history MLP. Matching or losing to the MLP is compatible with a useful baseline and must not trigger unreported task changes. The pilot should establish a usable controller and measurement pipeline, not create evidence for attention by construction.

## 7. Runtime and provenance

Measure isolated policy-call runtimes after warmup, with explicit device synchronization when required. The initial run uses CPU inference, two PyTorch threads and PyTorch `2.5.1+cpu` in an isolated environment. Record hardware, device, software versions, batch size, precision, thread settings and history length. Report median and 95th-percentile single-call runtime rather than only a single best time. These timings quantify implementation cost; the experiment's prescribed virtual durations are not measured processor latency.

Save resolved configuration, episode/split manifests, training seeds, parameter counts, training and validation loss traces, checkpoint selection epoch, checkpoints or their content hashes, per-episode evaluation metrics, and source/dependency fingerprints. Record source state accurately if the run precedes a commit. Results must link to the configuration that actually produced them. Keep large artifacts in the repository's ignored run directory and retain compact provenance and summaries in the tracked results directory.

Use the [CPU installation and staged execution commands](../README.md#run-locally). `--stage train` collects demonstrations, trains all configured models and writes closed-loop validation results without evaluating test control. Inspect those results before running `--stage evaluate`. The evaluation stage checks the configuration and hashes of source, saved data, checkpoints and recorded training outputs against the training manifest. Keep source and configuration unchanged between stages and use new output/artifact directories for a new run; an existing completed training manifest prevents accidental overwrite.

## 8. Acceptance and next decision

Accept the implementation when causality and masking checks pass, split generation and seeded replay are reproducible, a saved checkpoint reproduces its predictions, and learned policies complete useful held-out control across a nontrivial part of the timing grid. Evaluate competence relative to the measured teacher and PD trajectories, with effort and censoring visible. Do not invent a numerical success threshold after inspecting the result.

If imitation is accurate but closed-loop behavior fails, investigate distribution shift, teacher behavior under pending actions, and timing features before expanding the network. If both neural baselines fail, the protocol is not ready for mechanism discovery. If the teacher fails at a delay, do not interpret a learner's failure there as transformer-specific.

The subsequent milestone can freeze an operational competence threshold and identify candidate suppressive pathways on a separate discovery set. Intervention selection and confirmation require independent data. This pilot alone supplies no causal suppression result, no evidence of cortical E/I balance, and no test of the central suppression-by-temporal-demand interaction.
