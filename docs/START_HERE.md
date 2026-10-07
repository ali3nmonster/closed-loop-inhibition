# Starting the research program

**Status:** research plan, 7 October 2026. Steps 1–3 have an [implemented timing contract](EXPERIMENT_CONTRACT.md) and a [36-rollout classical validation report](../results/baseline/README.md). Step 4 has a completed transformer/MLP teacher-imitation pilot: three training seeds per architecture, 768 held-out rollouts and 108 tests at that milestone. See the [protocol](TRANSFORMER_PILOT.md) and [neural results](../results/transformer_pilot/README.md). The step 5 update below records the subsequent causal experiment. Implemented contracts and configurations supersede illustrative settings below.

**Step 5 update:** the [first suppression protocol](SUPPRESSION_PILOT.md) and [2,432-rollout causal experiment](../results/suppression_pilot/README.md) are complete; the current suite has 188 passing tests. Functional suppression replicated, but head scaling also shifted the operating point and controls did not establish a consistent beneficial temporal mechanism. The next revision should isolate these effects with new calibration and confirmation data. Direct closed-loop training and explicit E/I architectures remain future work.

## Research question and first claim to test

Do emergent or explicitly designed suppressive mechanisms in transformers improve the stability–responsiveness tradeoff of the combined controller–environment system? Does their causal contribution depend on environmental timescales, observation frequency and computation delay?

The first experiment should establish whether a measurable interaction exists between **a suppressive mechanism** and **temporal demands**. It should not attempt to establish cortical E/I balance from negative weights or successful control. Functional suppression, opposing population contributions and a dynamically balanced regime are different claims.

Start with an ordinary transformer and causal interventions. An explicit E/I-inspired architecture is a subsequent comparison, not a prerequisite for an informative result. See the [research background and proposal](RESEARCH_PROPOSAL.md) for the evidence and hypotheses.

## Step 1 Freeze the experimental contract before implementing a network

Use a fully observed, one-dimensional mass–spring–damper system:

\[
\dot q=v,\qquad
\dot v=\frac{-q-2\zeta\tau v+u+d(t)}{\tau^2}.
\]

Here \(\tau\) controls the plant's characteristic timescale, \(\zeta\) its damping ratio, \(u\) the applied command and \(d(t)\) an external disturbance. This normalized plant keeps static input-to-position gain fixed while changing speed. Begin with \(\tau=0.5\) s and \(\zeta=0.15\), then validate a small timescale sweep. Give the controller position, velocity, a reference trajectory and the history of applied commands. Exclude vision initially. “Fully observed” means full state at observation capture: both policy and teacher must use the delayed timestamped information available through the interface, without access to the current simulator state or future reference values unavailable to the tested policy.

Use three task families: recovery from a disturbance around a fixed reference; tracking a smooth reference; and recovery from randomized initial conditions. Keep disturbance amplitude and spectral content specified in physical units, and include a separate condition in which their timescales scale with the plant. These answer different questions and must not be silently mixed.

Predeclare the primary endpoint: the **within-controller change in normalized disturbance-recovery error caused by attenuating a candidate suppressive pathway**, and how that change varies with computation delay relative to \(\tau\). Analyze failure probability separately; a bounded controller can remain numerically bounded while failing the task.

**Deliverable:** an experiment specification recording units, task distributions, endpoints, failure thresholds and what information the policy receives. **Pass criterion:** another reader can reconstruct every controller input and score without guessing.

## Step 2 Implement a world that advances during computation

The first simulator should use a deterministic virtual clock. Model computation as a scheduled interval rather than making simulator progress depend on the workstation's incidental runtime. During that interval, the plant continues evolving under the last applied command. Deliver the result only at its scheduled completion time.

For the first policy, permit one inference job at a time. A job consumes a snapshot of observations available at its start; observations arriving while it runs enter a buffer for the next job. At completion, apply the new command and start the next eligible job according to an explicit dispatch rule. This deliberately simple baseline has no action chunking or incremental feedback within a forward pass.

In this serial schedule, increasing computation duration also reduces the effective rate of fresh policy decisions. Fixed sensor sampling alone does not isolate latency. Before confirmation, add a virtual fixed-cadence schedule: start an independent history-snapshot query every fixed interval and deliver each result after the assigned constant delay. Multiple queries may be pending, so latency changes while decision cadence stays fixed. This is a mechanism-isolation control with idealized throughput, not a claim that one processor can sustain it. Use the stateless history-window policy for this arm; any shared mutable cache would require an explicit concurrency rule. Compare both schedules, and report single-job results as a combined latency-and-cadence manipulation.

Record at least:

- Observation capture, availability and policy-selection timestamps.
- Inference start and completion timestamps.
- Command issuance and physical application timestamps.
- Applied action, reference, plant state and disturbance events.
- Buffer contents, dropped observations and controller reset events.

Specify deterministic ordering for simultaneous events. A sensor event at the same timestamp as a new inference start must either be included or excluded consistently. Separate observation sampling, computation delay and any actuator delay in the configuration. Do not call the actuator command rate the fresh-feedback rate.

Use exact linear propagation between piecewise-constant input events, or verify a numerical integrator against that solution. Define disturbances by their own clock, so changing the controller update frequency does not change the physical noise process.

**Deliverable:** replayable trajectories and a timeline plot demonstrating that the plant moves during an inference job. **Pass criterion:** changing virtual delay changes observation age without altering the underlying disturbance realization or introducing future information.

## Step 3 Establish analytical and classical control checks

Before training, implement proportional–derivative control and a delay-aware classical comparator. For the normalized plant, the continuous, zero-delay law \(u=-k_pq-k_d\tau v\) has characteristic polynomial

\[
\tau^2s^2+(2\zeta+k_d)\tau s+(1+k_p).
\]

Use its roots as a limiting check. For actual sampled control with zero-order hold, use the exact discretization \(A_d,B_d\): the zero-delay closed-loop matrix is \(A_d-B_dK\). For integer-step delays, augment the state with the corresponding command or observation history. Do not compare a sampled implementation directly with continuous-feedback roots and call a disagreement a simulator defect.

Check that simulated transients agree with the appropriate linear calculation, and that bounded perturbations produce the predicted local response before saturation. Include actuator limits, then mark which conclusions apply only near the equilibrium. Tune the classical controller on validation plants under the same action constraints; add a predictor or small MPC baseline if ordinary PD leaves an obvious delay-compensation advantage unused.

**Deliverable:** analytical-versus-simulation validation plots and classical performance curves. **Pass criterion:** numerical error is small relative to the intervention effect the experiment intends to detect, and expected stable and unstable cases are distinguished correctly.

## Step 4 Train the smallest useful transformer baseline

The initial implementation is specified in [TRANSFORMER_PILOT.md](TRANSFORMER_PILOT.md) and [configs/transformer_pilot.json](../configs/transformer_pilot.json). It uses a two-block causal transformer and a similarly sized MLP with the same 0.5-second history, three training seeds, independent episode splits, two schedules and four virtual computation delays. The plant and data family are fixed. Follow the [CPU installation and staged run instructions](../README.md#run-locally): train and inspect closed-loop validation before final test evaluation. Evaluation checks code, configuration, dataset and checkpoint integrity against the training manifest.

Begin with a small causal transformer policy: for example, two blocks, four heads and width 64. Treat these as pilot settings. Each history token contains timestamped observations, the reference, applied action and observation age. Use causal masking and normalized physical quantities. Record the normalization because apparent gains depend on units.

Start with supervised imitation of a competent delay-aware teacher to debug optimization and evaluation. Such a pilot establishes feasibility, not that suppression emerges from autonomous interaction. Then choose and document a closed-loop training objective, using differentiable simulation or a standard reinforcement-learning implementation as justified by the pilot. Evaluate all learned policies through the same event-driven interface.

Include a similarly sized feedforward or recurrent policy to assess whether a finding requires attention. Match the available history and information; a memoryless MLP is not an adequate architectural control for a transformer with a long history.

Use two distinct studies:

1. **Emergence:** train independent models under slow versus demanding timing distributions and compare the prevalence and causal strength of suppressive pathways.
2. **Functional role:** freeze trained models, vary deployment timing and intervene on their existing pathways.

Vary computation delay first while holding observation sampling and history duration fixed. Use the fixed-cadence control from milestone 2 to isolate delay, then quantify the combined deployment effect in the serial schedule. Later sweeps of sampling frequency must preserve or explicitly control history duration in seconds. Fixed token count otherwise creates a memory-duration confound. Timestamp embeddings and masked variable-length histories are possible solutions, with their token and runtime costs recorded.

**Deliverable:** reproducible training configurations and held-out control curves. **Pass criterion:** multiple independent seeds learn competent behavior across a useful, nontrivial timing range. Interpretability work should not begin with controllers that already fail throughout the range.

## Step 5 Identify suppression and test its causal role

The initial implementation and qualified outcome are recorded in [SUPPRESSION_PILOT.md](SUPPRESSION_PILOT.md) and its [result report](../results/suppression_pilot/README.md). The broader design below remains the roadmap; the first study deliberately focused on eight attention heads, paired pulse/sham responses and three frozen models.

Use a discovery set to identify head or MLP contributions that suppress a specified behaviorally relevant signal. State which signal is suppressed: for example, a disturbance-induced action transient, a distractor feature or an internally amplified response. Negative weight signs alone are insufficient, and a negative action can be appropriate control rather than an inhibitory mechanism.

Predefine an operational identification rule and check it across held-out states. A candidate can be state dependent; do not force a single excitatory/inhibitory identity on every unit. Residual contributions expressed in the same action-relevant direction offer a starting point, but projection evidence must be followed by interventions.

Scale candidate contributions gradually, for example by factors 0, 0.5, 1 and 1.5. Include:

- Random-pathway interventions matched for activation magnitude and location.
- Interventions matched for their immediate effect on output magnitude.
- Global action-gain changes as a simpler alternative explanation.
- Restoration of the original contribution as a rescue check.

Apply the same disturbance seeds to paired interventions. Examine short transients as well as whole episodes, because long interventions can move the policy far from its training distribution. Reserve independent data for selecting candidates, selecting intervention settings and confirming the final effect.

**Deliverable:** intervention-by-delay response curves with uncertainty across training seeds. **Pass criterion for a positive result:** the temporal interaction survives matched controls and held-out confirmation. **Valid negative outcome:** suppression is identifiable but does not improve the tested stability–responsiveness tradeoff.

## Step 6 Add one explicit mechanism with matched controls

Choose one interpretable modification after the baseline works. A first candidate could constrain a context-dependent gate to attenuate a predefined feature group, or introduce nonnegative E/I activities with specified sign-constrained outgoing effects. Fix the feature orientation and intervention target before evaluating benefit. An unrestricted branch with a minus sign is insufficient: the sign can be absorbed into its learned output weights, leaving an equivalent model class. Document the substantive constraint and check for such reparameterizations. Call the resulting mechanism suppressive or E/I-inspired; do not call it an E/I-balanced network without further measurements.

Compare it with an unconstrained branch of the same size, a standard branch with comparable parameter count and a simple gain-control baseline. If the proposed design adds persistent state or iterative refinement, give the controls equivalent state or refinement opportunities. Match or report optimization budget, parameter count, wall-clock latency and inference schedule. Equal FLOPs alone do not guarantee equal latency.

Evaluate both fixed virtual latency, which isolates functional effects, and measured latency, which exposes implementation cost. Conduct a strength sweep: excessive or delayed negative feedback can worsen control. A nonmonotonic optimum is compatible with the hypothesis.

**Deliverable:** one mechanism ablation with matched alternatives. **Pass criterion for claiming an architectural benefit:** an improvement beyond generic gain reduction, extra memory, extra parameters or extra computation.

## Step 7 Analyze the combined system and decide whether to scale

Report disturbance amplification, overshoot, recovery error, settling time, tracking error, action effort, action variation, failure probability and operational delay margin. Define the settling band and observation horizon before inspecting confirmation results. Plot tradeoffs: slower, weaker actions can reduce oscillations while making the controller less useful.

Estimate local modes only where the assumptions permit it. The augmented state includes the plant, controller memory, history buffer, pending observations and pending commands. For a differentiable periodic schedule, linearize the complete sampled transition or its one-period map. State which equilibrium, trajectory and perturbation size were used. For variable schedules, start with empirical impulse responses; do not present an arbitrary transformer-layer Jacobian as the physical closed-loop spectrum.

Keep distinct held-out splits for controller training seeds, plant parameters and disturbance families. If several mechanisms are screened, select on discovery/validation data and confirm the selected hypothesis with independent runs. Episodes from one trained network are not independent evidence of an architectural effect.

Scale to a nonlinear pendulum only after confirming the measurement pipeline. Next consider fresh observations during iterative action generation, following the timing questions raised by [RTC](https://arxiv.org/abs/2506.07339), [REMAC](https://arxiv.org/abs/2601.20130) and [πR²](https://arxiv.org/abs/2607.26055). Physical hardware is a later validation stage, with appropriate actuator bounds and emergency-stop procedures; it is not required for the first causal result.

**Deliverable:** a decision report identifying which hypotheses survived. **Stop or revise if:** effects vanish under output-gain matching, no suppression can be identified reproducibly, benefits depend on one seed, or ordinary delay compensation explains the result. These outcomes narrow the claim and should be recorded, not hidden.

## Suggested initial configuration

This is an illustrative research configuration, superseded for the implemented milestones by [baseline.json](../configs/baseline.json) and [transformer_pilot.json](../configs/transformer_pilot.json). In particular, the neural pilot uses a 0.05-second sensing/decision interval, a 0.5-second history, an action limit of 5 and no inhibition intervention. Observation interval and computation duration are independent; the delay sweep represents computation duration, not an extra delay added on top of measured inference.

```yaml
plant:
  kind: normalized_mass_spring_damper
  tau_seconds: 0.5
  damping_ratio: 0.15
  actuator_limit: null  # Must be selected from pilot trajectories before confirmation.
timing:
  mode: virtual_event_driven
  observation_interval_seconds: 0.01
  compute_duration_over_tau: [0.02, 0.05, 0.1, 0.2, 0.4]
  actuator_delay_seconds: 0.0
  concurrent_inference_jobs: 1
  # This pilot serial schedule changes both decision cadence and observation age.
  action_hold: zero_order
  dispatch: start_on_new_observation_when_idle_otherwise_latest_at_completion
policy:
  kind: causal_transformer
  blocks: 2
  heads: 4
  width: 64
  history_seconds: 1.0
  include_timestamps: true
intervention:
  contribution_scales: [0.0, 0.5, 1.0, 1.5]
evaluation:
  paired_disturbance_realizations: true
  separate_discovery_validation_confirmation: true
  primary: attenuation_by_delay_interaction_in_recovery_error
```

The initial delay grid may miss the informative regime. Use pilot classical and learned-controller curves to locate it, then freeze a confirmation grid and the fixed-cadence comparison schedule before testing the primary hypothesis. The chosen grid is not evidence that a transition occurs at any particular ratio.

## First working day and proposed code organization

The first day should produce a reviewed experimental contract, the event-driven plant, its analytical checks and a plot showing one delayed observation-to-action cycle. Do not spend it designing a large architecture. Next implement PD and the teacher, then estimate training cost from a small pilot before allocating a full sweep.

Implementation layout, with planned modules marked:

```text
src/closed_loop_inhibition/
  plants.py              # Dynamics and exact linear propagation.
  timing.py              # Event queue, observations, inference and application.
  controllers.py         # Implemented PD/predictor and analytical calculations.
  imitation.py           # Shared teacher/student information and demonstrations.
  neural.py              # Causal transformer and matched-history MLP.
  interventions.py       # Implemented native head scaling and residual diagnostics.
  suppression_tasks.py   # Paired pulse/sham tasks and absolute competence safeguards.
  suppression_analysis.py # Discovery statistics and matched-effect calibration.
  metrics.py             # Task, transient and local-dynamics measurements.
configs/                 # Pilot and frozen confirmation configurations.
experiments/             # Baseline, staged training and staged suppression runners.
tests/                   # Timing causality, analytical dynamics and replay checks.
results/                 # Manifests and small summaries; large data kept elsewhere.
```

Estimate compute from measured pilot training time, evaluation time, seed variance and the actual factorial design. Then set the number of independent training seeds and confirmation runs from the desired precision or detectable interaction, recording the decision before confirmation. No compute budget or statistical power is established yet.

The first successful deliverable is a trustworthy causal measurement pipeline. A robust negative result from that pipeline is more informative than an architectural improvement whose source cannot be identified.
