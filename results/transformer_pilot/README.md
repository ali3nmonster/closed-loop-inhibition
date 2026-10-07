# Trained transformer control pilot

Completed 7 October 2026. A two-block causal transformer and a matched-history MLP were trained to imitate a causal predictor-PD teacher. Each architecture has three independent training seeds. This establishes a small learned-controller baseline; no suppressive pathway or E/I-inspired mechanism was tested.

## Design and execution

The fixed oscillator has time constant 0.5 s and damping ratio 0.15. Both learned models and the teacher receive the same bounded, timestamped history: at most 11 observations over 0.5 s, plus the latest applied command and known time until action application. The teacher additionally has an explicit model of the plant. All actions share a limit of ±5.

Training used 32 independent exogenous scenarios (256 timing variants, 26,176 snapshots); validation used eight scenarios (64 variants, 6,544 snapshots). Twelve held-out test scenarios were paired across four prescribed computation durations and two scheduling rules, giving 96 episodes per controller and 768 total test rollouts. Episodes last six seconds. The same task family and plant parameters occur in every split.

The objective is unweighted normalized-action MSE on teacher trajectories. Checkpoints were selected by validation imitation loss, and validation feedback performance was inspected before test evaluation. No hyperparameter revision or additional training followed that inspection. Fast conditions have more decision snapshots and therefore more loss weight: each training condition contributes 3,872 rows except serial 0.1 s (1,952) and serial 0.2 s (992). Closed-loop summaries below weight episodes equally.

Sensing and fixed-cadence dispatch use 0.05 s intervals; control metrics use a 0.01 s reporting grid. Sensor and actuator transport delays are zero. Computation durations are virtual: 0, 0.05, 0.1 and 0.2 s, or 0–0.4 times the plant time constant. Serial inference also changes decision cadence; fixed-cadence inference assumes idealized concurrent throughput. These are distinct experimental conditions.

## Results

Both learned controllers closely matched the teacher: aggregate tracking RMSE differed by about 0.18% for the transformer and 0.39% for the MLP. At 0.2 s serial computation, transformer/MLP RMSE remained near 0.31 while ordinary PD reached 0.74. This demonstrates successful transfer of the teacher's delay-aware behavior on the tested episodes; it does not establish an architectural advantage or a suppressive mechanism.

![Validation learning and held-out control](pilot_results.png)

Lines show means across the 12 test scenarios and, for neural models, three training seeds. Shaded bands show the minimum and maximum of the three seed means, not confidence intervals. Timing variants of one scenario are paired observations. The plot omits censored episodes; the counts below make any such omissions explicit.

| Controller | Test rollouts | Censored | Mean tracking RMSE | Mean action effort | Mean action variation |
|---|---:|---:|---:|---:|---:|
| teacher | 96 | 0 | 0.2624 | 2.9453 | 12.0387 |
| pd | 96 | 0 | 0.3333 | 7.2064 | 17.3925 |
| transformer | 288 | 0 | 0.2629 | 2.9435 | 11.9234 |
| mlp | 288 | 0 | 0.2635 | 2.9642 | 12.0543 |

Tracking RMSE uses normalized position units; action effort is the exact integral of held action squared over time. Action variation is the sum of absolute applied-command changes. These are mixed-scenario tracking scores combining initial recovery, reference changes and a disturbance pulse; they do not isolate pulse recovery or establish a formal stability margin.

| Schedule | Computation (s) | Teacher RMSE | PD RMSE | Transformer RMSE | MLP RMSE |
|---|---:|---:|---:|---:|---:|
| serial | 0.00 | 0.2332 | 0.2332 | 0.2334 | 0.2330 |
| serial | 0.05 | 0.2509 | 0.2512 | 0.2514 | 0.2511 |
| serial | 0.10 | 0.2671 | 0.2855 | 0.2681 | 0.2690 |
| serial | 0.20 | 0.3060 | 0.7420 | 0.3067 | 0.3094 |
| fixed_cadence | 0.00 | 0.2332 | 0.2332 | 0.2334 | 0.2330 |
| fixed_cadence | 0.05 | 0.2509 | 0.2512 | 0.2514 | 0.2511 |
| fixed_cadence | 0.10 | 0.2662 | 0.2756 | 0.2668 | 0.2677 |
| fixed_cadence | 0.20 | 0.2920 | 0.3947 | 0.2923 | 0.2934 |

All individual episode results and seed-specific metrics are retained in [test_rollouts.csv](test_rollouts.csv); [condition_summary.csv](condition_summary.csv) also records effort, variation and censoring by condition. Sampled state bounds and horizon completion alone do not define successful control.

A representative reporting-grid check halved the interval from 0.01 to 0.005 s on validation scenario `validation-001`, using the teacher and transformer seed 11 in all eight timing conditions. Maximum relative ISE change was 0.83%; applied-command differences were below 9e-15, with no censoring. This is a convergence diagnostic, not an error bound over the entire test set; small differences between neural architectures should not be overinterpreted. See [convergence.json](convergence.json).

## Training and runtime

| Model | Seed | Parameters | Selected epoch / run | Validation action RMSE | Test action RMSE | Forward median / p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| transformer | 11 | 68,545 | 27 / 37 | 0.0146 | 0.0189 | 0.603 / 0.611 |
| transformer | 22 | 68,545 | 19 / 29 | 0.0157 | 0.0234 | 0.610 / 0.617 |
| transformer | 33 | 68,545 | 25 / 35 | 0.0096 | 0.0167 | 0.605 / 0.615 |
| mlp | 11 | 69,057 | 25 / 35 | 0.0185 | 0.0228 | 0.128 / 0.140 |
| mlp | 22 | 69,057 | 18 / 28 | 0.0193 | 0.0244 | 0.128 / 0.138 |
| mlp | 33 | 69,057 | 18 / 28 | 0.0193 | 0.0253 | 0.128 / 0.137 |

Action RMSE is measured on teacher-generated snapshots, in physical action units. It is distinct from closed-loop tracking RMSE. The MLP has about 0.7% more parameters; parameter counts are approximately matched, while execution cost is not. Both use the same bounded history and timing features.

Training and inference used float32 on an AMD EPYC 7513 CPU, with two PyTorch threads and one interop thread. Runtime probes used batch size one and 11 valid tokens, 20 warmup calls and 100 measured calls. They measure forward passes only, excluding encoding and physical I/O, on a shared host without CPU pinning. Virtual experiment latency was prescribed independently of these measurements. See [hardware.json](hardware.json).

## Interpretation and next experiment

The teacher comparison assesses whether imitation preserves useful feedback behavior. A similarly performing MLP is expected to be plausible on this fully observed, fixed, linear plant; attention is not necessary to represent its compact control law. Three seeds and 12 exogenous test scenarios support a feasibility report, not a general architectural ranking.

This pilot has no direct closed-loop learning objective, functional suppression measurement, suppressive intervention, explicit E/I architecture, or cortical balance test. Neither negative weights nor good delay tolerance would establish those mechanisms. The next milestone should define a competence criterion on separate development scenarios, identify candidate suppressive contributions, and compare causal interventions with matched random/strength controls across the two timing schedules. Any outcome-driven changes need a fresh confirmation set.

## Reproduction and provenance

Use the installation and two-stage commands in the [repository README](../../README.md). The [protocol](../../docs/TRANSFORMER_PILOT.md) explains information boundaries and limitations. The full [configuration](config.json) and [episode specifications](episodes.json) preserve the splits and paired signals. [training_history.csv](training_history.csv), [training_summary.json](training_summary.json), [validation_rollouts.csv](validation_rollouts.csv), [validation_readiness.json](validation_readiness.json) and [test_imitation.csv](test_imitation.csv) preserve selection and evaluation evidence. The full validation suite passed 108 tests.

Model checkpoints and training/validation datasets are retained locally under ignored `runs/transformer_pilot/`. SHA-256 digests are recorded in [training_manifest.json](training_manifest.json). The runner rejects evaluation if code, configuration, selected checkpoints or recorded training artifacts have changed. [manifest.json](manifest.json) records dependency versions and the pre-commit source fingerprint; a dirty working tree is expected because this implementation was evaluated before its commit. Compact results are tracked, while model weights can be regenerated by the seeded run.
