# Transformer background for suppression and physical feedback

Research note, 7 October 2026. This document records the transformer-specific background to the project. It distinguishes published findings, their interpretation, and the proposed research contribution. Literature coverage is targeted rather than exhaustive; absence of an identified paper is not proof of priority.

## Attention and hierarchy

For input representations X, an ordinary attention head forms Q = XW_Q, K = XW_K and V = XW_V. A query determines what information a receiving position requests; keys determine which source positions match; values determine what information those sources contribute. These are computational roles imposed by the architecture. Their particular features are learned jointly from the task loss through backpropagation; there is no separate teacher specifying the correct queries, keys or values.

This separates content-dependent addressing from the payload being transmitted. Stacked heads can compose retrieval and transformation across positions, while residual connections retain an accessible accumulated representation. Attention is one useful inductive bias, combined with optimization, data and parallel computation; its formula alone does not explain all of transformer success. [Vaswani et al., Attention Is All You Need, 2017](https://arxiv.org/abs/1706.03762).

Transformers are hierarchical in the ordinary sense of composing successive transformations. A standard language transformer does not impose the same progressively coarsening spatial hierarchy as a conventional pooling CNN. Nor does a stack of distinct layers automatically provide recurrent physical-time dynamics. These distinctions motivated looking beyond superficial resemblance to cortical anatomy.

Although softmax attention coefficients are nonnegative, the values and output projections can write signed vectors. Heads and MLPs can therefore suppress a target representation without negative attention coefficients. Conversely, subtracting two attention maps does not by itself create biological inhibitory populations.

## Exact recurrence and its state

Two results answer different versions of “are transformers RNNs?”

- **Compact recurrence for linear attention.** Katharopoulos et al. express causal finite-feature kernel attention through recurrent sufficient statistics. The exact compact construction applies to that attention mechanism. It should not be reported as an exact finite-dimensional replacement for arbitrary ordinary softmax attention. [Transformers are RNNs, ICML 2020](https://proceedings.mlr.press/v119/katharopoulos20a.html).
- **Growing-state recurrence for ordinary causal transformers.** Oren et al. describe the per-layer key/value history as a multi-state recurrent memory. The exact state grows as tokens arrive. Their later compression to a bounded number of states is a separate operation. This formulation applies to externally supplied sequences as well as autoregressive generation; sampled-token feedback is not necessary for the recurrent representation. [Transformers are Multi-State RNNs, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.1043/).

Thus, the obstacle is not whether an exact recurrence exists. The question is what its update structure actually does. Appending a KV entry while retaining older entries does not establish mutually adjusting excitatory and inhibitory populations, relaxation to an attractor, or a biologically balanced regime. A cache can preserve information that the model subsequently fails to retrieve.

For this project, declare the controller state explicitly: fixed-length observation/action history, KV cache if used, any recurrent refinement state, and pending actions. A history-window transformer remains a dynamical controller because its history changes across interactions. Layer index, sequence index, refinement iteration and wall-clock time must remain distinct.

## Levels of representation geometry

The following objects should not be conflated:

| Object | Samples used | Interpretation |
|---|---|---|
| Representation geometry across inputs | Many examples at a fixed layer and token convention | How a population of inputs is embedded |
| Geometry within a sequence | Token representations from one sequence at a fixed layer | Structure of that particular contextualized token set |
| Propagation through depth | Representations after successive layers | A computational trajectory through different transformations |
| Controller trajectory | Memory/history and neural representations across physical feedback updates | State evolution during interaction with the environment |

There is empirical precedent for the first two. Valeriani et al. measure intrinsic dimension and neighborhood structure across transformer layers, finding expansion followed by contraction in several studied models. This does not establish one universal low-dimensional manifold for every task or model. [The geometry of hidden representations of large transformer models, NeurIPS 2023](https://arxiv.org/abs/2302.00294).

Sharma and Kaplan explicitly compare GPT activations from a **single 1,024-token sequence** with samples across sequences, reporting much lower estimated intrinsic dimension for the former (Figure 10). This answers the narrow question of whether within-sequence geometry has been studied. It does not identify a physical-time attractor or prove that the full recurrent cache is low-dimensional. [Scaling Laws from the Data Manifold Dimension, JMLR 2022](https://jmlr.org/papers/volume23/20-1111/20-1111.pdf).

Cowsik et al. study randomly initialized transformers as interacting token representations propagating through depth and derive geometric contraction/expansion and Lyapunov exponents. This is relevant mathematical machinery, but its depth dynamics are not measured sensorimotor dynamics of a trained controller. [Geometric Dynamics of Signal Propagation Predict Trainability of Transformers, 2024](https://arxiv.org/abs/2403.02579).

A single smooth trajectory is locally curve-like regardless of ambient state dimension; an attractor can nevertheless have a higher-dimensional closure. Therefore “one rollout looks low-dimensional” is insufficient. Estimate shared structure over multiple initial conditions and disturbances, test held-out reconstruction and prediction, and compare perturbations along and transverse to the candidate subspace. Dimensionality is an optional mechanistic analysis here, not a required premise of the inhibition hypothesis.

## Functional suppression in ordinary transformers

The strongest evidence is causal and target-specific:

- **Copy suppression:** McDougall et al. identify GPT-2 attention heads that counteract earlier copying predictions. Suppression changes when upstream copying is removed. This is a concrete learned opposing-pathway motif. [BlackboxNLP 2024](https://aclanthology.org/2024.blackboxnlp-1.22/).
- **Compensation and self-repair:** McGrath et al. demonstrate downstream compensation after component ablation and downregulation of likely output tokens. Rushing and Nanda examine mechanisms including changed normalization and reduced erasure. A component's direct contribution can consequently differ from its total causal importance. These are feedforward compensation findings, not proof of a recurrent inhibition-stabilized circuit. [Hydra Effect, 2023](https://arxiv.org/abs/2307.15771); [Explorations of Self-Repair, ICML 2024](https://arxiv.org/abs/2402.15390).
- **Internal feature inhibition:** Anthropic's Claude 3.5 Haiku case studies identify known-answer/entity features that suppress inability-to-answer features, with interventions affecting answering and abstention. The discovery procedure uses an approximate replacement model and the authors report limitations, including large intervention strengths and incomplete circuit explanations. [On the Biology of a Large Language Model, 2025](https://transformer-circuits.pub/2025/attribution-graphs/biology.html#hallucinations).
- **Open-model knowledge awareness:** Ferrando et al. identify known/unknown-entity features with causal effects on behavior in Gemma and Llama. This supports investigating internal features rather than only final output subtraction. [Do I Know This Entity?, ICLR 2025](https://arxiv.org/html/2411.14257v2).
- **Context-dependent signs:** Gerstner and Schütze analyze gated neurons that enrich or deplete representation directions; a neuron's functional effect can change with its gate. A fixed assignment of every ordinary transformer neuron to an excitatory or inhibitory class is therefore unjustified. [Understanding Gated Neurons, 2025](https://arxiv.org/html/2505.17936v1).

These observations establish suppression, compensation and disinhibition in particular models and tasks. They do not establish that all transformers operate in a cortical-style balanced regime, or that the same pathways stabilize embodied feedback.

A negative weight is basis-dependent. Softmax output competition can lower one probability merely because another increased. Layer normalization can create apparent cancellation and alter gain. Discovery should therefore name a task-relevant target, choose its positive orientation, quantify actual pathway contributions and validate them by interventions in the original model.

An opposition index such as (E − I)/(E + I + ε), where E and I sum positive and negative contributions to a named target, is descriptive. Its value depends on decomposition and grouping. Near-zero net drive under negligible total drive is not evidence of balance. Even strong cancellation occurs in random signed sums. If two decompositions produce the identical complete state, deterministic downstream computation cannot distinguish their labels; mechanistic significance requires different recruitment or perturbation responses.

## Explicit inhibitory designs

**Differential Transformer** subtracts two softmax attention maps with a learned coefficient, enabling cancellation at the attention-map level. Its language-model results establish a useful architectural precedent. They neither demonstrate E/I population balance nor test physical closed-loop damping. [Ye et al., ICLR 2025](https://arxiv.org/abs/2410.05258).

**Controlled Dynamics Attractor Transformer (CDAT)** combines attention and Hopfield refinement energies with continuous-attractor-inspired excitation–inhibition modulation and a dissipation analysis. Its evaluated tasks are **graph anomaly detection and graph classification**. “Controlled dynamics” in this title refers to internal inference; the paper is not evidence of stabilization of a robot–environment loop. [Zhang et al., ICML 2026](https://proceedings.mlr.press/v306/zhang26ea.html).

**SpiLiFormer** provides a further explicit lateral-inhibition example in spiking transformers. It belongs in the prior-art boundary, although a nonspiking project need not adopt its implementation. [Zheng et al., ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Zheng_SpiLiFormer_Enhancing_Spiking_Transformers_with_Lateral_Inhibition_ICCV_2025_paper.pdf).

Consequently, “add inhibition to a transformer” is not an adequate novelty claim. Nor can success or limited adoption of these variants decide whether ordinary transformers already learn all useful suppression. Task, training budget, hardware efficiency and the temporal deployment regime are alternative explanations. This project makes no claim that all explicit inhibitory designs have failed or that adoption establishes their biological interpretation.

## Suppressive attention in control tasks

The **ICLR 2025 Robot Learning workshop version** of Kachaev et al., *A New Perspective on Transformers in Online Reinforcement Learning for Continuous Control*, compares Differential Attention with ordinary attention (Figures 11 and 18). It reports no consistent advantage across its tested control conditions. This is a bounded negative result, not a universal verdict. It does not isolate observation latency or measure damping of joint physical/neural modes. Cite [this workshop PDF](https://openreview.net/pdf?id=nfh9PSaASy): the later expanded arXiv manuscript is not an interchangeable source for that comparison. Version and access details are recorded in the [source guide](../SOURCES.md).

**AmpAttention/RVAF** uses differential-amplifier-inspired subtraction for task-guided multi-view robot perception. It reports simulation and physical manipulation results; arXiv metadata lists acceptance at IROS 2026. Its signal-to-noise and amplifier-gain interpretation concerns representations, not a demonstrated feedback stability margin. Dataset recording rates should not be reported as fresh-observation inference rates. [Yang et al., July 2026 manuscript](https://arxiv.org/abs/2607.02845).

Together these papers establish that subtractive attention in embodied tasks is already explored. They do not complete the causal argument connecting inhibition, relative timescales and joint-system stabilization.

## Fast transformer feedback

“Continuous control” often means continuous-valued actions at discrete simulation steps. If the simulator waits for inference, it does not test the world-changing-during-computation issue. Three more direct precedents are:

| Study | Relevant evidence | Boundary |
|---|---|---|
| **RTC**, NeurIPS 2025 | Generates the next action chunk while executing the current one; commits an action prefix and inpaints the remainder. Evaluates dynamic simulation tasks, physical manipulation and inference delay. | Addresses scheduling and continuity, not inhibitory circuitry. |
| **REMAC**, ICLR 2026 | Trains for discrepancies between current perception and already committed actions. A real robot uses 15 Hz commands with roughly 122–140 ms end-to-end inference; additional delays of 75/150 ms are injected. | Measures task outcomes and trajectory smoothness, without identifying E/I control of joint modes. |
| **πR²**, July 2026 preprint, under review | Refreshes proprioceptive conditioning at successive denoising calls while vision–language features update asynchronously; approximately 25 Hz replanning on the reported platform. | Fresh proprioception does not mean equally fresh vision; denoising-call updates do not imply interruptible updates inside every transformer layer. |

Primary sources: [RTC](https://arxiv.org/abs/2506.07339v2), [REMAC](https://arxiv.org/html/2601.20130), [πR²](https://arxiv.org/abs/2607.26055).

These results support the premise that temporal embedding matters. They do not yet show that suppressive organization is what improves delay tolerance. High actuator-command frequency also does not imply equally frequent incorporation of fresh observations.

A separate 2026 preprint, [Stable Transformer-Actor-Critic Model Predictive Control](https://arxiv.org/html/2606.20197v3), attempts a contraction/small-gain treatment of a transformer–MPC–plant system. It is not used here as proof of generic transformer closed-loop stability: assumptions, constrained components and the transition from layer-indexed representations to physical-time state require careful independent checking. It contains no direct inhibition-by-delay test.

## The proposed causal and temporal link

The defensible research gap is a causal interaction: **suppressive organization changes joint controller–environment dynamics, and its contribution depends on environmental speed relative to computation and feedback**. None of the inspected studies establishes this entire conjunction for ordinary transformers or a matched E/I-inspired variant.

Native-mechanism discovery and architectural comparison should be separate experiments. An explicit component could help without reproducing emergent native mechanisms; conversely, native suppression could be necessary while an additional component adds nothing. Removing suppression may impair perception or simply increase action magnitude, so a decline in task reward alone cannot identify dynamical stabilization.

The distinctive evidence would combine pathway-specific perturbation, timing sweeps, matched controls, restoration/rescue and measured disturbance responses. It should test both useful responsiveness and stability, allowing an optimal inhibitory strength and timing rather than assuming more inhibition is always better. Delayed negative feedback can itself produce oscillations. These are project hypotheses and proposed discriminating tests, not findings attributed to the papers above.
