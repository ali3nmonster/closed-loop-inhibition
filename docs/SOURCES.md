# Sources and provenance

Research scope: primary literature relevant to the discussion, reviewed through 7 October 2026. This is a targeted research foundation, not an exhaustive systematic review or a certification of novelty. Evidence descriptions and qualifications accompany the claims in the [proposal](RESEARCH_PROPOSAL.md) and the two evidence notes.

## Foundational reading

| Source | Role in this project |
|---|---|
| [Vaswani et al. 2017, Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Query, key, value projections and transformer architecture |
| [Gu et al. 2020, HiPPO](https://arxiv.org/abs/2008.07669) | Memory as online polynomial approximation |
| [Katharopoulos et al. 2020, Transformers are RNNs](https://proceedings.mlr.press/v119/katharopoulos20a.html) | Compact recurrence for finite-feature linear attention |
| [Oren et al. 2024, Transformers are Multi-State RNNs](https://aclanthology.org/2024.emnlp-main.1043/) | Recurrence with a growing transformer memory state |
| [Balwani, Cho, and Choi 2025, Exploring the Architectural Biases of the Cortical Microcircuit](https://doi.org/10.1162/neco.a.23) | Cortical architectural biases and predictive-coding-inspired modeling |
| [Sanzeni et al. 2020, Inhibition stabilization is a widespread property of cortical networks](https://elifesciences.org/articles/54875) | Recurrent inhibitory stabilization and causal perturbation logic |
| [Fink et al. 2014, Presynaptic inhibition of spinal sensory feedback ensures smooth movement](https://doi.org/10.1038/nature13276) | Inhibition, sensory feedback gain, and physical movement |

## Suppression and intervention methods

| Source | Role in this project |
|---|---|
| [McDougall et al. 2024, Copy Suppression](https://aclanthology.org/2024.blackboxnlp-1.22/) | Identifiable native suppressive attention circuits |
| [McGrath et al. 2023, The Hydra Effect](https://arxiv.org/abs/2307.15771) | Compensatory responses to component ablation |
| [Rushing and Nanda 2024, Explorations of Self-Repair in Language Models](https://arxiv.org/abs/2402.15390) | Direct effects versus downstream compensation |
| [Lindsey et al. 2025, On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) | Feature-level suppression and intervention case studies |
| [Ameisen et al. 2025, Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) | Interpretation methods and approximation limitations |

Additional primary references for geometry, cortical functions, and designed inhibition are linked directly in the [neuroscience](notes/neuroscience_evidence.md) and [transformer](notes/transformer_evidence.md) evidence notes.

## Transformer control studies closest to the question

| Source | Evidence boundary |
|---|---|
| [Black, Galliker, and Levine 2025, Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/abs/2506.07339) | Asynchronous action generation and inference delay |
| [Wang et al. 2026, Real-Time Robot Execution with Masked Action Chunking](https://arxiv.org/abs/2601.20130) | Delay-dependent training and physical deployment |
| [Park and Tulsiani 2026, piR2](https://arxiv.org/abs/2607.26055) | July preprint; fresh proprioception during iterative action generation |
| [Kachaev et al. 2025, A New Perspective on Transformers in Online RL for Continuous Control](https://openreview.net/pdf?id=nfh9PSaASy) | Workshop version contains the Differential Attention comparison |
| [Yang et al. 2026, Differential Amplifier-Inspired AmpAttention for Multi-View Robotic Manipulation](https://arxiv.org/abs/2607.02845) | Subtractive attention for robotic perception and manipulation |

These studies do not by themselves establish the proposed inhibition-specific timing mechanism. Their experiment designs and their remaining gaps serve different roles in this project.

## Version and interpretation cautions

- The Differential Attention comparison belongs to the linked 2025 workshop paper. The later expanded manuscript, arXiv:2510.13367, omits that comparison and should not be cited for the negative result. OpenReview access can return a browser challenge; preserve the exact version identifier when obtaining the PDF.
- AmpAttention's arXiv metadata reports acceptance at IROS 2026. Its July manuscript was described earlier in the discussion as a preprint; publication status should follow the current primary record. Its visual and action recording rates are not verified policy-feedback rates.
- The Controlled Dynamics Attractor Transformer addresses internal iterative dynamics and reports graph tasks. Its use of “dynamics” does not establish physical closed-loop control.
- The 2026 Stable Transformer–Actor–Critic MPC preprint is a relevant lead for joint stability theory, but is not used here as established certification of a general deployed transformer controller. In particular, a proof must connect the physical-time history update to its chosen transformer state; depth-wise contraction alone does not supply that connection.
- Exact recurrence, fixed-dimensional compression, internal stability, physical stability, cancellation, and biological balance are separate claims. References supporting one do not automatically support another.

## Related local research

The following existing files informed the conceptual continuity. They remain in their original repositories and are not external empirical evidence.

| Existing file | Connection |
|---|---|
| [Native LLM E/I balance review](/home/ball/nwm/review/NATIVE_LLM_EI_BALANCE_2026-09-27.md) | Signed contributions, identifiability, balanced-network qualifications, and recurrent formulations |
| [Native transformer dynamic balance protocol](/home/ball/nwm/review/NATIVE_TRANSFORMER_DYNAMIC_BALANCE_PROTOCOL_2026-09-27.md) | Perturbation, compensation, feedback, and claim-level decision criteria |
| [Inverter-Mamba project](/home/ball/iwbm-paper/projects/mamba-inverter/README.md) | Planning over state-space-model dynamics parameters |
| [Mamba on hardware concept](/home/ball/iwbm-paper/concepts/mamba-on-hardware/README.md) | Physical dynamical substrates and selective state-space computation |
| [Memory and neurobiology background](/home/ball/iwbm-paper/concepts/mamba-on-hardware/neurobiology-background.md) | Related HiPPO, temporal reconstruction, and inverse-Laplace discussion |

The memory files are related antecedents; the exact earlier document described as “transformer memory as an inverse problem” was not positively identified. Their speculative biological equivalences and priority claims are not adopted as established findings. The present project asks a different question about suppressive mechanisms in a physically evolving feedback loop.

Local absolute links work on this server and will need adjustment if the repository is moved. The research argument and primary-source links remain usable independently of those local files.

## Maintaining the evidence record

For each future addition, record the exact paper version, publication status, studied system, intervention, measured endpoint, and the strongest claim it supports. Keep computational inference stability distinct from stability of an agent acting on an external world. Update the interpretation when a source is revised or a proposed mechanism receives a direct test.
