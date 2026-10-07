# Neuroscience background for suppression and embodied dynamics

Research note checked on 7 October 2026. This document distinguishes established observations, model-dependent interpretations and hypotheses to transfer into transformer research. It complements the repository's main research proposal; it does not claim that transformers reproduce cortical physiology.

## Cortical microcircuit computation

The useful starting point is a family of overlapping computational hypotheses. A recurring anatomical motif need not implement one algorithm everywhere. Prediction, normalization, selection, recurrent inference and feedback regulation can describe different aspects of the same circuit. Their relevance to this project is that suppression may organize a computation's dynamics, rather than merely remove unwanted features.

| Hypothesis | Computational interpretation | Implication for this project |
| --- | --- | --- |
| Predictive coding | Compare sensory evidence with predictions and propagate discrepancies. Rao and Ballard's hierarchical visual model reproduces several receptive-field effects through feedforward residuals and feedback predictions. | Prediction cancellation is a possible suppressive function, but observing cancellation would not establish an E/I-balanced dynamical regime. [Rao & Ballard, 1999](https://doi.org/10.1038/4580) |
| Normalization and gain control | Responses depend on both the preferred input and a broader suppressive pool. Heeger's model uses mutual inhibition to account for contrast-dependent response normalization. | Suppression may regulate sensitivity to input magnitude. A divisive operation alone does not identify its circuit implementation or establish closed-loop stability. [Heeger, 1992](https://www.cns.nyu.edu/heegerlab/content/publications/Heeger-VisNeurosci1992a.pdf) |
| Recurrent integration and selection | Population dynamics combine evidence accumulation with selection of task-relevant inputs. A trained RNN reproduces important features of monkey prefrontal activity during context-dependent decisions. | Internal modes and input-dependent routing can matter more than individual neuron labels. [Mante et al., 2013](https://www.nature.com/articles/nature12742) |
| Attractor-based decisions and memory | Recurrent excitation and competition can maintain evidence and produce a decision. Wang's cortical circuit model links slow reverberation to probabilistic choice. | Inhibition may shape competition and transition times without being primarily a noise filter. [Wang, 2002](https://pubmed.ncbi.nlm.nih.gov/12467598/) |
| Dynamical regulation | Coupled excitatory and inhibitory populations can support stable states, hysteresis and oscillations. | The sign, strength and timescale of feedback jointly determine behavior; inhibition is not automatically synonymous with damping. [Wilson & Cowan, 1972](https://pubmed.ncbi.nlm.nih.gov/4332108/) |

**The paper that prompted this part of the discussion:** Balwani, Cho and Choi's *Exploring the Architectural Biases of the Cortical Microcircuit* appeared in *Neural Computation* 37(9), 1551–1599 (2025). Its anatomically constrained RNN models compare feedback architectures and training objectives. Feedback and interareal delay can make expected and unexpected inputs distinguishable at initialization; predictive-coding-inspired training further changes representation and specialization. The important qualification is that architectural bias and imposed learning objective both contribute. This is evidence about a model's inductive biases, not proof that all cortical circuitry implements predictive coding. The earlier author manuscript includes “Canonical” in its title. [Published DOI](https://doi.org/10.1162/neco.a.23), [author manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC11142214/).

For architecture design, these alternatives motivate separate interventions: subtract a predicted signal, normalize population gain, stabilize recurrent amplification, or gate communication. Putting them all into an “inhibition” category would obscure what an experiment actually tests.

## Generality of cortical population manifolds

Population geometry is a useful language, but the strongest claim is conditional: particular tasks and measurements often reveal relatively low-dimensional, behaviorally relevant structure. It does not follow that all cortical activity lies on one small manifold.

In a closed-loop brain–computer interface, Sadtler and colleagues changed the mapping from monkey motor-cortical activity to cursor movement. Perturbations compatible with the population's existing low-dimensional activity structure were easier to learn than mappings requiring activity outside that structure on the experimental timescale. This links geometry to learning constraints, rather than merely providing a visualization. [Sadtler et al., 2014](https://www.nature.com/articles/nature13665).

Prefrontal cortex illustrates why “low-dimensional” needs a reference set. Mante and colleagues described task-relevant population dynamics that support context-dependent integration. In a different prefrontal task, Rigotti and colleagues connected nonlinear mixed selectivity to high-dimensional representations and flexible readout. These are compatible: a high-dimensional representational repertoire can contain lower-dimensional trajectories or task-specific readout subspaces. [Mante et al., 2013](https://www.nature.com/articles/nature12742), [Rigotti et al., 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC4412347/).

Large recordings also challenge a universal low-dimensional sensory picture. Stringer and colleagues measured responses to thousands of natural images in mouse visual cortex and found a high-dimensional representation with a slowly decaying variance spectrum. Their analysis relates the spectrum to smoothness of the neural code. Expanding the stimulus ensemble can expose structure missed by a restricted task. [Stringer et al., 2019](https://www.nature.com/articles/s41586-019-1346-5).

For this project, distinguish at least four objects:

1. **Geometry across inputs:** the cloud of representations produced by different observations or histories.
2. **Geometry within a rollout:** the controller's state trajectory as it interacts with one environment.
3. **Locally accessible dynamics:** directions reached by perturbations around an operating state or trajectory.
4. **Joint geometry:** the controller, physical state, observation history and action queues considered together.

A visually simple trajectory need not imply a small accessible state space. Any smooth finite trajectory is locally a curve; its causal relevance requires perturbations, repeated conditions and generalization tests. Likewise, variance explained by a few principal components is not proof that those components contain every behaviorally important direction. In transformers, depth, token position and physical time must remain separate indices before making a comparison with cortical trajectories.

## Inhibition balance and stabilization

Use a hierarchy of operational definitions:

- **Functional suppression:** changing one pathway causally reduces a specified downstream signal, action component or response to an input. Define what is reduced and under which conditions.
- **Opposing contributions:** separately identifiable pathways make counteracting contributions to a specified variable. Large opposing contributions are stronger evidence than a negative parameter alone.
- **E/I balance:** substantial excitation and inhibition approximately cancel at an explicitly defined level and timescale. Global averages, neuron-specific cancellation and stimulus-specific cancellation are distinct possibilities.
- **Inhibition stabilization:** an excitatory subsystem would be unstable without inhibitory feedback, while the coupled network is stable in the tested regime.
- **Joint-loop stabilization:** the controller–body–environment system recovers from disturbances or tolerates delay better because of an identified mechanism.

These definitions do not imply one another. In particular, network activity can remain bounded while the controlled body oscillates or diverges. Conversely, the body and its feedback can change neural dynamics, so testing the controller in isolation can miss its operative regime.

Van Vreeswijk and Sompolinsky's balanced-network theory shows how large, strongly coupled excitatory and inhibitory populations can generate irregular activity while responding rapidly to external input. The original model also exhibits chaotic dynamics. Thus “balanced” does not mean internally quiescent, globally contracting or necessarily well suited to controlling a delayed plant. [van Vreeswijk & Sompolinsky, 1996](https://pubmed.ncbi.nlm.nih.gov/8939866/).

Sanzeni and colleagues tested predictions of inhibition-stabilized networks, including paradoxical suppression of inhibitory firing following stimulation of inhibitory neurons, in mouse visual, somatosensory and motor cortex. Their results provide causal evidence consistent with inhibition stabilization across those sampled areas and conditions. They do not establish universal cortical balance or show how that regime stabilizes every behavioral feedback loop. [Sanzeni et al., 2020](https://elifesciences.org/articles/54875).

For transformers, a biological cell-type constraint is one possible design choice, but the first measurement should be functional. Negative weights, positive attention coefficients, signed value vectors, subtractive branches and normalization have different meanings. Choose an observable and a causal intervention before importing a physiological label.

## Neural mechanisms in embodied feedback

### Presynaptic inhibition controls movement smoothness

Fink and colleagues manipulated spinal GABAergic interneurons that regulate transmission from sensory afferents. Ablation severely disrupted reaching and uncovered stereotyped forelimb oscillations. A model with excessive sensory-feedback gain captured core properties of the movement disturbance. This is strong evidence that a specific inhibitory circuit supports smooth behavior by regulating sensory–motor transmission. It concerns spinal presynaptic inhibition, not cortical E/I balance; the experiment did not sweep environmental speed against neural computation time. Nevertheless, it provides an especially direct biological precedent for the project's gain-and-damping hypothesis. [Fink et al., 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4292914/).

### The environment itself changes neural gain

Buckley and Toyoizumi developed a theory in which actions alter subsequent sensory input, thereby forming feedback that changes neuronal gain and coherent fluctuations. Their support combines a rodent whisking model with analysis of zebrafish data. Closed-loop sensory feedback and replayed input need not produce the same response to perturbation, because only the former lets current neural activity alter future sensory input. This directly motivates analyzing the joint system. It does not identify a particular cortical interneuron population as the cause of the effect. [Buckley & Toyoizumi, 2018](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005926).

There is no contradiction with determinism: the same complete input trajectory and initial state would reproduce the same deterministic controller trajectory. Active coupling changes how disturbances propagate and changes the future input trajectory; replay removes that dependency.

### Delays expose a stability–responsiveness tradeoff

Demarchi and colleagues used virtual reality and brain-wide imaging in *Danionella cerebrum*, manipulating sensory feedback during visually guided navigation. They related observed oscillations and adaptation to a delayed sensorimotor model with logarithmic sensory and motor transformations. The work demonstrates that encoding nonlinearities can support adaptive stabilization in the presence of delay. It does not establish inhibition as the underlying cellular mechanism. For our experiments, it is also an alternative explanation: an apparent E/I advantage could arise from the response nonlinearity or effective gain introduced by the component. [Demarchi et al., 2025](https://doi.org/10.1073/pnas.2510385122).

Together, these studies support three different propositions: inhibitory circuits can regulate feedback gain; environmental feedback can alter neural dynamics; and delay can expose benefits of nonlinear gain regulation. Connecting all three specifically inside a transformer remains an experimental task.

## Bridges to nonspiking artificial controllers

Neural Circuit Architectural Priors (NCAP) transfer structured *C. elegans*-inspired circuitry into a differentiable controller for a simulated swimmer. Bhattasali and colleagues report strong data efficiency, small parameter counts and transfer to altered body designs; ablations implicate constrained excitation/inhibition. This is direct prior art for useful sign-constrained architecture in embodied ANN control. It does not test transformers, isolate computation–environment timescale overlap, or establish a universal superiority of E/I constraints. Initialization and structural constraints must also be disentangled in a new comparison. [Bhattasali, Zador & Engel, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/52e431bd7689d98426300cb103bb0ee3-Abstract-Conference.html).

The transfer principle does not require spikes. Continuous-valued populations can express recurrent excitation, suppressive coupling and different response times. Wilson–Cowan dynamics already supply a theoretical example, including regimes with oscillations rather than stabilization. A rate-based E/I component is therefore a legitimate experimental choice, but a successful result must be compared with an unconstrained component that has the same state, update schedule and computational budget.

## Implications for the transformer hypothesis

The neuroscience supports a plausible research hypothesis: suppressive organization might reshape the gain, timing and disturbance response of an embodied transformer, with effects that depend on relative timescales. It does not predict that stronger suppression always helps, that biological balance must emerge, or that a new architecture is necessary.

The most informative translation is a causal experiment. Identify or introduce a suppressive pathway; alter its strength or timing; independently vary environmental dynamics, inference latency and sensory refresh; measure both responsiveness and stability. Compare against nonselective attenuation, action scaling, normalization and equally stateful unconstrained controllers. A positive result should demonstrate more than lower action magnitude or slower behavior.

Population geometry can then explain which directions are damped, preserved or amplified. It should accompany disturbance-response measurements rather than substitute for them. Evidence of an E/I-like regime would be a further result with its own operational tests. The primary claim can remain narrower and useful: a particular suppressive mechanism causally improves the combined system's stability–responsiveness tradeoff under specified temporal demands.
