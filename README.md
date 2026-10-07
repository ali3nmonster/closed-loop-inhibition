# Suppressive organization in transformer feedback control

Research project investigating whether emergent suppression and explicit excitation–inhibition inspired mechanisms improve the stability and responsiveness of transformer controllers when the environment evolves during computation.

**Central question:** Do suppressive mechanisms change the combined controller–environment dynamics beneficially, and does their causal contribution depend on the relationship between environmental timescales, observation-to-action latency, and fresh-feedback intervals?

**Status on 7 October 2026:** Research background and experimental design. No simulator, trained models, or experimental results are included yet.

## Read first

1. [Research background and proposal](docs/RESEARCH_PROPOSAL.md) — the complete conceptual argument, evidence, definitions, hypotheses, and intended contribution.
2. [Step by step starting plan](docs/START_HERE.md) — the first experiment, milestones, controls, measurements, and decision criteria.
3. [Neuroscience evidence](docs/notes/neuroscience_evidence.md) — cortical computation, population geometry, inhibition, and embodied dynamics.
4. [Transformer evidence](docs/notes/transformer_evidence.md) — recurrence, geometry, native suppression, explicit mechanisms, and real-time control.
5. [Source and provenance guide](docs/SOURCES.md) — primary references, version cautions, and connections to existing local research.

## First implementation milestone

Build an event-driven continuous-time oscillator with timestamped observations, inference completion, and action application. Validate its delayed-feedback behavior using a classical controller before training a transformer. Then establish a competent ordinary transformer baseline and test identified suppressive pathways through graded interventions and rescue.

The primary outcome is the **stability–responsiveness tradeoff**, including disturbance recovery and delay tolerance at useful tracking performance. A general reward increase, smaller actions, negative weights, or a static cancellation score alone would not establish the proposed mechanism.

The initial repository contains documentation only. Suggested implementation paths and configurations in the starting plan describe future work.
