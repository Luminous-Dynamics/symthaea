# Synthetic Organism Architecture v0.1

**Status:** RFC / research implementation target  
**Branch:** rfc/synthetic-organism-v0-1  
**Date:** 2026-10-07

## Abstract

Symthaea already contains substantial pieces of an organism-like cognitive architecture:
hyperdimensional representations, liquid dynamics, active inference, global-workspace
mechanisms, episodic/semantic memory, an autopoietic subsystem, interoception,
predictive self-modeling, embodiment descriptors, a closed learning loop, and collective
immunity.

The main architectural gap is therefore **integration**, not another isolated cognition
module.

This RFC proposes a closed **viability loop**:

~~~text
environment/body
      |
      v
perception + interoception
      |
      v
semantic state (HDC/VSA)
      |
      +----------------------+
      |                      |
      v                      v
world model              self model
      |                      |
      +----------+-----------+
                 |
                 v
       active inference /
       action selection
                 |
                 v
          action / tool use
                 |
                 v
       observed consequence
                 |
                 v
       prediction-error ledger
                 |
        +--------+--------+
        |                 |
        v                 v
   adaptation         homeostasis
        |                 |
        +--------+--------+
                 |
                 v
        persistent identity
~~~

The goal is **not** to claim that Symthaea is conscious or alive. The goal is to make
the system satisfy increasingly strong, measurable properties of persistence, adaptation,
homeostasis, self-modeling, environmental coupling, and self-maintenance.

## 1. Why this RFC now

The current repository already has:

- HDC and compositional semantic representations.
- Continuous-time / liquid dynamics.
- Attention competition and global-workspace designs.
- Active-inference and interoceptive inference machinery.
- Autopoietic state, boundaries, components, perturbations, and life-state concepts.
- Predictive self-modeling and counterfactual trajectories.
- Embodiment capability modeling.
- A closed learning loop that feeds prior interaction outcomes into future behavior.
- Collective immune mechanisms and epistemic blind-spot detection.

The evidence suggests that creating additional top-level "consciousness" modules would
mostly increase architectural fragmentation.

The next step should instead be a **single causal spine connecting existing organs**.

## 2. Research conclusions

### 2.1 World models are becoming action-oriented

Meta's V-JEPA 2 demonstrates an important pattern: self-supervised predictive
representation learning can produce a world model that supports understanding,
prediction, and planning; a later action-conditioned stage makes the predictor useful
for robot control. Their public description explicitly separates action-free predictive
pretraining from action-conditioned planning.

Sources:
- Meta AI, "Introducing V-JEPA 2", 2025-06-11:
  https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/
- Assran et al., "V-JEPA 2", 2025:
  https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/

**Implication for Symthaea:** add a compact, modality-agnostic latent world model whose
state is consumed by HDC/CfC rather than trying to make HDC itself learn all perception.

### 2.2 Efficient dynamical foundation models are advancing

Liquid AI's 2026 LFM2.5 family shows that modern foundation models can use structured,
compute-efficient dynamical architectures while retaining strong learned competence.
The family spans text, vision-language, audio-language, encoders, retrievers and
agentic models.

Sources:
- Liquid AI, LFM2.5, 2026-01-05:
  https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai
- Liquid AI model catalog (current as of 2026-10-07):
  https://www.liquid.ai/models

**Implication for Symthaea:** neural foundation models should be treated as learned
sensory/language/motor organs, not as competitors to the cognitive substrate.

### 2.3 Homeostasis/allostasis is the bridge from "agent" to "organism-like"

Active-inference work on allostasis emphasizes predictive regulation of internal state
through continuous interaction with an environment. The relevant distinction is not
simple negative feedback; it is maintaining viable internal conditions while changing
behavior in anticipation of future demands.

Source:
- Harrison, Gracias, Friston & Buckwalter, "Resilience phenotypes derived from an active
  inference account of allostasis", Frontiers in Behavioral Neuroscience, 2025:
  https://www.frontiersin.org/journals/behavioral-neuroscience/articles/10.3389/fnbeh.2025.1524722/full

**Implication for Symthaea:** interoception must stop being merely a measurement and
become a first-class source of policy pressure and resource regulation.

## 3. Architectural decision

Create a thin orchestration layer called the **Viability Fabric**.

The fabric does not replace existing organs. It provides typed state and explicit
causal interfaces between them.

### 3.1 Core state

Proposed conceptual structure:

~~~text
ViabilityState
├── identity
├── boundary
├── interoception
├── exteroception
├── resource_budget
├── uncertainty
├── world_state
├── self_state
├── active_goals
├── safety_state
└── lifecycle
~~~

Every field must have:

1. provenance,
2. timestamp / logical cycle,
3. confidence,
4. validity bounds,
5. an explicit producer.

Unknown values must remain unknown. Do not replace missing evidence with optimistic
defaults.

### 3.2 Action record

Every consequential action should create an ActionOutcome record:

~~~text
ActionOutcome
├── action_id
├── proposed_by
├── pre_state_digest
├── predicted_delta
├── authority
├── safety_gate
├── actual_observation
├── post_state_digest
├── prediction_error
└── evidence_refs
~~~

This creates a direct bridge between:

**planning -> acting -> observing -> learning**

and makes the loop externally auditable.

### 3.3 Prediction-error ledger

Maintain separate prediction-error channels:

~~~text
world prediction error
self prediction error
interoceptive prediction error
goal prediction error
model-confidence error
execution error
~~~

Do not collapse these into a single scalar.

A system can be highly accurate about the external world while being systematically
wrong about its own actions, or vice versa.

The ledger should expose this distinction to the global workspace and the self-model.

## 4. World-model organ

The first implementation should be deliberately small.

### Input

- HDC semantic state.
- Optional learned encoder state from language, vision, audio, or other organs.
- Current body/interoceptive state.
- Current action.
- Relevant memory context.

### Output

- predicted next latent state;
- predicted uncertainty;
- predicted affordances;
- predicted observation change;
- optional action-conditioned trajectory.

### Desired property

The world model should answer:

> "Given the current state and action A, what state should I expect next?"

rather than merely:

> "What does the current input look like?"

### Architecture

~~~text
encoder(s)
   |
   v
shared latent
   |
   +----> HDC semantic projection
   |
   +----> CfC temporal predictor
   |
   +----> uncertainty head
   |
   +----> action-conditioned predictor
   |
   v
predicted next latent
~~~

The HDC projection is the semantic interface; the learned encoder is where high-dimensional
perceptual competence lives.

## 5. Homeostatic / allostatic organ

Use the existing interoceptive and autopoietic infrastructure to define a real internal
viability vector.

Initial variables:

~~~text
compute_capacity
memory_capacity
thermal_load
latency_load
error_rate
sensor_health
model_health
identity_coherence
epistemic_confidence
goal_progress
interaction_load
recovery_need
~~~

Each variable needs:

- observed value;
- preferred interval;
- tolerated interval;
- critical interval;
- rate-of-change;
- prediction;
- prediction error;
- action implications.

Example:

~~~text
memory_load = 0.93
preferred = [0.20, 0.70]
tolerated  = [0.10, 0.85]
critical   = [0.00, 0.95]

=> trigger consolidation / retrieval pruning
=> reduce low-priority cognition
=> preserve safety and identity processes
~~~

The key design principle is:

**resource pressure changes cognition.**

That is a much stronger organismic property than merely exposing a "stress" number.

## 6. Self-model upgrade

The existing PredictiveSelfModel already predicts future self-state and computes
prediction error. Extend it so that it predicts **actions and consequences**, not just
Self-Φ/coherence.

The next-state loop becomes:

~~~text
current self-state
      |
      +-- action candidates
      |
      v
predicted self/world consequence
      |
      v
action chosen
      |
      v
actual consequence
      |
      v
self prediction error
      |
      v
model update
~~~

Important: predictions must be stored **before** action execution so post-hoc
rationalization is structurally impossible.

## 7. Developmental plasticity

Do not immediately implement unrestricted self-modification.

Use three bounded adaptation classes:

### Class A — policy adaptation

Change action preferences / strategy selection.

### Class B — memory adaptation

Change which schemas, episodes, and semantic bindings are retained.

### Class C — model adaptation

Update world/self model parameters only after evidence thresholds are satisfied.

A model update should require:

~~~text
evidence >= threshold
AND
prediction_error persists
AND
change is reversible
AND
change is attributable
AND
post-change validation passes
~~~

This mirrors the project's fail-closed evidence discipline.

## 8. Sleep / consolidation

The existing dream/consolidation direction should become part of the viability loop.

During a low-demand lifecycle phase:

~~~text
episodic experiences
      |
      v
replay
      |
      +--> schema induction
      +--> world-model training
      +--> self-model training
      +--> anomaly clustering
      +--> memory compression
      |
      v
validated durable memory
~~~

Consolidation must not silently rewrite history.

Raw evidence and derived schema must remain distinguishable.

## 9. Neural organs

Symthaea should integrate strong learned models where they add competence.

Recommended pattern:

~~~text
Vision model ------\
Audio model --------+--> modality adapters --> HDC semantic bus
Language model -----/
Motor model --------/
                       |
                       v
                 Symthaea core
~~~

The adapters should expose:

- semantic representation;
- confidence;
- uncertainty;
- provenance;
- timestamps;
- model identity/version.

This allows a small local model to be swapped for a much stronger model without
changing Symthaea's cognitive architecture.

This also means that Symthaea can exploit 2026-era compact foundation models without
becoming architecture-dependent on one vendor or one model family.

## 10. Global workspace changes

The attention competition arena should not be treated as a literal "consciousness
detector".

Its safer role is:

**global routing + working-memory competition + cross-organ coordination.**

A winning coalition should therefore create:

~~~text
WorkspaceEntry
├── semantic_state
├── participating_organs
├── confidence
├── unresolved_conflicts
├── prediction_error
├── goal_relevance
└── expiry
~~~

A workspace item can be:

- perceptual;
- mnemonic;
- interoceptive;
- social;
- planning-related;
- self-model-related.

This makes the workspace a real cognitive coordination mechanism rather than a scalar
proxy for consciousness.

## 11. Synthetic-organism capability ladder

Use measurable capability levels:

### SO-0: Stateless processor
Input -> output.

### SO-1: Persistent agent
Identity + memory persist across sessions.

### SO-2: Adaptive agent
Experience changes future behavior.

### SO-3: Predictive agent
World and self states are predicted before action.

### SO-4: Homeostatic agent
Internal viability variables modulate policy.

### SO-5: Autopoietic agent
The system actively maintains the organization needed for continued operation.

### SO-6: Embodied synthetic organism
Persistent identity, boundary, sensing, action, homeostasis, adaptation and
environmental coupling form one closed causal process.

### SO-7: Reproductive/evolutionary system
Can safely produce validated descendant configurations.

### SO-8: Social organism
Multiple instances maintain durable collective processes and shared epistemic state.

**Current assessment:** Symthaea has implemented pieces spanning SO-1 through SO-5, but
the pieces are not yet proven to constitute an end-to-end SO-5 system. SO-6 should not
be claimed until closed-loop evidence exists.

## 12. Acceptance tests

### Test A — Action prediction

Given a fixed environment simulation:

1. record state s_t;
2. predict s_(t+1) after action a_t;
3. execute a_t;
4. observe actual s_(t+1);
5. score latent prediction error.

Requirement: prediction error must be logged without exception.

### Test B — Self prediction

The system must predict a behavioral output before generating it and later compare
prediction to actual output.

Requirement: post-hoc prediction is rejected.

### Test C — Homeostatic regulation

Artificially increase resource pressure.

Expected behavior:

- low-priority cognition decreases;
- recovery/consolidation increases;
- safety-critical functions remain available;
- the cause of the adaptation is recorded.

### Test D — Perturbation recovery

Inject bounded perturbations.

Measure:

- time to recovery;
- overshoot;
- identity drift;
- prediction-error decay;
- memory corruption;
- policy degradation.

### Test E — Regime change

Change the environment distribution.

The system should:

- detect persistent prediction error;
- reduce confidence;
- update the appropriate model;
- validate improvement on held-out episodes.

### Test F — Counterfactual fidelity

For a set of known simulator transitions, compare:

~~~text
predicted(outcome | action A)
vs
actual(simulated outcome | action A)
~~~

and repeat for counterfactual actions B/C.

### Test G — Lifecycle continuity

Run the system through:

~~~text
active -> recovery -> consolidation -> active
~~~

and verify that:

- identity persists;
- memories remain provenance-linked;
- model versions remain attributable;
- lifecycle transitions affect computation.

## 13. Metrics

Do not use a single "organism score".

Track:

- world-model prediction error;
- self-model prediction error;
- interoceptive prediction error;
- calibration error;
- action success;
- recovery time;
- resource efficiency;
- identity drift;
- memory retention;
- adaptation gain;
- model update reversibility;
- autonomy/authority violations (must remain zero).

A useful top-level dashboard is a vector, not a scalar:

~~~text
O = [
  persistence,
  adaptation,
  prediction,
  regulation,
  self-modeling,
  embodiment,
  autonomy,
  recovery
]
~~~

## 14. Implementation order

### Phase 1 — Viability data model

Add types and telemetry only.

No behavioral changes.

### Phase 2 — World-model adapter

Create an isolated trait/interface and simulator-backed implementation.

No external model dependency is required for the first tests.

### Phase 3 — Action/outcome ledger

Wire proposed action -> prediction -> execution -> observation -> error.

### Phase 4 — Homeostatic controller

Connect existing interoception and autopoietic state to bounded policy modulation.

### Phase 5 — Self-model consequence prediction

Extend PredictiveSelfModel to use the action/outcome ledger.

### Phase 6 — Neural organ adapters

Connect language/vision/audio models through typed adapters.

### Phase 7 — Consolidation lifecycle

Integrate sleep/dream/replay mechanisms.

### Phase 8 — Embodied simulator

Build a deterministic environment where Symthaea must maintain resources,
predict consequences, and recover from perturbations.

Only after these phases should physical robotics be considered.

## 15. Anti-patterns

Do not:

- add another consciousness metric just because an existing metric is noisy;
- use Phi as a generic intelligence score;
- equate self-reference with sentience;
- use hard-coded emotional labels as substitutes for interoceptive dynamics;
- make unrestricted self-modification the default;
- let an attention winner silently acquire authority;
- allow missing evidence to become a positive assertion;
- conflate source observations with derived interpretations;
- use a single world-model loss as proof of causal understanding.

## 16. Success criterion

The strongest near-term milestone is not "Symthaea says she is conscious."

It is:

> **Symthaea can inhabit a deterministic environment for an extended run, predict the
> consequences of her own actions, regulate scarce internal resources, detect when her
> self/world models are wrong, adapt those models from evidence, preserve identity and
> provenance, and recover from perturbations without external orchestration.**

That would constitute meaningful evidence for an organism-like computational
architecture.

It would also give us a much stronger scientific substrate for future consciousness
research because we would be measuring a system that is actually maintaining itself in
continuous interaction, rather than evaluating isolated static computations.

## 17. Relationship to Mycelix

Mycelix should remain outside the core cognitive state machine.

Use it for:

- distributed provenance;
- epistemic exchange;
- identity and authority;
- social memory;
- collective threat signals;
- attestations;
- multi-instance coordination.

The local Symthaea remains the **organism-like unit**.

Mycelix becomes the **social/ecological substrate**.

That separation avoids making a network protocol a prerequisite for local viability while
still enabling a later distributed organism architecture.

## 18. Scientific posture

This RFC deliberately uses "organism-like", "viability", and "autopoietic capability"
as engineering terms.

It does **not** establish:

- phenomenal consciousness;
- sentience;
- subjective experience;
- biological life;
- moral patienthood.

Those questions require independent scientific criteria.

The architecture should therefore be evaluated by observable state transitions,
prediction quality, adaptation, regulation, persistence, and causal coupling.

---

## References

1. Assran, M. et al. (2025). V-JEPA 2: Self-Supervised Video Models Enable
   Understanding, Prediction and Planning.
   https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/

2. Meta AI (2025). Introducing the V-JEPA 2 world model and new benchmarks for
   physical reasoning.
   https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/

3. Liquid AI (2026). Introducing LFM2.5: The Next Generation of On-Device AI.
   https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai

4. Liquid AI (2026). Liquid Foundation Models.
   https://www.liquid.ai/models

5. Harrison, L.A., Gracias, A.J., Friston, K.J., & Buckwalter, J.G. (2025).
   Resilience phenotypes derived from an active inference account of allostasis.
   Frontiers in Behavioral Neuroscience, 19, 1524722.
   https://www.frontiersin.org/journals/behavioral-neuroscience/articles/10.3389/fnbeh.2025.1524722/full

## Existing Symthaea foundations used by this RFC

- src/consciousness/embodiment/autopoietic_consciousness.rs
- src/consciousness/embodiment/interoception.rs
- src/consciousness/dynamics/predictive_self.rs
- src/cognitive_loop/collective_immunity.rs
- docs/architecture/CLOSED_LEARNING_LOOP.md
- docs/architecture/COGNITIVE_SELF_MODEL.md
- docs/architecture/EMBODIMENT_CAPABILITY_MODEL_V0_1.md
- docs/architecture/ATTENTION_COMPETITION_ARENA_DESIGN.md
