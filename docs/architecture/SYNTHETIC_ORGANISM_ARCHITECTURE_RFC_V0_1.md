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

The main architectural gap is therefore **integration and empirical closure**, not another isolated cognition module. Symthaea already exposes a `WorldModelBridge` and ODE-based trajectory planning inside `FepModule`; this RFC does not propose replacing those systems.

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

## 4. World-model qualification boundary

The first implementation should deliberately **qualify and extend the existing `WorldModelBridge` / `FepModule` path**, rather than create a second world-model implementation.

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

The HDC projection is the semantic interface; learned encoders are where high-dimensional
perceptual competence lives. The existing ODE trajectory planner can provide the first
action-conditioned simulator path; the next question is whether its predictions are
actually grounded by measured environment transitions.

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

## 10. Structured decision organs

Not every cognitive decision should be routed through a generative language model.

Liquid AI's October 2026 d1 release is a useful architectural signal: a specialized
decision model can consume state/questions and return calibrated probabilities without
generating output tokens. This pattern is well suited to routing, scoring, filtering,
and other low-latency decisions.

For Symthaea, add a generic decision-organ adapter alongside generative language organs:

~~~text
state + typed question
        |
        v
decision organ
        |
        +--> probability distribution
        +--> confidence/calibration
        +--> model identity
        +--> provenance
        |
        v
Viability Fabric / policy selection
~~~

Candidate uses include:

- action ranking;
- attention routing;
- anomaly triage;
- tool selection;
- memory retention;
- sensor confidence;
- safety escalation recommendation.

A decision organ must recommend rather than silently acquire authority. Authority remains
owned by the existing safety/actuator gates.

This gives Symthaea a practical decomposition:

**generative organs produce possibilities; decision organs score possibilities; the
Viability Fabric records the prediction and outcome; the action layer executes only with
authority.**

Reference:
- Liquid AI, "Introducing d1: The most capable decision model, now with vision",
  2026-10-05:
  https://www.liquid.ai/blog/d1-decision-model

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

## 12.1 Initial implementation landed

The RFC's first qualification layer is now implemented on the RFC branch:

- `src/cognitive_loop/viability_fabric.rs`
  - typed viability state;
  - lifecycle phase;
  - signed action consequences;
  - separated prediction-error channels;
  - pre-action prediction ledger;
  - outcome provenance;
  - fail-closed rejection of post-hoc predictions;
  - preservation of pending predictions after rejected/tampered outcomes.
- `src/cognitive_loop/viability_micro_world.rs`
  - deterministic six-action environment;
  - replay-stable transition function;
  - deterministic perturbation schedule for recovery testing;
  - persistence predictor baseline;
  - generic predictor trait with evidence-weighted confidence;
  - prediction-error evaluator;
  - scenario suite;
  - episode-isolated online-adaptation suite;
  - frozen held-out cross-scenario transfer protocol;
  - side-effect-free multi-step counterfactual rollouts;
  - survival-aware homeostatic policy;
  - scenario-aware reactive and horizon runners.

The deterministic micro-world is intentionally simple. Its role is to establish a test
oracle and a reproducible experimental boundary before attempting to qualify real
HDC/CfC/FEP predictions.

The first live integration is observational: `FepModule` owns a bounded
`ViabilityFabric`, refreshes canonical thermodynamic load, canonical FEP energy reserve,
and prediction/error state each cycle, and exports a `ViabilityTelemetry` view through the
existing `CycleMetadata` stream. The existing temporal planning-depth factor also has an
opt-in viability modulation; it is neutral by default. The measured state therefore cannot
silently change behavior while the control path remains disabled during qualification.

The fabric derives finite-difference trends for observed viability variables. Worsening
movement toward a preferred-band boundary can contribute anticipatory pressure before the
absolute state becomes critical. Absolute state pressure and rate pressure remain separate
inside the variable representation, allowing later ablations of reactive versus anticipatory
regulation.

The world-model bridge has also gained a lightweight action-conditioned delta model with
evidence-weighted confidence. Confidence rises with repeated accurate transitions and is
suppressed by persistent prediction error. A predictor can therefore be routed into a
policy only after accumulating actual transition evidence.

The action model reports confidence as evidence quantity multiplied by empirical accuracy.
Persistent prediction error therefore suppresses confidence instead of allowing repetition
to manufacture certainty.

The micro-world policy layer now evaluates finite counterfactual horizons. It tracks the
minimum confidence and minimum predicted viability margin across the imagined trajectory,
and subtracts explicit uncertainty/risk penalties from candidate utility. This is a
measurement-oriented bridge from prediction to actionable control; it is not yet an
embodiment claim.

The first benchmark questions are therefore:

1. Can a predictor beat persistence?
2. Can a policy maintain viability while pursuing progress?
3. Does prediction error remain attributable to a pre-action forecast?
4. Do identical runs produce identical evidence?
5. Does learning transfer to a different deterministic scenario when the test phase is frozen?
6. Does multi-step counterfactual planning improve action selection without mutating the predictor during evaluation?
7. Does the planner preserve a positive predicted viability margin under uncertainty?

No current result from this harness should be interpreted as evidence of consciousness
or biological life.

## 12.2 Next integration boundary

The next implementation should wrap the existing `WorldModelBridge` and FEP trajectory
planner, rather than invent a separate predictor. The micro-world becomes the
deterministic external oracle:

~~~text
existing WorldModel/FEP prediction
              |
              v
        Viability Fabric
              |
              v
      MicroWorld transition
              |
              v
      observed consequence
              |
              v
       prediction-error ledger
~~~

Once this is working, we can measure whether the existing world-model path actually
improves with experience, whether that improvement transfers to changed initial
conditions, and whether confidence tracks evidence instead of being treated as an
unearned scalar.

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

### Phase 2 — World-model qualification

Wrap the existing `WorldModelBridge` and `FepModule` trajectory-planning path in the
Viability Fabric. Add a deterministic simulator-backed adapter only where the current
model lacks an observable transition oracle. No external model dependency is required
for the first tests.

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

## 15. Current research position

A recent embodied-intelligence framing distinguishes a basic input-output system (S0)
from systems that maintain a boundary and endogenous stake in persistence (S1), systems
that model consequences of their own actions (S2), and increasingly social/collective
organization above that. This RFC is deliberately aimed at making those transitions
measurable rather than inferred from labels.

Reference:
- "The Embodied Hijack: when Pleistocene minds meet disembodied artificial intelligence",
  Frontiers in Psychology, 2026:
  https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2026.1889386/full

## 16. Anti-patterns

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

## 17. Success criterion

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

## 18. Relationship to Mycelix

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

## 19. Scientific posture

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
