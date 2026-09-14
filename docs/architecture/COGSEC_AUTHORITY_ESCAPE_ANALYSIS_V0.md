# CogSec Authority Escape Analysis v0

Status: initial live-code audit. This document asks a different question from the mutation census: **which objects, handles, traits and APIs currently carry the practical ability to mutate privileged cognition?**

The security objective is capability minimization. A component that only needs inference, observation or proposal authority should not possess an object through which it can also train a model, activate a goal, alter thermodynamics, seed persistent cognition or execute an external effect.

## Core rule

> Possession of a mutable handle is authority, regardless of whether the API calls it a capability.

CogSec therefore protects both state-transition sinks and the topology by which mutation-capable objects can escape into ordinary cognition.

## A0 — `AsyncMindHandle` is an ambient authority bundle

`AsyncMindHandle` is deliberately lightweight and `Clone`. Its single command channel can currently request:

- perception and text perception;
- arbitrary raw `MindInput`;
- `SetGoal`;
- activation/shutdown;
- social input;
- swarm gossip including mutation-bearing messages;
- direct thermodynamic changes;
- memory seeding;
- tick execution and state queries.

The actor loop then directly applies several of those commands to the owned `ContinuousMind`. In particular, `SetGoal` reaches `mind.set_goal`, `UpdateThermodynamics` directly changes load/mood/dimensionality and may call `LLMBackend::update_affect`, and swarm gossip reaches `receive_swarm_message`.

### Required migration

Do not rely only on hiding the handle. The actor is the true state owner and must enforce CogSec at the receiving sink. In addition, split caller-facing authority so most code receives less than the full command surface.

Candidate shape:

- `MindObserverHandle`: snapshot/stats/health/read-only subscriptions;
- `MindPerceptionHandle`: submit labeled observations/proposals;
- `MindInfluenceHandle`: bounded social/affect/attention proposals;
- `MindAuthorityHandle`: submit permit-bearing privileged mutation commits;
- `MindLifecycleHandle`: activate/shutdown under explicit local authority.

The full actor mailbox may remain one implementation channel, but P0 commands must carry state-bound permits or be converted to proposals before the actor mutates protected state. Handle separation is defense in depth; sink mediation is the actual security boundary.

## A1 — `LLMBackend` mixes inference and model mutation authority

The current `LLMBackend` trait combines ordinary generation with methods that can affect persistent/ongoing model behavior:

- `generate` / `generate_streaming` / direct-channel generation;
- `update_affect`;
- `apply_lora`;
- `export_gradient`;
- `apply_aggregated_gradient`;
- FEP/distillation modulation methods.

Any component holding `Arc<dyn LLMBackend>` therefore potentially holds more than inference authority.

### Required migration

Split the capability surface conceptually into:

- `LlmInference`: generation and read-only telemetry;
- `LlmAffectControl`: bounded runtime affect modulation, if retained;
- `LlmLearningExport`: read-only candidate/update export;
- `LlmMutationControl`: stage/promote/rollback model updates.

Ordinary cognition should hold only `Arc<dyn LlmInference>`. The learning controller/CogSec adapter owns mutation control. `apply_lora` and `apply_aggregated_gradient` should disappear from the inference-facing object.

A valid remote update or valid ZK proof is still only a **candidate**. Installation requires a local `LearningPromotion` authorization.

## A2 — Direct goal APIs are privileged sinks

`ContinuousMind::set_goal` constructs a `MindInput::Goal`; `process_inputs` then activates a `Goal`. `AsyncMindHandle::set_goal` exposes that path through a cloneable actor handle.

### Required migration

- `set_goal` becomes proposal-only or explicitly legacy/audit-only during migration;
- the active-goal collection is mutated only by an actor-local commit function that consumes `MutationPermit<GoalActivation>` (or the non-generic equivalent in v1);
- external/raw `MindInput` cannot directly represent authoritative goal activation;
- goal proposal and active goal use distinct Rust types.

## A3 — Raw `MindInput` is an authority-confusion carrier

The async API accepts arbitrary `MindInput`, while `MindInput::new` historically defaults source metadata toward an internal/trusted-looking source unless callers override it.

### Required migration

- all externally constructible inputs default to unknown/untrusted origin;
- trusted internal origin is created only by narrow internal adapters;
- `MindInput` must not carry live authority merely through fields such as `is_verified`, source strings or metadata;
- authoritative requests carry separate capability/permit objects established independently of the payload.

## A4 — Persistent memory stores are mutation-capable objects

`SemanticMemory` exposes public `store` and `store_with_timestamp`; the memory coordinator accepts graduation events and can process them into episodic storage. A higher-level CogSec check is therefore insufficient if arbitrary callers can still obtain mutable access to the lower-level store.

### Required migration

Prefer one of two patterns:

1. protected store owns the monitor/commit contract and accepts only admitted/enveloped entries; or
2. lower-level raw store remains crate-private and only a CogSec-aware facade exposes mutation.

A semantic/episodic memory object handed out as `&mut` is itself persistent-cognition authority.

## A5 — Parallel metadata arrays create security-envelope escape

Working memory currently tracks content and several metadata dimensions in parallel structures. This permits future code to move or transform the HDC value while forgetting to carry security metadata.

### Required migration

Use one indivisible `WorkingMemoryItem` / `SecurityEnvelope<T>` containing:

- cognitive value;
- arrival/lifecycle metadata;
- security label;
- provenance roots;
- epistemic-assessment reference;
- taint/revocation dependencies;
- ordinary application metadata.

Security labels are never reconstructed from HDC similarity or cognitive metrics.

## A6 — Trust relationships currently imply active cognitive influence

Social processing can weight peer thoughts by relationship trust and blend them into active thought; multi-agent paths can couple peer neuromodulator state. Mesh wisdom/affect can enter social/Hyperfeel pathways.

### Required migration

`Trust(agent, context)` and `InfluenceGrant(agent, mutation_class, budget)` are separate objects.

Authentication or trust may contribute to a local policy decision, but does not itself authorize affective, attentional or learning influence. Cumulative budgets must prevent many-small-delta attacks.

## A7 — Federated aggregation has an authority source-confusion risk

Remote `GradientMessage` values include a sender-side trust value that currently participates in aggregation. A principal must never authoritatively declare the security weight applied to its own mutation candidate.

### Required migration

Remote message carries source identity/provenance/update/version/signature. A local `TrustResolver` produces effective weight from local policy and trusted state. Contributor count is not independent evidence and must not be translated directly into security trust.

No aggregated update applies to the production model until `LearningPromotion` succeeds.

## A8 — Swarm mutation and resuscitation are separate P0 sinks

Swarm LoRA mutations and holocell resuscitation may pass structural/proof/health checks before being applied. Those checks establish candidate validity properties, not local authority.

### Required migration

- proof verification -> `QualifiedCandidate`;
- health validation -> `QualifiedStateCandidate`;
- local CogSec authorization -> `LearningPromotionPermit` / `StateRecoveryPermit`;
- commit-time state/epoch revalidation -> actual installation.

Never collapse `valid`, `healthy`, `signed`, `verified` or `authorized` into one boolean.

## A9 — Egress objects carry confidentiality authority

Any object that can send a message, tool argument, federation update, gradient or external request is an egress sink. If the sending interface accepts raw strings/vectors with no label, confidentiality policy can be bypassed even if ingestion is perfectly secured.

### Required migration

Public/remote sinks consume `EgressCandidate<T>` carrying a confidentiality label and derivation/provenance commitment. The sink or a immediately-adjacent trusted adapter performs a final flow/declassification check.

## A10 — Direct mutable field visibility is part of the TCB audit

`pub`, `pub(crate)`, `&mut`, `Arc<dyn Trait>` and actor command variants are all security-relevant when they expose a P0/P1 sink. The mutation census therefore needs an **authority escape column** in addition to the sink itself.

For each protected resource record:

- canonical owner;
- all mutable access paths;
- all trait objects capable of mutation;
- all actor/mailbox command paths;
- all serialization/deserialization routes that can reconstruct mutation requests/capabilities;
- whether a live permit is required at the final sink;
- whether the final sink revalidates state/policy/revocation immediately before commit.

## Static/API ratchets

Future CI should add source/API checks that fail when privileged authority escapes accidentally. Candidate ratchets:

- no public constructor / serde / `Default` for live permit types;
- mutation-control traits are not supertraits of inference traits;
- actor P0 command variants contain a permit or are proposal-only;
- raw persistent store mutators are not exported from the public facade;
- no new `apply_*gradient`, `apply_lora`, direct goal/policy/trust write or actuator method outside an adjudicated sink list;
- no externally constructible input defaults to trusted/internal origin;
- no P0 sink accepts a bare confidence/Phi/trust scalar as authorization.

These checks are not a replacement for review. They are ratchets against architectural regression.

## Target topology

The desired authority graph is intentionally asymmetric:

```text
untrusted world / peers / tools
          |
          v
 labeled observation + proposals
          |
          v
 ordinary Symthaea cognition
  (inference/read/propose only)
          |
          v
      MutationRequest
          |
          v
 CogSec reference monitor <--- trusted local facts / policy / revocation
          |
          v
 one-use state-bound permit
          |
          v
 protected actor/store/model/action sink
          |
          v
        receipt
```

Ordinary cognition should not possess a second path around the bottom half of this graph.

## Exit criterion for authority-escape audit

Before CogSec enforcement can claim P0 completeness:

1. every P0 sink has exactly one canonical state owner;
2. every mutable path to that owner is classified;
3. every externally reachable P0 path terminates in permit-bearing commit-time enforcement;
4. inference/observation handles do not expose mutation methods;
5. no serialized object can deserialize directly into a live permit;
6. `unknown P0 mutable handles = 0`.
