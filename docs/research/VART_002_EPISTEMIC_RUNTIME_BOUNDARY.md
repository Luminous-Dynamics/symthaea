# VART-002 Epistemic Runtime Boundary

Status: implementation boundary for post-VART-001 development. This document does not authorize VART-002 confirmatory execution or scientific claims.

## Required runtime order

The intended cognitive boundary is:

`perception -> provenance envelope -> memory/storage -> provenance-aware retrieval -> epistemic readiness -> proposal -> normal authority -> action -> receipt -> grounding/revisit`

No stage to the left of `normal authority` grants permission to mutate a world.

## Transition policy

`ProvenanceTransitionPolicy` controls reality-domain transitions.

- grounded history may produce a new counterfactual/dream/replay/imported child without changing the parent;
- ordinary derivation cannot create `PhysicalGrounded` or `DigitalCommitted` objects;
- `DirectObservation` evidence grounds only the exact subject digest carried by that evidence;
- `CommitReceipt` evidence grounds only the exact subject digest carried by that receipt;
- counterfactual taint propagates through derivation;
- explicit grounding may clear active taint but never erases `counterfactual_ancestry`.

A future runtime integration must not duplicate these rules with weaker ad-hoc booleans.

## Retrieval policy

`ProvenanceRetrievalMode` defines explicit epistemic query scopes:

- `GroundedHistory`: untainted `PhysicalGrounded` / `DigitalCommitted` only;
- `GroundedOrImported`: grounded history plus untainted imports;
- `CounterfactualOnly`: counterfactual/dream or any actively tainted object;
- `AllWithProvenance`: no epistemic filtering, but provenance remains attached.

Filtering emits `RetrievalAudit` counts. Exclusion must therefore be observable to evaluation and diagnostics rather than appearing as an empty-memory condition.

## Readiness policy

`EpistemicReadinessPolicy` may return:

- `ReadyToPropose`
- `ObserveMore`
- `RequestCorroboration`
- `Abstain`

`ReadyToPropose` is intentionally weaker than authority. It means only that the configured epistemic sufficiency policy is satisfied.

## Episodic-memory integration decision

The active memory path is `MemoryCoordinator + EpisodicMemory`; the older `HippocampusActor` is deprecated. `EpisodicMemory` currently serializes `Episode` directly and exposes content-based recall methods.

This tranche intentionally does **not** add `ProvenanceEnvelope` directly to `Episode` because doing so would be a storage/schema migration affecting replay, serialization compatibility, and potentially existing evidence lineages.

Before wiring provenance into episodic storage, perform an explicit migration tranche that must include:

1. backward-compatible deserialization for legacy episodes;
2. a stable subject/content digest binding for every stored episode;
3. provenance preservation through store, replay, consolidation, pruning, and retrieval;
4. retrieval modes applied before recalled material is admitted as grounded-history evidence;
5. audit accounting for provenance-filtered candidates;
6. regression tests proving legacy unannotated episodes fail closed into `Unknown` rather than being silently promoted to grounded history;
7. no changes to spent VART-001 evidence or exact benchmark fixtures.

Until that migration lands, the provenance retrieval firewall is a reusable policy primitive, **not a claim that all Symthaea memory retrieval is already provenance-filtered**.

## VART-002 measurement hooks

Future VART-002 evidence should preserve, separately:

- pre-filter candidate memory count;
- post-filter returned memory count;
- taint exclusions;
- domain exclusions;
- readiness input counts/confidence/conflict state;
- readiness disposition;
- proposal formation result;
- authority result;
- action/abstention/observation result;
- eventual grounding receipt.

This allows the experiment to distinguish failures in memory availability, epistemic filtering, readiness judgment, proposal generation, authority, and world action.

## Claim boundary

These runtime primitives are architecture, not evidence of general safety, rationality, or intelligence. VART-002 must test them on fresh hidden benchmark families behind the DEVART/VART firewall, under prospectively frozen matched controls.
