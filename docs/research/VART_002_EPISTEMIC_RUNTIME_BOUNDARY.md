# VART-002 Epistemic Runtime Boundary

Status: implementation boundary for post-VART-001 development. This document does not authorize VART-002 confirmatory execution or scientific claims.

## Required runtime order

The intended cognitive boundary is:

`perception -> provenance envelope -> memory/storage -> provenance-aware retrieval -> epistemic readiness -> proposal formulation gate -> existing proposal type -> normal authority -> action -> receipt -> grounding/revisit`

No stage to the left of `normal authority` grants permission to mutate a world.

The following bypasses are forbidden:

- `retrieval -> proposal` without readiness evaluation;
- caller-supplied `ReadyToPropose` used as a substitute for policy evaluation;
- `readiness -> action`;
- `formulated proposal -> action` without the existing authority path;
- provenance, confidence, memory salience, or retrieval rank treated as permission.

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

Filtering emits audit counts. Exclusion must therefore be observable to evaluation and diagnostics rather than appearing as an empty-memory condition.

## Readiness policy

`EpistemicReadinessPolicy` may return:

- `ReadyToPropose`
- `ObserveMore`
- `RequestCorroboration`
- `Abstain`

`ReadyToPropose` is intentionally weaker than authority. It means only that the configured epistemic sufficiency policy is satisfied.

`formulate_if_epistemically_ready` evaluates this policy inside the formulation gate and invokes the proposal builder only for `ReadyToPropose`. Blocked dispositions do not execute the builder. Invalid readiness inputs also fail before proposal construction.

The formulation result deliberately carries no action permission, edit authority, commit token, or receipt.

## Episodic-memory provenance sidecar

The active memory path is `MemoryCoordinator + EpisodicMemory`; the older `HippocampusActor` is deprecated.

The compatibility migration is implemented as an additive sidecar rather than by changing the historical serialized `Episode` schema.

`episode_subject_sha256` binds provenance to immutable episode content under a domain-separated SHA-256 identity. It includes encoding-time epistemic content/metadata and excludes mutable replay/reconsolidation state. Therefore replay count, retrieval count, and consolidation strength cannot silently change the provenance identity.

`EpisodicProvenanceIndex` provides immutable subject-bound provenance attachment:

- subject mismatch fails closed;
- conflicting re-binding of an existing episode fails closed;
- identical repeat attachment is idempotent;
- legacy/unannotated episodes resolve to explicit `RealityDomain::Unknown` with no fabricated grounding evidence.

`ProvenanceAwareEpisodicMemory` wraps the existing `EpisodicMemory` without changing existing episode serialization. Provenance-aware retrieval applies the requested epistemic view after similarity eligibility and before final top-k return.

The episodic retrieval audit separates:

- similarity-eligible candidates;
- provenance-admitted candidates;
- final returned candidates;
- taint exclusions;
- domain exclusions;
- unknown/legacy exclusions;
- top-k truncation.

Top-k truncation must never be reported as epistemic rejection.

This sidecar does **not** prove that every existing Symthaea memory consumer already uses provenance-filtered retrieval. Call-site migration remains explicit work and must not be inferred from the existence of the wrapper.

## Canonical World Forge integration status

The remote VART-002 branch does not currently establish a canonical World Forge proposal-construction hook that can be safely patched here: recursive source-tree inspection at the qualified pre-gate tree found no `world_forge` path and no `WorldRevisionProposal` symbol.

Therefore this tranche adds the generic formulation policy primitive but does **not** invent a parallel World Forge author loop or a substitute proposal type.

Runtime wiring remains pending exact source closure for the canonical author path. When that source is durably available, integration must preserve this order:

`provenance-aware retrieval -> readiness input -> EpistemicReadinessPolicy -> formulate_if_epistemically_ready -> existing World Forge proposal constructor -> existing authority -> action`

The integration must demonstrate that no direct `retrieval -> proposal` or `readiness -> action` bypass remains.

## Required call-site migration tests

When canonical runtime call sites are wired, add tests proving:

1. grounded-history consumers cannot receive legacy `Unknown` memory unless they explicitly choose a broader retrieval mode;
2. counterfactual-tainted recall cannot enter grounded-history proposal evidence;
3. `ObserveMore`, `RequestCorroboration`, and `Abstain` execute zero proposal-construction side effects;
4. `ReadyToPropose` constructs at most one proposal per gate invocation;
5. proposal construction still produces no authority token;
6. existing authority independently rejects unauthorized mutation even after successful formulation;
7. replay, consolidation, pruning, and retrieval do not mutate provenance identity;
8. VART-001 spent fixtures/seeds/evidence remain untouched.

## VART-002 measurement hooks

Future VART-002 evidence should preserve, separately:

- pre-filter/similarity-eligible memory count;
- provenance-admitted memory count;
- final returned memory count;
- taint exclusions;
- domain exclusions;
- unknown/legacy exclusions;
- top-k truncation;
- readiness input counts/confidence/conflict state;
- readiness disposition;
- proposal formation result;
- authority result;
- action/abstention/observation result;
- eventual grounding receipt.

This allows the experiment to distinguish failures in memory availability, ranking, epistemic filtering, readiness judgment, proposal generation, authority, and world action.

## Claim boundary

These runtime primitives are architecture, not evidence of general safety, rationality, intelligence, or world improvement. VART-002 must test them on fresh hidden benchmark families behind the DEVART/VART firewall, under prospectively frozen matched controls.

`confirmatory_execution_authorized = false`

`claim_authorized = false`
