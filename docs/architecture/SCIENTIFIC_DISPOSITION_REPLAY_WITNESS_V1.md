# Scientific Disposition Replay Witness v1

**Status:** architecture contract candidate only; non-authorizing; non-qualifying.

**Parent:** `#783@16febe6615072f9a1a3d07560c66e53845a2f04b`

## 1. Purpose

#783 defines a bounded `ScientificDispositionAssessment` as a deterministic projection over an exact closed evidence view, exact scientific-policy/evaluator lineage, exact lifecycle/adjudication/dependency/triangulation generations, and a complete reason topology.

The next missing SCI-014 boundary is what happens after such an assessment is persisted, transmitted, cached, copied, or loaded from storage.

A serialized assessment is evidence-shaped data. It is not a live proof that:

- its internal commitment is valid;
- its cited proposition/evidence/policy objects still exist;
- its complete reason topology was actually derived by the declared evaluator;
- its stored primary disposition matches a fresh replay;
- its historical information cutoff was enforced;
- its referenced scientific-state generations are current;
- the caller was entitled to choose the current scientific-state roots.

This contract therefore separates **replay verification** from **present-state currentness** and requires positive witnesses to be verifier-owned runtime capabilities rather than serializable status objects.

---

## 2. Core theorem

The shared scientific layer must preserve:

```text
serialized disposition record
    != structurally valid record
    != identity-verified record
    != replay-verified disposition
    != current scientific-state binding
    != current qualified disposition witness
    != truth
    != canonical belief
    != recommendation
    != action authority
```

and:

```text
stored primary label == replayed primary label
    != complete replay equivalence

matching caller-supplied generation ids
    != authoritative currentness

replay-valid historical assessment
    != current assessment

current at verification time
    != eternally current

serializing a positive witness
    != preserving its qualification
```

The target architecture is:

```text
PersistedScientificDispositionRecord
        ↓ untrusted/deserialized input
identity + structure verification
        ↓
exact input reconstruction
        ↓
exact evaluator/policy replay
        ↓
full reason-topology equality
        ↓
ReplayVerifiedScientificDisposition
        ↓ authoritative current-state binding
currentness / anti-staleness verification
        ↓
CurrentScientificDispositionWitness
```

---

## 3. Persisted record remains ordinary evidence

A future durable record may be serializable and content-addressed, conceptually:

```text
PersistedScientificDispositionRecordV1 {
    record_profile_id,
    assessment_id,
    proposition_id,
    requested_scientific_use,
    information_cutoff,

    evidence_view_snapshot_id,
    source_lifecycle_generation_id,
    argument_adjudication_generation_id,
    dependency_graph_generation_id,
    triangulation_generation_id,

    disposition_policy_id,
    evaluator_implementation_id,
    evaluator_execution_ref,

    primary_disposition,
    reason_topology_commitment,
    policy_trace_commitment,
    assessment_commitment,
}
```

The exact Rust shape is not frozen here.

The record exists so a past scientific assessment can be retained and independently replayed.

It must **not** derive a type name or field such as:

```text
verified = true
current = true
qualified = true
trusted = true
```

that downstream code can treat as a live capability after deserialization.

---

## 4. Replay verification is a transaction, not a flag check

A replay verifier must reconstruct the assessment from the exact declared scientific inputs.

Conceptually:

```text
verify_replay(record, replay_context)
```

must perform at least:

1. record/profile/version validation;
2. canonical commitment recomputation for the persisted record;
3. exact proposition identity resolution;
4. exact evidence-view snapshot reconstruction and closure validation;
5. source lifecycle history reconstruction for the bound generation/cutoff;
6. external argument/adjudication graph reconstruction;
7. dependency graph reconstruction;
8. compatibility/triangulation state reconstruction;
9. exact immutable disposition-policy profile resolution;
10. exact evaluator implementation/execution lineage validation;
11. evaluator execution over only the declared inputs;
12. full reason-topology recomputation;
13. full policy-trace recomputation;
14. primary-disposition recomputation;
15. exact comparison against the persisted record.

The verifier must not stop at:

```text
stored primary disposition == replayed primary disposition
```

Two different reason topologies can yield the same summary label.

The replay theorem therefore requires equality of the complete scientifically material result, not merely the label.

---

## 5. Replay-positive witness

Successful replay should create a private-construction runtime object, conceptually:

```text
ReplayVerifiedScientificDispositionV1
```

Candidate properties:

- private fields;
- no public constructor;
- no `Deserialize`;
- preferably no `Serialize`;
- constructed only by the owner verifier path;
- retains the exact persisted record or its canonical identity;
- retains the exact replay-context identities;
- retains the recomputed reason topology / trace identity;
- exposes read-only scientific fields needed by downstream Atlas/query code;
- carries no action/governance authority.

A copied persisted record can recreate this witness only by rerunning verification.

Therefore:

```text
bytes from disk/network
    -> ReplayVerifiedScientificDispositionV1
```

is impossible without the verifier transaction.

---

## 6. Replay validity is not present currentness

Replay verification answers:

> Did this exact assessment faithfully follow from these exact declared inputs and evaluator semantics?

It does **not** answer:

> Are those inputs still the current scientific state?

A historical assessment from 2028 may replay perfectly in 2031 even after:

- new evidence arrives;
- a contribution is corrected or retracted;
- an external invalidation appears;
- the dependency graph changes;
- proposition equivalence is revised;
- the disposition policy changes;
- a new evaluator implementation qualifies;
- the evidence registry advances.

Therefore:

```text
ReplayVerifiedScientificDisposition
    != CurrentScientificDispositionWitness
```

is mandatory.

---

## 7. Authoritative current-state context

Currentness must not be established by asking the caller to supply generation IDs that merely match the assessment.

Invalid privileged shape:

```text
is_current(
    assessment,
    caller_supplied_lifecycle_generation,
    caller_supplied_argument_generation,
    caller_supplied_registry_head,
)
```

when the same caller is free to choose the supposed current universe.

That proves internal equality only.

A future production Atlas should instead obtain current scientific-state roots from an independently authoritative registry/storage boundary.

Conceptually:

```text
AuthoritativeScientificStateSnapshotV1 {
    scientific_registry_head,
    proposition_registry_generation,
    evidence_registry_generation,
    source_lifecycle_generation,
    argument_adjudication_generation,
    dependency_graph_generation,
    triangulation_generation,
    policy_registry_generation,
    evaluator_registry_generation,
    snapshot_identity,
}
```

The exact root mechanism is deferred to the Scientific Artifact/Registry storage architecture.

This snapshot is still **scientific-state authority**, not governance/effect authority.

---

## 8. Currentness verifier

Conceptually:

```text
qualify_current_disposition(
    replay_verified,
    authoritative_scientific_state,
    requested_use,
)
    -> CurrentScientificDispositionWitnessV1
```

must require exact agreement or explicitly qualified compatibility for every currentness-sensitive dependency used by the policy.

At minimum the verifier should check:

```text
proposition identity/current registration
closed evidence-view registry/source generations
source lifecycle generation
external adjudication generation
dependency graph generation
compatibility/triangulation generation
disposition policy identity/current registration
evaluator implementation/profile currentness
requested scientific use
information-cutoff semantics
```

A changed current head does not make the historical assessment invalid. It makes it stale for the requested current view.

---

## 9. Current positive witness

Successful present-state verification should produce a second private-construction object:

```text
CurrentScientificDispositionWitnessV1
```

Candidate properties:

- private fields;
- no public constructor;
- no serde derives;
- retains or owns the `ReplayVerifiedScientificDispositionV1`;
- retains the exact authoritative scientific-state snapshot identity;
- retains the requested scientific-use identity;
- retains verification/currentness generation information;
- exposes the bounded primary disposition and complete reason topology read-only;
- carries no recommendation, governance, medical, resource, or effect authority.

A caller must not be able to construct it from:

```text
PersistedScientificDispositionRecordV1
+ matching strings/hashes
```

without owner verification.

---

## 10. Historical replay is a first-class valid use

Historical scientific-state reconstruction should not require today's state to equal the historical state.

For an explicit historical query:

```text
AtlasView(as_of = t0)
```

replay verification should bind the exact historical-information snapshot admitted for `t0`.

The result may be represented by the generic replay-verified witness plus a typed historical-view context rather than by `CurrentScientificDispositionWitnessV1`.

The architecture must preserve:

```text
replay-valid historical disposition
    != stale/broken record
```

while also preserving:

```text
historical disposition
    != current disposition
```

This prevents today's retraction, correction, or newly discovered old paper from leaking backward into what the system actually could have known at `t0`.

---

## 11. Complete equality, not label equality

The replay verifier must compare scientifically material fields including at least:

```text
proposition identity
scientific-use identity
information cutoff/evidence-view identity
admitted contribution set
excluded/unresolved accounting
support/opposition relations
falsifier outcomes
active/resolved/unresolved defeaters
dependency topology references
compatibility/triangulation receipts
pooling dispositions
unresolved assumptions/contradictions/scope questions
discriminating experiment references
policy predicate trace
primary disposition
reason-topology commitment
```

If the stored record says:

```text
SupportedWithinScope
```

and a replay also says:

```text
SupportedWithinScope
```

but one support contribution or one active defeater differs, replay verification fails.

The summary label cannot hide topology drift.

---

## 12. Recompute, do not trust persisted derived fields

Any field that can be deterministically derived from lower-level canonical inputs must be recomputed by the verifier rather than accepted by assertion.

Examples include:

- record commitment;
- evidence-view closure/accounting totals;
- reason-topology commitment;
- policy predicate outcomes;
- primary disposition;
- currentness comparison;
- target compatibility already defined by deterministic qualified receipts.

Persisted derived fields are useful for audit/display/indexing, but remain untrusted until replay.

---

## 13. Policy implementation must reach the real evaluation call

#783 already distinguishes policy profile from evaluator implementation.

Replay must strengthen this further:

```text
exact policy profile resolved
    != evaluator claims to support profile
    != exact profile semantics reached evaluation operation
```

The verifier path should make it difficult for an evaluator to validate one profile and then execute another/default interpretation.

A future implementation may use a typed evaluator interface whose evaluation call receives the exact qualified policy/profile object rather than only a compact label.

This mirrors the broader repository principle:

```text
profile validation
    != profile bound into consequential operation
```

---

## 14. Ambient-state prohibition

Replay must be closed over explicit inputs.

The evaluator may not silently read:

- current wall-clock time;
- process environment;
- mutable global configuration;
- unbound database rows;
- live network results;
- hidden LLM/model memory;
- stochastic RNG state;
- current user preferences;
- current repository HEAD;

unless such data are explicitly part of the bound replay/execution context.

If external computation is required, its exact input/output/execution evidence enters the assessment as a scientific artifact rather than ambient state.

---

## 15. Currentness is vulnerable to TOCTOU

A current witness can become stale after verification if the authoritative scientific-state head advances.

Therefore:

```text
current at verification
    != current forever
```

A point-of-use consumer must use one of the following safe patterns:

1. operate entirely against the immutable state snapshot/head bound into the witness;
2. re-check that the authoritative state head is still equal immediately before consequential scientific use;
3. use an atomic/transactional registry snapshot that prevents the viewed state from changing underneath the operation.

For an informational UI, a stale witness may still be displayed as:

```text
last replay-verified at state H
```

but it must not be presented as current after H has advanced.

---

## 16. Scientific use does not become effect authority

Even the strongest witness in this contract means only:

> this exact bounded scientific disposition was faithfully replayed and is current against this exact authoritative scientific-state snapshot for this exact requested scientific use.

It does not authorize:

- policy enactment;
- clinical treatment;
- physical actuation;
- resource allocation;
- governance action;
- censorship/publication decisions;
- autonomous self-modification.

The boundary remains:

```text
CurrentScientificDispositionWitness
    != normative recommendation
    != governance authorization
    != effect capability
```

---

## 17. No verifier-selected truth universe

The verifier itself must not be able to choose arbitrary scientific roots and then certify their currentness.

Avoid a universal public trait such as:

```text
trait TrustedScientificStateProvider
```

that any downstream crate can implement and thereby mint currentness by promise.

Pluggable transport/storage may exist beneath an owner-local registry verifier, but the privileged currentness path should be explicit about which independently provisioned/registered scientific-state root it trusts.

This is not because scientific databases are infallible. It is because **currentness has no meaning without a defined reference state**.

---

## 18. Registry failure and degraded operation

If the authoritative current scientific-state root cannot be obtained or verified, current qualification fails closed.

The system may still expose a replay-verified historical/cached assessment with an explicit state such as:

```text
ReplayVerifiedButCurrentnessUnavailable
```

for human inspection.

It must not silently promote cached equality to currentness.

Therefore:

```text
registry unavailable
    != cached record is current
```

and:

```text
cannot verify currentness
    != scientific proposition false
```

---

## 19. Staleness is typed, not destructive

When a bound generation changes, the old replay/current witness should not be deleted.

A currentness check can return exact stale reasons such as:

```text
EvidenceRegistryAdvanced
SourceLifecycleAdvanced
ExternalAdjudicationAdvanced
DependencyGraphAdvanced
TriangulationGraphAdvanced
PolicyProfileSuperseded
EvaluatorProfileSuperseded
PropositionRegistryAdvanced
EvidenceViewProfileChanged
RequestedUseChanged
```

These are not automatically scientific opposition.

A stale witness remains provenance for a past assessment and can often seed a new replay request.

---

## 20. Reverification should not mutate the old record

If current state changes from H1 to H2:

```text
assessment A@H1
    ↓ replay/current verification
witness W1

state advances to H2
```

we do not mutate W1 into H2.

Instead:

```text
reconstruct/reassess under H2
    ↓
new persisted assessment A2
    ↓
new replay witness R2
    ↓
new current witness W2
```

if the disposition or reason topology changes.

If the new state is scientifically identical under the registered policy, an optimization may reuse canonical material only if exact equality/compatibility is itself proven; it must not be inferred from an unchanged summary label.

---

## 21. Negative qualification cases

A future implementation should prove at least:

1. downstream code cannot construct a replay-positive witness;
2. downstream code cannot deserialize a replay-positive witness;
3. downstream code cannot construct/deserialise a current-positive witness;
4. a forged persisted `SupportedWithinScope` record fails when reason topology does not replay;
5. equal primary labels with unequal reason topology fail replay verification;
6. a forged reason-topology commitment fails recomputation;
7. missing/excluded candidate evidence cannot silently disappear during replay;
8. evaluator execution against the wrong policy profile fails;
9. hidden ambient-state dependence is prohibited or captured as explicit execution input;
10. replay-valid historical assessment remains replay-valid after current state advances;
11. that historical replay cannot satisfy present currentness after state advances;
12. caller-supplied matching generation IDs cannot mint a current witness;
13. authoritative state-head mismatch fails current qualification;
14. registry-currentness unavailability fails closed without rewriting the proposition disposition to opposition;
15. stale current witnesses remain historical artifacts but fail point-of-use currentness;
16. changing requested scientific use requires a distinct eligibility/currentness assessment where policy says use matters;
17. currentness does not transfer governance/effect authority.

---

## 22. Suggested SCI-014 implementation split

The next implementation can remain very small:

```text
SCI-014f.1  persisted disposition record validation
SCI-014f.2  exact replay context / artifact resolution
SCI-014f.3  deterministic disposition replay
SCI-014f.4  private ReplayVerifiedScientificDisposition witness
SCI-014f.5  authoritative scientific-state snapshot interface
SCI-014f.6  currentness verifier
SCI-014f.7  private CurrentScientificDispositionWitness
SCI-014f.8  stale/current reason diagnostics
```

Do not begin with a database/UI.

The first executable theorem should be the smallest possible one:

```text
forged serialized primary disposition
    cannot produce replay-positive witness
```

followed by:

```text
replay-positive witness
    cannot produce current witness
    from caller-selected matching heads
```

---

## 23. Relationship to adjacent Symthaea patterns

This contract intentionally converges with strong patterns already emerging elsewhere in the repository:

```text
raw/serializable evidence input
    != non-deserializable qualified capability

profile declaration
    != profile bound into verifier operation

internal equality/currentness
    != legitimate reference-root selection
```

SCI-014 should reuse these architectural principles without importing domain-specific safety, cryptographic, or actuator authority into the scientific layer.

---

## 24. Refined Theory Atlas path

```text
SCI-006 evidence dependency graph
    -> #668 target compatibility / triangulation
    -> #701 defeater-aware argument graph
    -> #729 immutable proposition identity
    -> #769 append-only evidence lifecycle
    -> #783 disposition + complete reason topology
    -> replay verification                              [this contract]
    -> authoritative current scientific-state binding  [this contract]
    -> time-indexed SCI-014 storage/query
```

The actual Theory Atlas storage layer should persist replayable records and scientific artifacts, not persist a magical durable `current=true` capability.

---

## 25. Important non-claims

This contract does not:

- implement Rust types;
- define the scientific registry storage mechanism;
- choose a universal disposition policy;
- prove repository/registry operators are trustworthy;
- define cryptographic signatures for Atlas roots;
- guarantee complete literature discovery;
- prove proposition truth;
- turn scientific currentness into governance/action authority.

It only freezes the boundary that **serialized scientific conclusions are evidence to be replayed, while replay-validity and present-state currentness are separate verifier-owned capabilities**.
