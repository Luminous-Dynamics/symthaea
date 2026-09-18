# REGEN-042S1A — Bounded State and Commitment Oracle v1

Status: preregistration hardening only.

Parent state contract: REGEN-042S1 / #3823.
Primitive prerequisite: REGEN-042S0A / #3831.
Mycelix authoritative semantics: Luminous-Dynamics/mycelix#1431, #1519, #1523.

## 1. Purpose

REGEN-042S1 freezes immutable validated state and a canonical cryptographic commitment. S1A closes two additional assurance gaps before executable S1 work begins:

1. a structurally valid state must still be bounded enough to evaluate predictably;
2. commitment correctness must not be established only by one implementation agreeing with itself.

Core theorem:

```text
validated immutable state
+ explicit structural budgets
+ deterministic canonical framing
+ independent known-answer commitment oracle
= bounded replayable model-state identity
```

not observed truth, source authentication, resilience, hazard probability, authority, or actuation.

## 2. `u32` framing is not a resource bound

S1's canonical encoding uses `u32` collection/string lengths. That makes byte framing unambiguous, but it does not make multi-billion-element inputs acceptable.

S1A therefore requires explicit model budgets that are materially smaller than serialization-format maxima.

Conceptually:

```rust
pub struct ModelLimits {
    pub max_services: usize,
    pub max_dependencies: usize,
    pub max_stocks: usize,
    pub max_failure_domains: usize,
    pub max_provision_paths: usize,
    pub max_edges_per_service: usize,
    pub max_failure_domains_per_dependency: usize,
    pub max_dependencies_per_provision_path: usize,
    pub max_canonical_bytes: usize,
}
```

Exact names may differ, but executable S1 must bind one immutable limits profile before accepting a state.

## 3. Limits are part of model identity

A resource limit can affect whether an input is admissible. It is therefore not an invisible runtime tuning knob.

The accepted state or assessment envelope must bind the exact limits-profile revision used during validation.

```text
same semantic payload
+ different admissibility limits
!= same qualification context
```

A later deployment may adopt a larger profile, but that is a new model/limits revision rather than silent widening.

## 4. Fail before expensive materialization where practical

The builder should reject obvious over-budget input counts before constructing large ordered maps, canonical buffers, or graph indexes.

This does not require a zero-allocation parser. It does require that validation not knowingly allocate an unbounded canonical representation first and only then discover the state was too large.

## 5. Total canonical-byte budget

Per-collection count limits are insufficient because a legal combination of many maximum-length identifiers can still create an unexpectedly large commitment input.

S1A therefore requires an explicit maximum canonical encoded length.

The encoder must return a typed overflow/budget error if the canonical representation would exceed the adopted bound.

No truncation is permitted.

## 6. Canonical byte count is exact

If buffered canonical encoding is implemented, the reported encoded length must equal the actual byte vector length.

If streaming encoding is implemented, the byte-count accumulator must use checked arithmetic and produce the same total length as the independent oracle for the same fixture.

## 7. Streaming commitment is preferred but not a different theorem

S1 may avoid allocating the complete canonical byte vector by streaming canonical fields directly into SHA-256.

If both surfaces exist:

```text
SHA256(encode_to_bytes(state))
== streaming_commit(state)
```

must be a regression property.

Streaming must preserve the exact same domain prefix, field order, enum tags, integer encoding, collection ordering, and length prefixes as canonical byte materialization.

## 8. Commitment domain separation remains mandatory

S1 already freezes the canonical-state domain prefix:

```text
symthaea.regen042s.model-state.v1\0
```

S1A preserves it exactly.

The prefix is part of the hashed canonical byte sequence, not merely metadata beside the digest.

Changing the prefix or framing requires a new commitment version.

## 9. Commitment is not authentication

A `StateCommitment` proves only identity of the canonical modeled state bytes under the declared algorithm/version.

It does not prove:

- who supplied the state;
- whether Mycelix authorized/adopted it;
- whether an evidence snapshot is authentic;
- whether the represented observations are true;
- whether the state is fresh;
- whether the model is scientifically adequate.

Those are separate bridge/evidence/authentication propositions.

## 10. Independent known-answer oracle

The S1 qualification campaign must include at least one implementation-independent canonical encoder/commitment oracle.

Preferred initial form:

- a small standard-library Python program or another independently authored implementation;
- no import of the Rust encoder's output as expected data;
- explicit implementation of the frozen v1 byte-framing rules;
- fixed fixtures with frozen expected canonical byte hex and SHA-256 digests.

The oracle is test evidence, not runtime authority.

## 11. Known-answer vector family

The first vector family should include at least:

1. minimal one-service valid state;
2. state containing `Resolution::Unresolved`;
3. state containing `Resolution::NotApplicable`;
4. multiple services/dependencies inserted in non-canonical order;
5. multiple failure domains and provision paths;
6. non-zero fixed-decimal scales;
7. negative signed quantity in a field where signed deltas are permitted by the fixture grammar;
8. boundary identifier length;
9. boundary collection count under a deliberately small test limits profile;
10. tick-only mutation.

Each vector freezes:

```text
input semantic fixture
canonical bytes
canonical byte length
state commitment
```

## 12. Oracle independence test

The qualification workflow should compare at least:

```text
Rust buffered encoding
Rust streaming commitment
independent oracle bytes/digest
```

where available.

A PASS requires all applicable identities to agree exactly.

Agreement of two functions sharing one internal encoder is useful but is not the independent-oracle theorem.

## 13. Insertion-order metamorphism stays mandatory

For semantically identical input collections supplied in different orders:

```text
validated state A == validated state B semantically
canonical bytes A == canonical bytes B
commitment A == commitment B
```

must hold.

The independent oracle should include at least one reordered input fixture so this property is not verified only inside Rust.

## 14. Duplicate-before-map theorem

Duplicate identities must fail before `BTreeMap`/`BTreeSet` normalization can erase evidence of duplication.

An input such as:

```text
service_id = X
service_id = X
```

is invalid input, not one canonical service.

The commitment API must never receive a state manufactured through last-write-wins duplicate collapse.

## 15. Canonical sort order

Canonical ordering of identifiers is bytewise ordering of their validated exact ASCII representation.

It is not locale-sensitive, case-folded, Unicode-normalized, natural-sort, or path-normalized ordering.

This rule must be reflected in the independent oracle.

## 16. Unknown/unresolved discriminants are committed

The following are distinct byte-level states:

```text
Known(0)
Unresolved(reason)
NotApplicable
```

The commitment must therefore differ for each under otherwise identical state.

No omitted/default field may make those states collide semantically.

## 17. Budget errors are not resilience outcomes

Input rejected because it exceeds a model/encoding budget is an invalid/unexecutable campaign input under that profile.

It is not:

```text
service failure
resilience failure
hazard event
unresolved physical dependency
```

The result taxonomy must keep model-execution admissibility separate from modeled-world outcomes.

## 18. Resource budget must not become silent scientific censorship

A limit profile is a computational boundary, not evidence that nodes beyond the bound are unimportant or nonexistent.

If a real assessment requires more nodes than the qualified profile permits, the correct result is to widen and requalify the model profile or partition the analysis under an explicit composition theorem.

Silent dropping/truncation is forbidden.

## 19. No hidden graph pruning

State validation may reject invalid/dangling references, but it may not prune apparently unused services, dependencies, failure domains, stocks, or provision paths merely to satisfy size limits.

Any graph reduction belongs to a separately explicit transformation with its own before/after commitment lineage.

## 20. Hash dependency discipline

Executable S1 should use a maintained cryptographic SHA-256 implementation rather than custom hash code.

The exact crate/version is fixed by the qualified Cargo lock and recorded in qualification evidence.

S1A does not claim that a dependency is trustworthy merely because it is common or locked.

## 21. Algorithm agility is versioned, not ambient

V1 state commitments use SHA-256.

A future algorithm change creates an explicit new commitment algorithm/version identity. One digest byte string must never be interpreted without its algorithm/version context.

No runtime negotiation may silently reinterpret an existing v1 commitment.

## 22. Budget mutation controls

The first executable campaign should include adversarial mutations proving that:

- one-above-limit service count fails;
- one-above-limit dependency count fails;
- one-above-limit edge count fails;
- one-above-limit canonical byte length fails;
- exact-at-limit cases succeed when otherwise valid;
- oversized input does not produce a commitment;
- no error path truncates and hashes a prefix as if it were the whole state.

## 23. Commitment mutation controls

Qualification should also prove rejection/detection of at least:

- changed enum tag in the oracle;
- changed integer endianness;
- removed domain prefix;
- changed collection length prefix;
- unsorted map encoding;
- omitted unresolved discriminator;
- changed current tick;
- changed model/profile/evidence revision identity.

A mutation control that would still pass reveals an under-bound commitment theorem.

## 24. Qualification receipt contents

A later S1/S1A receipt should bind at minimum:

```text
exact ProductHead
exact S0A predecessor
Rust/Cargo/rustfmt identity
Cargo.lock digest
SHA-256 implementation identity
limits-profile identity
canonical encoding version
oracle implementation digest
fixture/vector digests
assertion results
postflight source identity
```

and preserve:

```text
system_closure=unfrozen
```

unless a later separately qualified environment theorem establishes otherwise.

## 25. Handoff rule

Executable S1 should not proceed directly from the original S1 document if it would omit these S1A conditions.

Preferred sequence:

```text
S0 exact-head evidence
-> S0A implementation + qualification
-> S1/S1A combined immutable-state implementation
-> exact-head bounded-state + independent-oracle qualification
-> S2 effect algebra
```

S1A may be implemented in the same product commit as S1 if the exact preregistered scope and qualification campaign cover both contracts; it does not require artificial code fragmentation.

## 26. Deliberate non-claims

REGEN-042S1A establishes no S0/S0A/S1 executable PASS, no complete denial-of-service resistance, no hermetic environment, no cryptographic source authentication, no observation truth, no model adequacy, no real-world resilience, no hazard probability, no policy ranking, no authority, and no physical-control permission.

Its proposition is narrow:

> a committed resilience-model state should be both semantically valid and computationally bounded, and its canonical byte identity should be checked against an independently implemented known-answer oracle rather than accepted because one encoder hashes its own output consistently.
