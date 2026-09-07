# Physical Agency Authenticated Independent Evidence v1

Status: design contract only. This document introduces no HAL path, actuator API, execution permit, or physical authority.

## Motivation

PA-15 freezes the safety-obligation set before confirmatory simulation and proves that completed obligations preserve the same case/obligation lineage. PA-16 strengthens that R1 structural boundary further: every judged metric must be explicitly present in the exact machine request, the public strict minting surface is claim/request-bound, and the complete **normalized-canonical** outcome-claim definition is bound into v4 safety-evidence identity rather than relying on `claim_id` or caller criterion order.

PA-16 canonicalizes `AllCriteria` ordering, rejects exact duplicate criteria, normalizes IEEE-754 signed zero because the comparison semantics treat `-0.0` and `+0.0` identically, rejects mathematically unsatisfiable scalar conjunctions before solver execution, and rejects interval-backed claimed metrics whose reported point estimate lies outside the reported interval before claim evaluation.

`Normalized-canonical` is intentionally narrower than a universal semantic normal form. Metric units are still string identities, and logically redundant-but-distinct criteria can remain distinct. PA-16 therefore claims deterministic normalization, satisfiability checking for the current scalar predicate algebra, and internal estimate/interval consistency—not ontology-level equivalence across all possible physical descriptions.

The strict R1 profile permits exactly one `EvidenceKind::Simulation` safety obligation. That obligation is the exact confirmatory-outcome anchor and must cite the exact v4 lineage. Additional simulation-derived safety claims require future typed per-obligation or multi-run evidence bindings rather than copied receipts attached to free-form prose.

PA-16 still inherits one explicit limitation from `symthaea-formal-safety`: independent evidence is ultimately represented as string references. A non-simulation evidence reference can therefore be structurally independent from the exact solver lineage without proving who issued it, whether the issuer was authorized for that evidence purpose, whether its key is current, or whether the evidence is still fresh.

The next boundary must therefore distinguish:

```text
EvidenceReference
    != SignatureVerifiedEvidence
    != LifecycleGovernedEvidence
    != IndependentEvidence
    != PhysicalExecutionAuthority
```

The existing fabrication-kernel attestation/trust design is the reference pattern: canonical bytes, detached signatures, explicit verification policy, key-usage authorization, key lifecycle, revocation, fresh sequence-numbered trust snapshots, and non-serializable verified receipts. Physical Agency should reuse those semantics, but must not depend on fabrication-specific manifests or trust types.

## Evidence tiers

Physical Agency should name evidence strength explicitly.

```text
R0 ReferenceOnly
   opaque/string evidence reference; research/reporting only

R1 StructuralBound
   preregistered obligation
   + exact proposal/run/context lineage
   + exact requested-metric lineage
   + normalized-canonical complete claim lineage
   + pre-run scalar claim satisfiability
   + claimed estimate/interval internal consistency
   + exactly one exact-run Simulation safety anchor

R2 SignatureVerified
   canonical evidence envelope + cryptographically valid signature

R3 LifecycleGoverned
   R2 + signer known/current/not revoked + usage allowed
   + fresh trust snapshot + bounded natural expiry

R4 IndependentQuorum
   R3 + role separation + configured minimum distinct issuers
   and, when required, algorithm/provider diversity
```

PA-15 established preregistered R1 safety-obligation lineage. PA-16 closes additional R1 aliasing and preflight gaps: claimed metrics must appear in `SimulationRequest::requested_metrics`; same-ID/different-definition claims produce different structural safety identities; criterion permutations have one normalized identity; signed zero has one normalized identity; exact duplicate criteria fail before execution; impossible scalar conjunctions fail before execution; interval-backed claimed metrics must have point estimates inside their intervals; and multiple free-form Simulation safety obligations fail before execution until typed bindings exist. Neither PA-15 nor PA-16 may be relabeled R2-R4 retroactively.

Strict **simulation qualification** may remain available at R1. Any future boundary that contributes to physical execution authority must declare the minimum evidence tier it requires; no default may silently upgrade R1.

## Canonical evidence envelope

A serializable evidence envelope is untrusted data until verification succeeds.

Conceptual shape:

```text
SignedIndependentEvidenceEnvelope
    schema_version
    evidence_id
    evidence_kind
    evidence_usage
    subject_scope
    artifact_digest_algorithm
    artifact_digest
    issuer_id
    signer_key_id
    signature_algorithm
    issued_at
    expires_at?
    provenance
    signature
```

`subject_scope` must make replay semantics explicit. A scope may bind to some or all of:

```text
SafetyCase UUID
ProofObligation UUID
proposal id
transition id
world snapshot identity
canonical simulation request transcript
normalized-canonical outcome-claim transcript
artifact/model/calibration identity
```

Generic reusable evidence such as a standard or calibration certificate must declare its reuse scope explicitly rather than relying on omission.

## Canonical bytes

Verification must operate on deterministic canonical bytes produced inside the trusted evidence layer. Human-readable labels must not be authoritative when a typed identity exists.

The canonical encoding must:

- carry a schema/version domain separator;
- use length-prefixing for variable-length fields;
- canonicalize set/map ordering;
- canonicalize logically unordered claim/evidence collections where the schema defines them as unordered;
- normalize representation aliases whose semantics are explicitly defined as identical;
- encode digest algorithm identity together with digest bytes;
- bind evidence usage and subject scope;
- exclude the detached signature itself from the signed message;
- reject duplicate or ambiguous issuer, criterion, or scope identities.

A compact digest may be derived for indexing, but canonical bytes remain the source of truth.

## Trust snapshot

Cryptographic signature validity alone is insufficient.

A trust snapshot for independent evidence must be bounded, sequence-numbered, canonical, and fresh at evaluation time. A signer must be:

```text
known
+ active
+ inside not_before/not_after window
+ not retired/revoked
+ authorized for the requested evidence usage
```

Required usage classes should be generic rather than fabrication-specific, for example:

```text
IndependentSafetyEvidence
FormalProofEvidence
CalibrationEvidence
InspectionEvidence
HumanReviewEvidence
StandardAuthorityEvidence
TelemetryEvidence
```

No generic `AnyEvidence` usage should exist on a strict path.

Trust-snapshot rollback, sequence collision, stale snapshot use, and issued-at regression must fail closed.

## Verification receipt

Successful verification mints a non-serializable runtime receipt with private construction.

Conceptual shape:

```text
VerifiedIndependentEvidence
    evidence_id
    evidence_kind
    evidence_usage
    exact subject scope
    artifact digest
    issuer identity
    signer identity
    signature algorithm
    trust snapshot digest
    trust snapshot sequence
    evaluated_at
    valid_until
```

The serialized signed envelope is never equivalent to `VerifiedIndependentEvidence`.

A verified receipt must not survive process restart as authority. Persist the signed envelope and reverify it against current trust before recreating the receipt.

## Natural expiry

A verified receipt must expire when any upstream authority expires.

```text
valid_until = min(
    envelope_expiry_if_any,
    signer_key_expiry_if_any,
    trust_snapshot_expiry
)
```

If any boundary is unknown where strict policy requires it, verification fails closed. Before a receipt is consumed, the consumer must establish `now < valid_until`. Long-lived operations require current-state revalidation rather than assuming an earlier receipt remains valid.

## Independence policy

`non-Simulation` is not by itself proof of independence.

A strict independence policy must reason about issuer/provider roles. At minimum, configured policy should be able to reject evidence when the same authority controls both:

```text
simulation backend
and
independent safety evidence
```

Higher assurance profiles should support disjointness from:

```text
candidate/model provider
simulation backend
outcome evaluator
safety evidence issuer
release/execution authority
```

Policy shape:

```text
IndependentEvidencePolicy
    minimum_evidence_tier
    allowed_usages
    allowed_issuers?
    denied_issuers?
    minimum_distinct_issuers
    minimum_distinct_algorithms?
    forbidden_role_overlap
    maximum_evidence_age?
```

A provider may satisfy multiple roles only when policy explicitly permits that overlap.

## Quorum

For high-consequence evidence, one issuer should not necessarily constitute sufficient independence.

R4 may require:

```text
N distinct authorized issuers
+ optional algorithm diversity
+ optional organizational/provider diversity
```

Duplicate signatures from the same signer identity do not increase quorum weight.

## Binding to PA-16

PA-16 should remain an honest R1 structural qualification path.

A later implementation should add a stricter qualifier rather than weakening or silently changing PA-16 semantics:

```text
ClaimBoundConfirmatorySimulationQualification
        +
VerifiedIndependentEvidence[]
        ↓
AuthenticatedSafetyQualification
```

For each preregistered non-simulation obligation, the authenticated qualifier must prove that at least one accepted verified receipt:

- names the exact frozen obligation identity;
- matches the expected evidence usage/kind;
- satisfies the configured minimum tier;
- is current at qualification time;
- satisfies issuer-role independence policy;
- has not been replaced by a neighboring case/run/claim/scope.

The single Simulation obligation remains bound to the exact confirmatory v4 run/claim lineage. If a future safety case requires multiple simulation-derived obligations, each must acquire a typed obligation-to-run/claim binding before the strict profile permits it.

The existing string `evidence_refs` may remain for human/audit reporting but cannot carry authenticated authority by itself.

## No verifier oracle

Physical Agency must not accept a caller-supplied boolean such as `is_verified=true`.

Likewise, a public verifier trait whose arbitrary implementation can simply return `true` is not by itself an authority root. Cryptographic-provider abstraction may exist, but trust-policy evaluation and receipt construction must occur inside a trusted layer whose provider configuration is not supplied by cognition on each qualification call.

The preferred long-term architecture is to extract generic attestation/trust primitives from existing engineering-grade implementations into a core trust package, then have fabrication and Physical Agency consume that shared package. Physical Agency should not gain a dependency on `symthaea-fabrication-kernel` merely to reuse fabrication-specific types.

## Implementation order

1. Qualify the PA-15 baseline and the current PA-16 PHYSIS v1 strict head first.
2. Repair any exact hosted fmt/check/test/clippy failures before adding dependencies.
3. Extract a generic attestation/trust core only after dependency/lockfile review.
4. Add canonical independent-evidence envelopes and usage-scoped trust records.
5. Add fresh/revocation-aware verification and non-serializable receipts.
6. Add role-separation policy and quorum tests.
7. Extend PHYSIS to an authenticated-evidence profile.
8. Only then consider whether any future physical-execution qualification may consume that evidence tier.

## Adversarial qualification tests

The authenticated-evidence implementation must reject at least:

- valid signature from unknown signer;
- valid signature from revoked/retired/expired signer;
- signer not authorized for the required evidence usage;
- stale or rolled-back trust snapshot;
- envelope scope for neighboring SafetyCase/obligation/proposal/run/claim;
- changed artifact digest with reused signature;
- issuer-role overlap forbidden by policy;
- duplicate signer counted twice toward quorum;
- evidence that expires between verification and consumption;
- serialized/fabricated `VerifiedIndependentEvidence` construction;
- generic `evidence_refs` string offered where authenticated evidence is required.

## Non-goals

This contract does not:

- authenticate simulation backends;
- authorize hardware;
- create an execution permit;
- define weapon or destructive actuation semantics;
- make the current `SafetyCase` an authority root;
- claim that cryptographic evidence is factually correct merely because it is signed.

The invariant remains:

```text
CanModel(effect)
    != CanPropose(effect)
    != CanQualifySimulation(effect)
    != CanExecute(effect)
```
