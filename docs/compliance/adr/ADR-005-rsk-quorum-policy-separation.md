# ADR-005: Separate RSK Quorum Policy from Requester Assertions

**Date**: 2026-09-12  
**Status**: Proposed  
**Change Class**: A (Safety-Critical)

## Context

The RSK reference request currently contains:

```text
QuorumEvidence {
    independent_approvals,
    required_independent_approvals,
}
```

and the pure evaluator historically used `request.quorum.required_independent_approvals` as the minimum requirement for R1-R3, while only R4/R5 had a kernel-owned hard floor of two.

That mixes two different classes of fact:

- **evidence**: how many independent approvals the current snapshot claims to contain;
- **policy**: how many approvals are required before authority may be granted.

A requester may safely ask for a stricter condition on itself, but it must not be able to weaken the trusted policy requirement for its own request.

This ADR contains no physical replication mechanism, fabrication recipe, molecular design, or biological implementation.

## Decision

Introduce a separate `QuorumPolicy` input owned by the authority-policy side of the boundary.

Conceptually:

```text
RequestedAction
+ QuorumEvidence
+ QuorumPolicy
+ Grant / lineage / monitoring / safety evidence
-> ReplicationDecision
```

The effective requirement is:

```text
effective_required_approvals =
    max(
        constitutional_floor(risk_class),
        policy.required_independent_approvals(risk_class),
        request.required_independent_approvals
    )
```

The request-side value therefore becomes **self-narrowing only**. It may demand more evidence, but it cannot reduce a constitutional or policy floor.

## Reference baseline

Until verified deployment policy is introduced, the reference evaluator uses an explicit baseline policy rather than deriving policy solely from the request.

The v0.1 baseline is:

| Risk class | Reference minimum independent approvals | Rationale |
|---|---:|---|
| R0 | 0 | Replication is denied independently of quorum. |
| R1 | 0 | Explicit subject-bound grant remains mandatory; no additional approval count is constitutionally modeled. |
| R2 | 0 | Independent monitoring is mandatory elsewhere; additional approval count remains deployment policy. |
| R3 | 1 | R3 requires an explicit nonzero additional approval in the reference abstraction, in addition to its explicit grant. |
| R4 | 2 | Existing high-consequence hard floor; cannot be weakened. |
| R5 | 2 | Existing high-consequence hard floor; cannot be weakened. |

These counts are **reference semantics**, not proof of signer or failure-domain independence. #1668 remains responsible for authenticated identities, key lifecycle, roles, revocation, and practical failure-domain independence.

A deployment policy may only raise these requirements. R4/R5's hard floor remains kernel-owned even if a supplied policy asks for less.

## Compatibility

The existing `QuorumEvidence.required_independent_approvals` field is retained in this tranche to avoid a broad breaking migration. Its meaning changes from authoritative policy input to an untrusted **requester self-restriction**.

The existing two-argument `evaluate_replication_authority(request, grant)` remains as a reference convenience wrapper. It delegates to the policy-aware evaluator with `QuorumPolicy::REFERENCE_BASELINE`.

A new pure policy-aware evaluator allows a future verified policy adapter to supply stronger requirements without changing request semantics.

The existing ledger-bound evaluator continues through the reference-baseline wrapper, so its minimum requirement is no longer requester-controlled. This tranche deliberately does **not** expose an unverified caller-supplied `QuorumPolicy` through the bound ledger. A policy-aware bound-ledger entry point belongs with the future trusted/verified policy adapter, where policy provenance can be established before it becomes authority-bearing input.

## Fail-closed properties

The implementation must establish:

1. requester `required = 0` cannot reduce an R3 policy requirement below one;
2. requester `required = 0` cannot reduce the R4/R5 constitutional floor below two;
3. a supplied pure policy requiring more approvals dominates a weaker requester value;
4. a requester may self-restrict by asking for more than policy requires;
5. lower supplied policy values cannot weaken constitutional floors;
6. missing grant, negative containment, stale evidence, and all other independent denial facts remain unaffected;
7. the default bound-ledger path inherits the reference baseline rather than a requester-selected minimum.

## Important non-claim: policy is not yet cryptographically verified

`QuorumPolicy` in this tranche is a semantic policy object, not an authenticated production trust object.

Production admission still requires #1673/#1668 work to bind policy to:

- immutable policy version/digest;
- approved policy authority / trust root;
- freshness and supersession state;
- signer roles and key lifecycle;
- exact admitted build/runtime identity;
- durable evidence and recovery epoch.

A future opaque `VerifiedRiskPolicy` may wrap or produce these semantic requirements. This ADR intentionally does not conflate semantic separation with cryptographic provenance.

## Alternatives considered

### Keep using only the request field

Rejected. A requester should not define the trusted minimum evidence required to authorize itself.

### Remove the request-side field immediately

Deferred. Removing it would create a wider migration across the current stacked reference tests and consumers. Treating it as self-narrowing is safe and gives a clean migration path.

### Hard-code all risk-class requirements with no policy object

Rejected. Deployments need to be able to require more than constitutional/reference floors without changing code.

### Expose unverified policy directly through the bound ledger now

Rejected for this tranche. The pure semantic object is useful for testing policy dominance, but the authority-bearing ledger path should not imply that an arbitrary caller-constructed policy is trusted. That integration belongs with verified policy provenance.

## Evidence discipline

Authored tests and static review are design evidence. This ADR remains `Proposed` until the focused Rust 1.96 lane executes the exact candidate commit. A queued workflow is not a pass.

## Consequences

### Positive

- requesters can no longer weaken their own R3/R4/R5 quorum requirement;
- policy requirements become a distinct input rather than a property of the requested action;
- stronger deployment policy has a semantic representation without widening the request type;
- creates a clean seam for future `VerifiedRiskPolicy` and verified quorum evidence;
- avoids prematurely treating a caller-constructed semantic policy object as production authority.

### Residual risk

- approval count is still raw reference evidence, not authenticated independent identities;
- policy provenance is not yet verified;
- the exact semantics of stronger R1/R2 deployment profiles remain policy choices;
- production admission remains denied.

## Related work

- #1335 — Class A authority/evidence umbrella blocker
- #1668 — verified positive authority evidence
- #1673 — verified risk policy / requester separation
- #1676 — threat model and policy-weakening misuse cases
- #1724 — exact action binding
- #1726 — temporal monotonicity
