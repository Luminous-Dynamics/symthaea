# RSK Quorum Policy Separation — Implementation Mapping v0.1

**Status:** candidate executable mapping; not production evidence until CI executes  
**Class:** A safety-critical authority semantics  
**Scope:** abstract quorum requirement separation only

This document maps ADR-005 to the reference Rust authority evaluator. It does not establish cryptographic policy provenance or authenticated signer independence.

## Problem closed by this tranche

Historically, R1-R3 quorum requirements came directly from:

```text
request.quorum.required_independent_approvals
```

while only R4/R5 had a kernel-owned minimum. A requester could therefore present a lower requirement for lower risk classes, including zero for R3.

The reference model now separates:

```text
QuorumEvidence  // what the current request snapshot claims exists
QuorumPolicy    // what the authority-policy side requires
```

## Effective requirement

The evaluator computes:

```text
policy_required = max(
    QuorumPolicy[class],
    constitutional_floor[class]
)

effective_required = max(
    policy_required,
    request.required_independent_approvals
)
```

Therefore the request-side value is self-restricting only. It can increase its own required threshold but cannot lower policy or constitutional floors.

## Kernel-owned floors

The candidate code defines:

```text
R0 -> 0  // replication denied independently
R1 -> 0
R2 -> 0
R3 -> 1
R4 -> 2
R5 -> 2
```

The R3 floor means the reference abstraction requires at least one additional independent-approval count in addition to an explicit grant. This is not a claim that the raw model proves the grant issuer and approval are practically independent.

R4/R5 preserve the pre-existing two-approval high-consequence floor.

## Reference baseline policy

`QuorumPolicy::REFERENCE_BASELINE` currently matches the kernel floors:

```text
QuorumPolicy::new(0, 0, 1, 2, 2)
```

The legacy convenience evaluator:

```text
evaluate_replication_authority(request, grant)
```

delegates to:

```text
evaluate_replication_authority_with_quorum_policy(
    request,
    grant,
    QuorumPolicy::REFERENCE_BASELINE,
)
```

This preserves the existing call shape while removing requester control over the trusted minimum.

## Stronger semantic policy

A separate policy-aware entry point accepts a `QuorumPolicy` and clamps every configured value to the kernel-owned risk-class floor.

This object is intentionally called `QuorumPolicy`, **not** `VerifiedRiskPolicy`. It can represent policy semantics but does not prove:

- who authorized the policy;
- which trust root/version it belongs to;
- whether it is current or superseded;
- whether its storage was rolled back;
- whether the running binary is admitted to consume it.

Those remain #1673/#1668/#1682 production gates.

## Candidate executable evidence

### Unit tests

The authority core now tests:

- requester zero cannot weaken the R3 baseline floor;
- stronger policy dominates a weaker requester value;
- requester can only self-restrict upward;
- even a `QuorumPolicy` filled with zeros cannot weaken R4/R5 below two.

### Public-API integration tests

`tests/quorum_policy.rs` exercises the same properties exclusively through exported API types and functions.

## Bound-ledger behavior

The existing ledger-bound evaluator continues to call the reference convenience evaluator, so its default path now inherits the explicit `REFERENCE_BASELINE` rather than trusting the request as the minimum policy source.

A future verified-policy adapter should add an explicitly policy-aware bound-ledger entry point only when the policy object itself has trustworthy provenance. This tranche does not expose an unverified caller-provided policy as if it were production authority.

## Why the old request field remains

`QuorumEvidence.required_independent_approvals` is retained for compatibility. It now means an untrusted requester-side **self-restriction**.

Removing or renaming the field can be a later API cleanup after stacked consumers migrate. Its presence is safe under the new evaluator because:

```text
request_requirement < policy_requirement  -> policy wins
request_requirement > policy_requirement  -> stricter request wins
```

## Deliberate non-claims

This tranche does not establish:

- authenticated approval signatures;
- distinct signer identities;
- signer key lifecycle/revocation;
- distinct practical failure domains;
- cryptographically verified policy provenance;
- policy freshness/supersession;
- trusted time;
- durable policy anti-rollback;
- production admission.

Raw `independent_approvals: u16` remains reference evidence only. #1668 must replace or wrap it with verified identities/failure-domain evidence before production admission.

## Formal-model mapping

TLA+ v0.2 (#1672) should distinguish policy requirements from approval evidence and prove at least:

```text
RequesterCannotLowerPolicyRequirement
PolicyCannotLowerConstitutionalFloor
StrongerPolicyCannotIncreaseAuthority
```

Policy changes should be modeled as governed state/version transitions rather than request fields.

## Promotion rule

Do not describe these properties as executed evidence until the focused Rust 1.96 compile/test/Clippy lane runs on the exact candidate commit. A queued workflow is not a pass.
