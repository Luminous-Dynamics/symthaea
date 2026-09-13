# Replicator Safety Kernel — Xenia Witness Context Admission v0.1

**Status:** Reference composition contract; not production-admitted  
**Change Class:** A  
**Production admission:** DENIED / NOT YET ELIGIBLE

## Purpose

A Xenia state-witness commitment contains a `trust_context_digest`. RSK must not
accept that value merely because it was signed. The value is admissible only when
RSK independently reconstructs the exact trust context from authenticated policy
and verified witness-key evidence, proves the required key/signer/failure-domain
quorums, and obtains the same digest.

## Admission theorem

```text
Xenia commitment signature/quorum valid
AND RSK witness-key quality valid
AND commitment.trust_context_digest == RSK_ReconstructedTrustContextDigest
```

is required before the external witness evidence may proceed to the separate
monotonic-anchor exact-state comparison.

Any missing predicate freezes new positive authority.

## Ordering

The reference path deliberately evaluates witness quality before digest equality.
A matching digest cannot compensate for insufficient independent signer or
failure-domain evidence.

Likewise, a strong witness quorum cannot compensate for a commitment bound to a
different trust context.

## Inputs

The composition consumes:

- the exact trust-context digest cryptographically bound by the verified Xenia
  state commitment;
- exact Xenia-verified key bindings;
- the RSK `XeniaWitnessTrustContextPolicy`;
- the RSK trusted-time interval.

The commitment digest is not treated as self-authenticating metadata: the lower
Xenia signature/context verifier must establish that it is part of the exact
signed commitment first.

## Output

Success produces an opaque reference type carrying only:

- the reconstructed trust-context digest;
- admitted distinct key IDs;
- admitted distinct signer IDs;
- admitted independent failure domains.

It does not carry grant or replication authority.

## Failure behavior

The predicate freezes on at least:

- malformed trust-context digest;
- unknown/untrusted verified key;
- duplicate verified key;
- lifecycle or validity failure;
- signature-profile mismatch;
- insufficient key quorum;
- insufficient signer-identity quorum;
- insufficient failure-domain quorum;
- commitment/reconstructed trust-context mismatch;
- any malformed trust policy.

Freeze here means denial of new positive authority. It does not imply destructive
action, quarantine, revocation, or automatic recovery.

## Separation from monotonic state comparison

This contract proves **who/what trust configuration witnessed the commitment**.
The monotonic-anchor layer separately proves **which exact RSK durable state the
commitment represents and whether its counter/state agree with local state**.

Both predicates are required. Neither substitutes for the other.

## Non-amplification

A verified witness context cannot:

- mint or widen grants;
- clear quarantine or revocation;
- extend expired authority;
- refresh stale evidence;
- replace trusted time;
- replace runtime/build admission;
- repair forks or rollback disagreement;
- advance/reset external monotonic state.

## Reference implementation

- `reference/rsk_xenia_witness_admission.py`
- `reference/test_rsk_xenia_witness_admission.py`

These are reference semantics only. Production remains blocked on executed exact-
head evidence and integration with opaque Rust/Xenia verified types.
