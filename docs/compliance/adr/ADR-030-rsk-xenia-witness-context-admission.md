# ADR-030: RSK Xenia witness context admission

**Status:** Proposed  
**Change Class**: A

## Context

RSK now has two separate ingredients for external witness trust:

1. Xenia can verify exact cryptographic witness keys for an exact state commitment;
2. RSK can map those keys through a governed trust snapshot and prove independent
   key, signer, and failure-domain quorums.

A remaining composition gap exists if a signed Xenia commitment carries a
`trust_context_digest` different from the trust context RSK independently
reconstructs.

## Decision

RSK will require exact equality between:

- the trust-context digest cryptographically bound inside the verified Xenia
  state commitment; and
- the `TrustContextDigest` independently reconstructed from RSK policy and the
  exact Xenia-verified key evidence.

Witness-quality verification occurs before digest equality. A matching digest
cannot compensate for weak signer/failure-domain evidence, and strong witness
quality cannot compensate for a commitment signed under a different context.

Success yields only an opaque witness-context evidence capability. It is then one
input to the separate monotonic-anchor state/counter equality check.

## Consequences

- old witnessed state cannot be transparently reused after signer/domain policy
  changes;
- remote/self-asserted trust-context labels cannot become authority merely by
  being signed;
- trust identity and durable-state identity remain separate predicates;
- uncertainty freezes new positive authority rather than triggering destructive
  action;
- the composition remains provider-neutral and does not grant replication
  authority.

## Evidence status

Reference code, contract, and adversarial tests are authored. Exact-head CI has
not yet established production evidence.

Production admission remains **DENIED / NOT YET ELIGIBLE**.
