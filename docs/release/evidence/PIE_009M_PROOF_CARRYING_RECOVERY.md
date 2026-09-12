# PIE-009M proof-carrying recovery verifier evidence

Date: 2026-09-12

## Scope

This note records the independent deterministic reference semantics in `scripts/pie-009m-proof-carrying-recovery-oracle.py`.

The verifier is intentionally small and structural. It does not execute hardware, grant control authority, provide cryptographic signatures, or claim secure attestation.

## Independent execution

The final candidate self-test was executed locally with Python 3 on 2026-09-12 and returned:

`ok`

No Symthaea module is imported by the oracle.

## Frozen semantics

The reference establishes:

- canonical deterministic encoding for the exact recovery snapshot;
- SHA-256 binding of topology version, opening operational set, and admissible hidden-world belief;
- proposal binding to action + expected topology version + snapshot digest;
- a changed snapshot cannot reuse a proposal intended for an earlier state;
- proposal-body tampering is detected independently of semantic checking;
- stale topology expectations fail closed;
- unknown actions fail closed;
- the verifier independently recomputes universal safety across every admissible hidden world rather than trusting the planner's claim;
- exact repeated verification is deterministic;
- all receipts carry verifier version and explicit scope `admissibility-only`;
- an accepted receipt is evidence that the structural verifier accepted a proposal for that exact snapshot, not evidence that any hardware action executed.

## Executed synthetic fixtures

The final self-test demonstrates:

1. a well-formed `start_generator` proposal over a healthy/failed uncertain belief is rejected as `NOT_UNIVERSALLY_SAFE`;
2. `probe_generator` over the same belief is accepted;
3. changing the action while retaining the old proposal digest is rejected as `PROPOSAL_DIGEST_MISMATCH`;
4. replaying a proposal against changed operational state is rejected as `SNAPSHOT_DIGEST_MISMATCH`;
5. a correctly digest-bound proposal with an old expected topology version is rejected as `STALE_TOPOLOGY_VERSION`;
6. an exact known-healthy generator snapshot permits a generator-start proposal;
7. repeated exact verification yields an identical receipt;
8. malformed snapshots fail closed;
9. an unknown action fails closed even when its own proposal digest is internally consistent.

## Important limitations

SHA-256 here is only an integrity identifier inside the reference semantics. There is no signature, signer identity, nonce/freshness protocol, hardware root of trust, anti-replay ledger, secure transport, or cryptographic authorization. Those should be added through the appropriate Xenia/Mycelix authority/evidence layer rather than by silently expanding this oracle.

The structural action rules are intentionally tiny and synthetic. PIE-009L owns partial-observation/belief semantics, PIE-009J owns startup-energy/islanding semantics, PIE-009K owns structural restart criticality, PIE-009H owns future-blind policy semantics, and PIE-009I owns communications/authority timing.

## Promotion boundary

The intended promotion path is:

`independent verifier -> production proposal/receipt types -> exact snapshot binding -> cross-check with PIE-009L/J/K/H/I -> signed/authorized evidence bridge -> local deterministic controller remains final execution authority`

Tracks #1976, #1968, #1924, #1932, #1847, #1852, #1647 and master #1604.
