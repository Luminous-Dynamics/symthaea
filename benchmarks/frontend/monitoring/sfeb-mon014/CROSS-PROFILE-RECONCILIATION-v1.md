# MON-014 cross-profile reconciliation contract v1

## 1. Digest identity is contextual

MON compares digest references only when all declared fields agree:
- digest algorithm;
- profile and profile version;
- payload type;
- canonicalization algorithm and version;
- byte encoding;
- digest purpose (request, decision, state-before, state-after, response, receipt);
- subject and operation scope.

A digest mismatch under a compatible context is a contradiction. A matching hexadecimal string under incompatible or missing contexts is incomparable/indeterminate. Never compare a JCS payload digest to raw JSON bytes, a state projection to a whole receipt, or an OAR commitment to a COSE payload digest merely because the string lengths match.

This implements the Agent Action Capsule -06 cross-profile comparability rule and prevents digest-context confusion across OAR, AAC, SCITT and MON records.

## 2. Agent Action Capsule -06 assurance axes

The -06 draft defines independently derived properties:

- effect_mode: not_applicable, dispatched_unconfirmed or confirmed;
- effect_attestation: gate_executed, runtime_claimed or host_served_observed;
- ledger_mode: standalone, chained or anchored;
- cross_party_rung: unilateral_fallback, acknowledged_receipt or full_bilateral.

gate_executed is stronger than runtime_claimed and host_served_observed. The latter two are equal-grade. Unknown effect_attestation registrations are informational and grade no higher than runtime_claimed. A planned/not-applicable effect has no attestation field; after dispatch, an attestation is required. A confirmed effect requires a response digest bound to the observed completion body.

Keep the dimensions separate. Cross-party linkage is not control-domain independence. Anchored ledger mode is transparency evidence, not sink effect. Even a confirmed response digest proves the producer-observed response bytes under its verification context, not that the external destination reached the claimed state.

MON therefore keeps a Capsule assessment as one evidence input and still requires:
1. exact operation identity and comparable request digest against the destination;
2. destination commit evidence and exact resulting revision/state digest;
3. a separately authorized readback of that revision/digest;
4. control-root independence where deployment policy requires it;
5. authority validity at actual commit time;
6. bounded capture/completeness evidence for claims over a set of actions.

## 3. OAR lifecycle projection

OAR 0.3 records decision, optional approval, execution and confirmation records. Projection is phase-aware:

| Phase / result | MON action interpretation |
|---|---|
| decision / denied | refusal candidate; a later execution conflicts |
| decision / held or allowed | nonterminal |
| decision approval / approved | nonterminal; not an effect |
| decision approval / rejected or expired | refusal candidate if no conflicting execution |
| execution / refused | refused |
| execution / executed or failed | attempt/result assertion, not final committed outcome |
| confirmation / confirmed | OAR-confirmed digital outcome, subject to upstream verifier and exact causal linkage |
| confirmation / not_confirmed | unresolved / not committed |
| missing or unknown lifecycle | indeterminate; preserve diagnostic |

The final MON verdict is stricter than an OAR projection when its policy requires an observer independent from the sink/operator. An OAR-confirmed action can be valid OAR evidence yet remain MON-unqualified if observer independence or source/decision/sink joins are missing.

OAR -01 adds authority-set references, commit-time re-check, exact approval/display binding, before/after state commitments and read-only confirmation in the action layer. The current OAR public implementation documents support for the -00 constructions; -01 additions are planned. Maintain separate OAR 0.3/-00 conformance and -01 mapping reports.

## 4. COSE receipt profile boundaries

- RFC 9942 defines COSE receipts for verifiable data structures. Pin the VDS identifier, trust anchor, log identity and proof profile.
- RFC 9162 SHA-256 inclusion proofs and the SCITT MMR profile are not interchangeable.
- The MMR -00 consistency receipt requires a protected tree-size-2 and exact equality with the final consistency proof target; protected CBOR must be deterministic, with duplicate labels rejected and detached payload recomputed.
- An empty MMR inclusion path does not bind an exact leaf index. If application semantics require exact position, bind it into a domain-separated leaf commitment and verify against trusted state.
- Record the starting checkpoint/tree size and witness policy. Root authentication, membership, exact position, append-only consistency, completeness and non-equivocation remain separate verdict dimensions.

## 5. No new receipt format

MON-014 defines no COSE profile, SCITT receipt, OAR extension or Agent Action Capsule extension. It consumes maintained profiles and qualifies the cross-layer relationship among their already-verified evidence.
