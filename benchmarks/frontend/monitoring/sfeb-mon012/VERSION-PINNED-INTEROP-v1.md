# MON-012 — version-pinned standards interoperability v1

**Snapshot date:** 2026-11-26. Pin the exact draft revision and verifier/vector commit in every run. Internet-Drafts can change.

## 1. Inventory

| Artifact | Status at snapshot | MON role | Must not claim |
|---|---|---|---|
| RFC 9942 — COSE Receipts | Published IETF Standards Track, June 2026 | COSE receipt substrate and VDS proof interfaces | Receipt verification proves source truth, authorization, database commit or physical effect |
| RFC 9943 — SCITT architecture | Published IETF Standards Track, June 2026 | Signed-statement registration and transparency-service architecture | Registration proves every event was registered or a statement is true |
| COSE Receipts for MMRs -00 | Active SCITT WG Internet-Draft, 21 Nov 2026 | MMR-specific receipt adapter | MMR proof semantics are identical to RFC 9162 SHA-256 inclusion semantics |
| Open Action Record -01 | Active individual Internet-Draft, 25 Nov 2026 | Action lifecycle, authority re-check, approval, state commitments and outcome mapping | -01 additions are implemented in the current OAR repository or the draft is IETF-endorsed |
| Agent Action Capsule -06 | Active individual Internet-Draft, 24 Nov 2026 | Producer/action evidence, effect-attestation, chain/ledger and cross-party evidence | confirmed proves external-world state or structural cross-party references verify both signatures |

RFC 9942 and RFC 9943 are published Proposed Standards. OAR, Agent Action Capsule and the MMR profile are Internet-Drafts, not published RFCs.

## 2. OAR -01 mapping

OAR defines action-bound authority and commit-time re-check, approval bound to the exact action and what the approver saw, before/after state commitments, read-only system-of-record confirmation and outcome classes committed, refused and unresolved.

MON adopts the action lifecycle and outcome semantics, then joins OAR records to source/claim/transformation lineage, sink transaction evidence, independent readback, principal scope and bounded completeness. It must not create a competing action receipt.

OAR -01's implementation-status section says the public OAR implementation covers -00 constructions; constructions added in -01 are planned. Use the existing OAR 0.3/-00 corpus as a compatibility target and treat -01 as a version-pinned design mapping, not implemented conformance. Run maintained OAR verifiers and official vectors. The three-test MON outcome projection is not an OAR verifier.

Only OAR committed with verified system-of-record confirmation may contribute to MON's qualified digital-effect claim. Refused is a non-effect outcome. Unresolved remains unresolved, including executed-without-confirmation and not-confirmed. MON's independent-observer policy is a deployment qualification, not a universal OAR requirement.

## 3. Agent Action Capsule -06: update from -05

The latest Datatracker version is -06, published 24 November 2026. Earlier MON-009 interoperability notes cite -05; new adapter work must pin -06 and retain compatibility fixtures where required.

The -06 model adds:
- Effect status values planned, dispatched, confirmed, failed and reverted. Confirmed requires a well-formed response_digest over the actually observed response.
- effect_attestation states who vouches for execution. gate_executed is stronger than runtime_claimed and host_served_observed. The latter two are equal grades; unknown values never grade up.
- assurance independently expresses attestation, effect and ledger modes. cross_party_rung is a fourth orthogonal axis: unilateral_fallback, acknowledged_receipt or full_bilateral.
- The verifier re-derives the structural cross-party grade. The reference/correlator alone does not verify the referenced party's signature; verify the bilateral evidence substrate separately.
- provenance_mode distinguishes contemporaneous and backfilled records. A backfilled producer timestamp is not independently witnessed historical time.
- Re-anchoring supports long-term evidence across algorithm transitions, but issuer authority and whether it occurred before algorithm breakage remain outside the format's self-proof.

The -06 security section states that confirmed means response bytes observed and bound, not world-state. A spoofed upstream response can produce an honest-looking confirmed capsule. MON must join producer evidence to a destination-issued commit outcome and separately authorized readback at the exact resulting revision/digest. Missing or contradictory sink/readback evidence must not become a committed effect.

## 4. MMR profile -00

The SCITT WG MMR profile is a separate VDS profile. Its consistency receipt requires protected tree-size-2 and equality with the final proof target size. A verifier rejects a missing protected target size or mismatch.

An empty MMR inclusion path does not itself bind a leaf index. If exact position matters, the application leaf commitment must include position in a domain-separated preimage. The adapter must test deterministic protected-header CBOR, duplicate-label rejection, protected target-size equality, proof shape, accumulator reconstruction, empty-path position binding, rollback/stale trusted state and unknown VDS profiles.

Do not apply RFC 9162 SHA-256 inclusion assumptions to MMR. Keep the reported RFC 9942 RFC9162_SHA256 size/index issue scoped to that VDS profile. Distinguish root authenticity, membership, exact position, consistency and non-equivocation.

## 5. MON interoperability boundary

- OAR: action authority, approval and outcome lifecycle.
- Agent Action Capsule: producer action/disposition/effect-attestation evidence.
- RFC 9942/9943 plus selected VDS profile: receipts and transparency substrate.
- MON-010: source, claim, transformation and correction lineage.
- MON-009: destination-side conditional mutation, commit receipt and readback.
- MON-011: exact cross-layer evidence join.
- MON-012: principal/scope authorization, idempotency namespace and control-domain independence.

Every adapter returns raw profile bytes, profile/version, verified fields, verifier implementation/version, result and trust-anchor/checkpoint identity. The joiner must not trust a caller-supplied boolean that a signature or receipt was verified.

## 6. Reproducibility manifest

Each run pins the document URI and exact revision; source repository commit and vector manifest digest; verifier and cryptographic-library versions; wire profile, canonicalization and digest context; VDS/log ID, trust anchor, checkpoint and starting size; test input digest and expected/actual result; adapter mapping version; unsupported/unknown fields.

Unsupported versions/proofs, missing trust anchors, invalid checkpoints and unverifiable cross-party signatures remain unsupported or indeterminate, never best-effort success.
