# RFC 9942 receipt binding: security qualification note v1

**Status:** research qualification note; not a cryptographic verifier or RFC erratum.

## Why this gate exists

RFC 9942 is a published IETF Standards Track RFC. For the RFC 9162 SHA-256 inclusion proof profile, the protected COSE header carries the signature algorithm and VDS identifier, while the inclusion proof is carried in the unprotected header. The proof structure includes `tree-size`, `leaf-index` and the inclusion path. The signed payload is the Merkle Tree Hash/root.

That distinction matters. A valid COSE signature authenticates the signed root; it does not, by itself, authenticate every position-related value carried only in the unprotected proof structure.

A November 17, 2026 SCITT Decision Provenance Internet-Draft -03 records a concrete interoperability/security concern: the same inclusion path can be accepted with multiple tree-size/leaf-index pairs. It states that a receipt read alone establishes inclusion under the signed root, but does not establish the exact log size or position. The draft recommends binding size to authenticated signed data (for example, a signed tree head) and requiring evidence that binds or uniquely determines the occurrence's index. This is an Internet-Draft observation, not a published RFC correction; implementations must check the live RFC errata and current IETF discussion before claiming conformance.

## MON-009 qualification rule

MON-009 MUST distinguish these claims:

1. **Root authenticity:** the COSE signature validates under a trusted log key and authenticates the root payload.
2. **Membership under a root:** the inclusion proof verifies against that authenticated root under the profile's hash-tree rules.
3. **Authenticated tree size:** a separately authenticated checkpoint/tree head binds the same log identity, root and tree size. An unprotected `tree-size` value is not sufficient.
4. **Authenticated leaf position:** the verifier proves the claimed index under the authenticated tree size. It must establish that the proof uniquely determines that index under authenticated inputs, or consume an explicitly signed/profile-defined index binding.
5. **Append-only history / non-equivocation:** the verifier checks consistency against a previously persisted trusted checkpoint and, where required, independent witnesses. A single valid inclusion receipt does not establish these properties on its own.

The verifier output must expose the strongest achieved level. It MUST NOT promote root authenticity or membership-only evidence into an exact-position, append-only, timestamp, or global non-equivocation claim.

## Fail-closed outcomes

- Missing or invalid receipt signature: `invalid_signature`.
- Valid signature but no independently authenticated size checkpoint: `membership_only_position_unverified`.
- Checkpoint root or log identity mismatch: `checkpoint_mismatch`.
- Unprotected size differs from the authenticated checkpoint: `unprotected_metadata_mismatch`.
- Index is not uniquely established under the authenticated size: `position_unverified`.
- Consistency proof or witness comparison fails: `log_history_contradiction`.
- Required trust root, checkpoint, proof or witness is unavailable: `indeterminate`, never success.

A profile may choose a stricter policy and reject a membership-only receipt entirely where exact position is required. It must not silently infer the missing guarantees.

## Required adversarial vectors

The adapter conformance suite MUST include:

- mutate `tree-size` while preserving signed root and inclusion path;
- mutate `leaf-index` while preserving signed root and inclusion path;
- replay a receipt against a checkpoint with a different size or log identity;
- same root with a mismatched checkpoint signature/key;
- ambiguous index under an authenticated tree size;
- valid membership with no position claim;
- valid inclusion but failed consistency proof from the persisted checkpoint;
- split-view checkpoints presented to independent witnesses;
- checkpoint rollback, stale root and key-rotation/revocation cases.

Vectors must be generated from official RFC 9942 / RFC 9162 examples and independently reviewed implementations. Synthetic vectors are supplemental, not substitutes. Use maintained COSE and transparency-log libraries; do not invent a new receipt encoding or private header labels and call it interoperable.

## Relationship to MON-009

The destination receipt and MON-009 sink ledger establish destination-side mutation evidence. A SCITT/COSE receipt establishes transparency-service claims about signed statements. They are different evidence layers. The verifier must bind the exact sink/observation statement bytes to the registered statement, validate the receipt, and then qualify root, size, position, consistency and witness status separately.

This does not turn a transparency receipt into proof that the physical or external world reached the intended state. MON-009 still requires independently authorized readback from the destination/effect boundary.

## References

- RFC 9942, *CBOR Object Signing and Encryption (COSE) Receipts*, especially Sections 5.2 and 5.2.1: https://www.rfc-editor.org/rfc/rfc9942
- RFC 9162, *Certificate Transparency Version 2.0*: https://www.rfc-editor.org/rfc/rfc9162
- draft-lemonnier-scitt-decision-provenance-03, published November 17, 2026, especially its receipt-size discussion: https://datatracker.ietf.org/doc/html/draft-lemonnier-scitt-decision-provenance-03
- RFC 9943, *An Architecture for Trustworthy and Transparent Digital Supply Chains*: https://www.rfc-editor.org/rfc/rfc9943

**Limit:** this note specifies qualification requirements. No COSE receipt parser, cryptographic verifier, tree implementation, witness service or official-vector conformance result is claimed here.
