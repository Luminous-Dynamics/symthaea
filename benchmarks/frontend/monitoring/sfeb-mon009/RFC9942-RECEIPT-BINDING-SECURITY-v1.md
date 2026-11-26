# RFC 9942 receipt binding: security qualification note v1.1

**Status:** research qualification note; not a cryptographic verifier, RFC erratum or statement that all RFC 9942 VDS profiles share the same position issue.

## 1. Profile distinction

RFC 9942 is a published IETF Standards Track RFC defining COSE receipts for multiple verifiable data structures. The relevant profile in the original MON-009 note is `RFC9162_SHA256`, whose inclusion proof carries `tree-size`, `leaf-index` and the inclusion path in the unprotected header while the COSE signature authenticates the signed Merkle root.

Do not generalize that structure to every RFC 9942 VDS profile.

## 2. Reported RFC9162_SHA256 position/size issue

Errata 9316 and 9317 against RFC 9942 were reported on 18 November 2026. The 25 November 2026 Internet-Draft `draft-jones-scitt-pain-point-resolutions-00` says that the errata status had not been re-verified at the time it was written. It describes an IETF 127 hackathon observation: one receipt was accepted for the issued tree-size/leaf-index pair and 31 other pairs among 8,255 edited pairs. The same draft notes that with tree size held fixed, no other leaf-index verified. The described issue is therefore about position and size binding, not a claim that arbitrary membership proofs are forgeable.

This is a reported standards issue and an Internet-Draft analysis, not a published RFC correction or a verified current errata disposition. Re-check the live RFC Editor errata record and IETF discussion before making a conformance claim.

## 3. MON-009 qualification rule for RFC9162_SHA256

MON-009 MUST distinguish:

1. **Root authenticity:** the COSE signature validates under a trusted log key and authenticates the signed root payload.
2. **Membership under a root:** the inclusion proof verifies against that root using the RFC9162_SHA256 rules.
3. **Authenticated size:** the claimed size is bound to an authenticated checkpoint/tree head for the same log identity. Do not treat an unprotected `tree-size` value as a signed fact.
4. **Exact position:** the claimed index is established under the authenticated size according to the applicable proof semantics; do not claim an index merely because a receipt signature validates.
5. **Append-only history / non-equivocation:** consistency is verified from persisted trusted state; independent witnesses are used when the deployment requires protection against split views.

The output MUST expose the strongest verified level. It MUST NOT promote root authenticity or membership-only evidence into exact position, append-only history, timestamp, completeness or global non-equivocation claims.

## 4. Separate MMR profile

The SCITT working group adopted `draft-ietf-scitt-receipts-mmr-profile-00` as a WG document on 21 November 2026. It is an active Internet-Draft, not a published RFC, and defines a distinct MMR VDS profile.

For that MMR profile:
- the protected `vds` identifies the hash function and node value size and must match the trusted VDS;
- an empty inclusion path can represent a peak and does not itself bind an exact leaf index; applications needing position binding must use an unambiguous domain-separated leaf construction and bind the position as required;
- a consistency receipt MUST carry protected `tree-size-2`; a missing field is rejected and its value must equal the last consistency proof's target size;
- the verifier must check the first proof's `tree-size-1` against its persisted trusted size and validate all proof shapes, path counts, right-peaks, accumulator reconstruction and the signature over the reconstructed detached payload.

The MMR protected consistency `tree-size-2` is a profile-specific consistency-target binding. It is not a generic tree-size field to be assumed on every inclusion receipt, and the MMR proof structure must not be conflated with RFC9162_SHA256.

The detailed MMR qualification requirements and mutation cases are in MON-009's `TRANSPARENCY-PROFILE-QUALIFICATION-v1.md`.

## 5. Fail-closed outcomes

- Missing/invalid COSE signature: `invalid_signature`.
- Valid signature but required checkpoint or authenticated size missing: `membership_only_position_unverified` or `indeterminate`, according to policy.
- VDS/log identity mismatch: `checkpoint_mismatch`.
- RFC9162 unprotected size/index does not match independently authenticated profile state: `unprotected_metadata_mismatch` or `position_unverified`.
- MMR protected/unprotected consistency target mismatch or missing protected target: reject the consistency receipt.
- Proof shape, consistency path, right-peaks, accumulator or witness comparison fails: `log_history_contradiction`.
- Required trust root, checkpoint, proof or witness unavailable: `indeterminate`, never success.

## 6. Required adversarial vectors

### RFC9162_SHA256

- mutate tree size while preserving signed root and inclusion path;
- mutate leaf index with fixed and changed tree size;
- replay against a checkpoint with a different log identity or tree size;
- signed-root mismatch, stale key, key rotation and revocation;
- valid membership with failed consistency from persisted checkpoint;
- split-view checkpoints presented to independent witnesses.

### MMR profile -00

- empty path presented as an exact-position proof;
- leaf value / interior-node preimage confusion without domain separation;
- missing protected consistency `tree-size-2`;
- protected/unprotected `tree-size-2` mismatch;
- wrong/unsupported VDS identifier or node value size;
- first `tree-size-1` not equal to trusted size;
- incorrect path count, path length or right-peak count;
- valid signature over a detached accumulator not recomputed from the proof;
- rollback, fork and trust-anchor reset cases.

Official vectors and independently authored implementations are mandatory for interoperability claims. Synthetic mutations are supplemental.

## References

- RFC 9942, *CBOR Object Signing and Encryption (COSE) Receipts*: https://www.rfc-editor.org/rfc/rfc9942
- RFC 9162, *Certificate Transparency Version 2.0*: https://www.rfc-editor.org/rfc/rfc9162
- Reported Errata 9316 and 9317 against RFC 9942 (reported 18 Nov 2026; status not re-verified in the cited draft): https://www.rfc-editor.org/errata_search.php?rfc=9942
- Jones, *Resolution of SCITT and W3C Verification Ambiguities via Detached Evidence Profiles*, draft-jones-scitt-pain-point-resolutions-00 (25 Nov 2026): https://datatracker.ietf.org/doc/draft-jones-scitt-pain-point-resolutions/00/
- SCITT WG, *COSE Receipts for MMRs*, draft-ietf-scitt-receipts-mmr-profile-00 (21 Nov 2026): https://datatracker.ietf.org/doc/draft-ietf-scitt-receipts-mmr-profile/00/

**Limit:** these documents specify qualification requirements. No COSE receipt parser, cryptographic verifier, RFC9162 tree implementation, MMR implementation, witness service or official-vector conformance result is claimed here.
