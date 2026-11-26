# MON transparency qualification: RFC 9942, RFC 9162 and SCITT MMR profile v1

**Status:** research qualification note. This document distinguishes VDS profiles and records reported errata; it does not amend RFC 9942 or implement a verifier.

## 1. Do not conflate the tree profiles

RFC 9942 is a published Standards Track RFC that defines COSE receipts for multiple verifiable data structures. Its RFC 9162 SHA-256 profile and the new SCITT Merkle Mountain Range (MMR) profile have different proof structures and position semantics.

The MMR profile is `draft-ietf-scitt-receipts-mmr-profile-00`, dated 21 November 2026. It is an active SCITT Working Group Internet-Draft, not an RFC.

## 2. RFC 9162 SHA-256 receipt-position qualification

For RFC 9942's RFC9162_SHA256 inclusion profile, the signed COSE payload is the Merkle root. The inclusion proof structure—including `tree-size`, `leaf-index` and proof path—is in the unprotected header.

Errata 9316 and 9317 against RFC 9942 were reported on 18 November 2026. A 25 November Internet-Draft reports that their status had not been re-verified at the time of writing. It describes an IETF 127 hackathon observation: one issued receipt was accepted for the issued tree-size/leaf-index pair and 31 other pairs among 8,255 edited pairs; the draft notes that with tree size held fixed, no other leaf index verified. The reported issue is therefore about position and size binding, not a claim that arbitrary membership proofs are forgeable.

MON must not overstate this finding as applying identically to every RFC 9942 VDS profile. The exact profile and its proof semantics must be identified first.

### RFC9162_SHA256 adapter gate

1. Verify the COSE signature against a trusted log key and exact signed root payload.
2. Independently verify the inclusion path against the candidate leaf bytes and signed root.
3. Bind the claimed tree size to a separately authenticated checkpoint/tree head for the same log identity.
4. Verify position under the authenticated size according to the exact profile; do not infer an index merely from a bare receipt unless the profile's proof computation uniquely determines it.
5. Verify consistency from a persisted trusted checkpoint, and use witnesses when the deployment policy requires non-equivocation.
6. Report membership, exact position, checkpoint consistency, freshness and witness status as separate result fields.

The reported errata are not a substitute for reading RFC 9942, current RFC Editor errata status, and the applicable IETF discussion before shipping a standards-conformance claim.

## 3. SCITT MMR receipt profile -00

The MMR profile defines an MMR VDS and proof encodings for MMR inclusion and consistency. It uses the protected `vds` identifier to identify the hash function and node value size; unsupported VDS identifiers must be rejected.

### Inclusion proof details

- An MMR inclusion proof uses an inclusion path for the leaf's peak.
- For a non-empty path, the path structure contributes to binding the leaf's location under the selected MMR.
- An empty path can represent a peak. It does not, by itself, bind one exact leaf index.
- Where an empty path is accepted and the application requires position binding, the application-level leaf value must be domain-separated from interior-node preimages; including the position in the committed leaf value is one described option.
- The profile discusses a construction where an untagged leaf may be confused with an interior node. MON adapters must test domain separation and position binding explicitly.

### Consistency proof details

The MMR profile's consistency receipt includes a protected `tree-size-2` field. A verifier must reject a consistency receipt that omits it. It must equal the final `tree-size-2` carried in the unprotected consistency proof.

The verifier must also:
- match the protected VDS identifier to its trusted VDS;
- match the first proof's `tree-size-1` to its persisted trusted tree size;
- validate every intermediate `tree-size-1` against the prior accumulator size;
- check proof-path and right-peak counts and shape against the two sizes;
- reconstruct the accumulator and verify the signature over that reconstructed detached payload;
- reject a proof that is incomplete, ambiguous or inconsistent with the stored accumulator.

The protected MMR `tree-size-2` in a consistency receipt is not a generic replacement for all log checkpoints or a tree-size field to be assumed on normal inclusion receipts. It is specifically a consistency-target binding in this draft profile.

## 4. Unified MON verifier output

A verifier should report at least:

- `profile_id` / VDS identifier and supported version;
- signed-root or signed-accumulator authenticity;
- statement/leaf digest binding and leaf serialization profile;
- membership verification;
- whether exact position is bound, and by which mechanism;
- authenticated tree size / consistency target;
- consistency from the prior trusted checkpoint;
- log identity and trust-root status;
- checkpoint age and witness/non-equivocation status.

A generic `receipt_valid=true` is insufficient for a qualification decision where exact position, consistency or freshness is required.

## 5. Required adversarial vector families

### RFC9162_SHA256

- tree-size mutation with signed root/path unchanged;
- leaf-index mutation with fixed and changed tree size;
- replay under a different log identity or authenticated checkpoint;
- signed-root mismatch, stale key and key-revocation cases;
- valid inclusion with failed consistency from the persisted checkpoint;
- split-view checkpoints and witness disagreement.

### MMR profile -00

- empty inclusion path presented as an exact-position proof;
- leaf value without domain-separation prefix vs a crafted interior-node preimage;
- missing protected consistency `tree-size-2`;
- protected/unprotected `tree-size-2` mismatch;
- wrong VDS identifier or unsupported node value size;
- first `tree-size-1` not equal to trusted size;
- incorrect path count, path length or right-peak count;
- a valid signature over a detached accumulator that was not recomputed from the proof;
- rollback, fork and trust-anchor reset cases.

Official RFC/profile vectors and independently authored implementations must be used. Synthetic mutations are supplemental.

## References

- RFC 9942, *CBOR Object Signing and Encryption (COSE) Receipts*: https://www.rfc-editor.org/rfc/rfc9942
- RFC 9162, *Certificate Transparency Version 2.0*: https://www.rfc-editor.org/rfc/rfc9162
- SCITT WG, *COSE Receipts for MMRs*, draft-ietf-scitt-receipts-mmr-profile-00 (21 Nov 2026): https://datatracker.ietf.org/doc/draft-ietf-scitt-receipts-mmr-profile/00/
- Jones, *Resolution of SCITT and W3C Verification Ambiguities via Detached Evidence Profiles*, draft-jones-scitt-pain-point-resolutions-00 (25 Nov 2026): https://datatracker.ietf.org/doc/draft-jones-scitt-pain-point-resolutions/00/

**Limit:** this qualification note maps requirements and mutation targets. No MMR implementation, RFC 9162 receipt verifier, COSE parser, transparency witness service or official-vector conformance result is claimed.
